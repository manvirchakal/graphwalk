"""Run ``QASystem``s over questions; write ``results.json`` and ``summary.md``."""

import asyncio
import json
import math
import platform
import random
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from graphwalk.eval.metrics import Score, percentile, score
from graphwalk.eval.types import EvalQuestion, QASystem, SystemAnswer

PACKAGES = ("graphwalk", "typesafe-sdk", "litellm", "fastembed", "networkx", "numpy", "pydantic")


class Record(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: EvalQuestion
    answer: SystemAnswer
    score: Score
    linked: bool | None
    """Entry nodes used == gold entry nodes (``None`` if unknown or not applicable)."""


class Summary(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    dataset: str
    n: int
    hits1: float
    em: float
    f1: float
    latency_p50_s: float
    latency_p95_s: float
    latency_mean_s: float
    cost_per_query_usd: float | None
    """Mean provider-reported cost; ``None`` unless every query reported one."""
    decision_calls_per_query: float
    llm_calls_per_query: float
    input_tokens_per_query: float
    output_tokens_per_query: float
    status: dict[str, int]
    linking_accuracy: float | None


class SystemRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    config: dict[str, JsonValue]
    summary: Summary
    records: tuple[Record, ...]
    wall_s: float


def sample_questions(
    questions: Sequence[EvalQuestion], n: int | None, seed: int
) -> list[EvalQuestion]:
    """A seeded subset (order-stable: the same ``(n, seed)`` always gives the same set)."""
    if n is None or n >= len(questions):
        return list(questions)
    indices = sorted(random.Random(seed).sample(range(len(questions)), n))  # noqa: S311
    return [questions[i] for i in indices]


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else math.nan


def _query_cost(answer: SystemAnswer) -> float | None:
    """Reported cost; a query that made no paid calls cost exactly 0."""
    if answer.cost_usd is None and answer.decision_calls == 0 and answer.llm_calls == 0:
        return 0.0
    return answer.cost_usd


def summarize(system: str, dataset: str, records: Sequence[Record]) -> Summary:
    answers = [r.answer for r in records]
    costs = [_query_cost(a) for a in answers]
    latencies = [a.latency_s for a in answers]
    linked = [r.linked for r in records if r.linked is not None]
    return Summary(
        system=system,
        dataset=dataset,
        n=len(records),
        hits1=_mean([r.score.hits1 for r in records]),
        em=_mean([r.score.em for r in records]),
        f1=_mean([r.score.f1 for r in records]),
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        latency_mean_s=_mean(latencies),
        cost_per_query_usd=(
            _mean([c for c in costs if c is not None])
            if costs and all(c is not None for c in costs)
            else None
        ),
        decision_calls_per_query=_mean([a.decision_calls for a in answers]),
        llm_calls_per_query=_mean([a.llm_calls for a in answers]),
        input_tokens_per_query=_mean([a.input_tokens for a in answers]),
        output_tokens_per_query=_mean([a.output_tokens for a in answers]),
        status=dict(Counter(a.status for a in answers)),
        linking_accuracy=_mean([float(x) for x in linked]) if linked else None,
    )


async def run_system(
    system: QASystem,
    questions: Sequence[EvalQuestion],
    *,
    concurrency: int = 8,
    on_done: Callable[[int, int], None] | None = None,
) -> SystemRun:
    """Answer every question (at most ``concurrency`` at once); exceptions from a system
    are recorded as ``status="error"`` answers rather than aborting the run."""
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def one(question: EvalQuestion) -> Record:
        nonlocal done
        async with semaphore:
            started = time.perf_counter()
            try:
                answer = await system.answer(question)
            except Exception as error:  # noqa: BLE001 - record and continue
                answer = SystemAnswer(
                    answers=(),
                    answer_set=(),
                    status="error",
                    error=f"{type(error).__name__}: {error}",
                    latency_s=time.perf_counter() - started,
                )
        done += 1
        if on_done is not None:
            on_done(done, len(questions))
        linked = None
        if question.gold_start is not None and answer.start is not None:
            linked = set(answer.start) == set(question.gold_start)
        return Record(
            question=question,
            answer=answer,
            score=score(answer.answers, answer.answer_set, question.answers, question.kind),
            linked=linked,
        )

    started = time.perf_counter()
    records = await asyncio.gather(*(one(q) for q in questions))
    dataset = questions[0].dataset if questions else "empty"
    return SystemRun(
        system=system.name,
        config=system.describe(),
        summary=summarize(system.name, dataset, records),
        records=tuple(records),
        wall_s=time.perf_counter() - started,
    )


def environment() -> dict[str, JsonValue]:
    versions: dict[str, JsonValue] = {}
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - fixed argv
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        sha, dirty = None, None
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "packages": versions,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def summary_table(runs: Sequence[SystemRun]) -> str:
    header = (
        "| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | "
        "Jev calls/q | LLM calls/q | in tok/q | status | linking |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows: list[str] = []
    for run in runs:
        s = run.summary
        cost = s.cost_per_query_usd
        status = ", ".join(f"{k}={v}" for k, v in sorted(s.status.items()))
        rows.append(
            f"| {s.system} | {s.n} | {_fmt(s.hits1)} | {_fmt(s.em)} | {_fmt(s.f1)} "
            f"| {_fmt(s.latency_p50_s, 2)} | {_fmt(s.latency_p95_s, 2)} "
            f"| {_fmt(cost, 6)} | {_fmt(None if cost is None else cost * 1000, 3)} "
            f"| {_fmt(s.decision_calls_per_query, 2)} | {_fmt(s.llm_calls_per_query, 2)} "
            f"| {_fmt(s.input_tokens_per_query, 0)} | {status} | {_fmt(s.linking_accuracy)} |"
        )
    return header + "\n".join(rows) + "\n"


def write_results(
    out_dir: Path,
    *,
    dataset: str,
    runs: Sequence[SystemRun],
    params: dict[str, JsonValue],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    document = {
        "dataset": dataset,
        "params": params,
        "environment": env,
        "runs": [run.model_dump(mode="json") for run in runs],
    }
    (out_dir / "results.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    _write_summary(out_dir, dataset, runs, params, env)
    return out_dir


def _write_summary(
    out_dir: Path,
    dataset: str,
    runs: Sequence[SystemRun],
    params: dict[str, JsonValue],
    env: dict[str, JsonValue],
) -> None:
    lines = [
        f"# {dataset}",
        "",
        f"- params: `{json.dumps(params, sort_keys=True)}`",
        f"- git: `{env['git_sha']}`{' (dirty)' if env['git_dirty'] else ''}",
        f"- run at: {env['timestamp']}",
        "",
        summary_table(runs),
        "## Systems",
        "",
    ]
    for run in runs:
        lines.append(f"- **{run.system}** (wall {run.wall_s:.0f}s): `{json.dumps(run.config)}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resummarize(out_dir: Path) -> list[SystemRun]:
    """Recompute summaries from a run's stored records (e.g. after a metric fix) and
    rewrite ``results.json`` and ``summary.md``. Params and environment are kept, and
    the rewrite is recorded under ``environment.resummarized_at``."""
    document = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    runs: list[SystemRun] = []
    for raw in document["runs"]:
        run = SystemRun.model_validate(raw)
        records = [
            r.model_copy(
                update={
                    "score": score(
                        r.answer.answers, r.answer.answer_set, r.question.answers, r.question.kind
                    )
                }
            )
            for r in run.records
        ]
        summary = summarize(run.system, run.summary.dataset, records)
        runs.append(run.model_copy(update={"summary": summary, "records": tuple(records)}))
    document["runs"] = [run.model_dump(mode="json") for run in runs]
    document["environment"]["resummarized_at"] = datetime.now(UTC).isoformat()
    (out_dir / "results.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    _write_summary(out_dir, document["dataset"], runs, document["params"], document["environment"])
    return runs
