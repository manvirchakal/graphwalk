"""The traversal engine: greedy, beam, and sample walks driven by a ``DecisionBackend``.

One algorithm serves all three strategies. At each depth every active hypothesis
("beam") is turned into one choice question: its legal moves plus STOP. All questions
of a depth are sent together (``per_depth``; Jev shares ``state`` per call) or one call
per beam (``per_beam``). Then:

* **beam**: every option becomes a candidate. Candidates and already-finished beams
  compete for ``beam_width`` slots by length-normalized score (TypeSafe's hierarchical
  classification recipe). The walk ends when every surviving beam has finished.
  **greedy** is beam with width 1.
* **sample**: each of ``n_samples`` independent walks draws one option from the
  temperature/top-p reshaped distribution with its own seeded RNG. Answers are voted.

A move to a node the beam already visited is not offered (``exclude_visited``). A beam
with no legal move and STOP ends as a ``dead_end``; one legal option is taken without a
call and is not counted as a decision.
"""

import asyncio
import logging
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic import JsonValue

from graphwalk.core.errors import NodeNotFoundError
from graphwalk.core.hashing import content_hash
from graphwalk.core.model import Node, NodeId
from graphwalk.decisions.base import (
    ChoiceQuestion,
    ChoiceResult,
    DecisionBackend,
    DecisionBackendError,
    DecisionRequest,
    JSONContent,
)
from graphwalk.embeddings.base import Embedder, Vectors
from graphwalk.stores.base import GraphStore
from graphwalk.traversal import prompts
from graphwalk.traversal.batching import split_questions
from graphwalk.traversal.config import TraversalConfig
from graphwalk.traversal.prefilter import EmbeddingCache, rank_by_similarity
from graphwalk.traversal.prompts import STOP, Move
from graphwalk.traversal.sampling import draw, reshape
from graphwalk.traversal.scoring import normalized_score, step_logp
from graphwalk.traversal.trace import (
    Answer,
    Call,
    Hop,
    OptionInfo,
    Pruned,
    Step,
    Termination,
    Totals,
    Trace,
    TraversalResult,
)

logger = logging.getLogger(__name__)


class _Abort(Exception):  # noqa: N818 - internal control flow, never escapes
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _Beam:
    beam_id: int
    parent_id: int | None
    order: int
    """Creation order; the final tie-break, so rankings are deterministic."""
    walk: int
    """Sample strategy: which independent walk this beam belongs to."""
    start: tuple[NodeId, ...]
    frontier: tuple[NodeId, ...]
    path: tuple[Hop, ...] = ()
    path_text: tuple[str, ...] = ()
    visited: frozenset[NodeId] = frozenset()
    sum_logp: float = 0.0
    n_decisions: int = 0
    min_confidence: float | None = None
    finished: Termination | None = None


@dataclass(frozen=True)
class _Option:
    label: str
    move: Move | None  # None = STOP
    description: JSONContent | None


@dataclass
class _Plan:
    beam: _Beam
    options: list[_Option]
    pruned: list[Pruned]
    question: ChoiceQuestion | None = None


@dataclass
class _Run:
    query: str
    state: dict[str, JsonValue]
    started: float
    beams: list["_Beam"] = field(default_factory=list["_Beam"])
    """The current hypotheses, kept up to date so an abort can report them."""
    nodes: dict[NodeId, Node] = field(default_factory=dict[NodeId, Node])
    steps: list[Step] = field(default_factory=list[Step])
    calls: list[Call] = field(default_factory=list[Call])
    rngs: dict[int, random.Random] = field(default_factory=dict[int, random.Random])
    answer_type: dict[str, float] | None = None
    """Distribution from the speculative answer-type question (``answer_type != "off"``)."""
    next_id: int = 0
    depth_reached: int = 0

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1


def _ended(beam: _Beam) -> bool:
    """The walk ended by itself rather than being cut off."""
    return beam.finished in ("stop", "dead_end")


class Traverser:
    """Answers queries by walking a :class:`GraphStore` with a :class:`DecisionBackend`."""

    def __init__(
        self,
        store: GraphStore,
        decider: DecisionBackend,
        *,
        embedder: Embedder | None = None,
        config: TraversalConfig | None = None,
        node_types: Sequence[str] | None = None,
    ) -> None:
        self.store = store
        self.decider = decider
        self.config = config or TraversalConfig()
        self.node_types = tuple(dict.fromkeys(node_types or ()))
        if self.config.answer_type != "off" and len(self.node_types) < 2:  # noqa: PLR2004
            # An untyped graph has nothing to predict; run without the question.
            logger.warning(
                "answer_type=%r needs at least 2 node types, got %s; turning it off",
                self.config.answer_type,
                list(self.node_types),
            )
            self.config = self.config.model_copy(update={"answer_type": "off"})
        elif self.config.answer_type != "off" and len(self.node_types) > decider.max_options:
            # E.g. a graph extracted by an LLM, with hundreds of free-form types.
            logger.warning(
                "answer_type=%r: %d node types exceed the decider's %d options; turning it "
                "off (pass fewer node_types to keep it)",
                self.config.answer_type,
                len(self.node_types),
                decider.max_options,
            )
            self.config = self.config.model_copy(update={"answer_type": "off"})
        self._cache = None if embedder is None else EmbeddingCache(embedder)
        if decider.max_options < 2:  # noqa: PLR2004 - STOP plus one move
            msg = "decision backend must allow at least 2 options"
            raise ValueError(msg)

    def preload_embeddings(self, texts: Sequence[str], vectors: Vectors) -> None:
        """Seed the prefilter cache, typically with ``prompts.node_text`` of every node, so
        high-degree steps cost dot products instead of embedding calls."""
        if self._cache is None:
            msg = "no embedder configured"
            raise ValueError(msg)
        self._cache.preload(texts, vectors)

    @property
    def embedder(self) -> Embedder | None:
        return None if self._cache is None else self._cache.embedder

    async def traverse(self, query: str, start: Sequence[NodeId]) -> TraversalResult:
        """Walk from ``start`` to answer ``query``. Never raises for budget or backend
        failures: those return ``status="aborted"``/``"error"`` with partial answers.

        Raises :class:`NodeNotFoundError` if a start node does not exist, and
        ``ValueError`` if ``start`` is empty.
        """
        start_ids = tuple(dict.fromkeys(start))
        if not start_ids:
            msg = "at least one start node is required"
            raise ValueError(msg)
        found = await self.store.get_nodes(start_ids)
        missing = [node_id for node_id in start_ids if node_id not in found]
        if missing:
            raise NodeNotFoundError(missing[0])
        run = _Run(query=query, state=prompts.state_for(query), started=time.perf_counter())
        run.nodes.update(found)
        beams = self._initial_beams(run, start_ids)
        archive: dict[int, _Beam] = {}
        status: Literal["ok", "aborted", "error"] = "ok"
        abort_reason: str | None = None
        error: str | None = None
        try:
            beams = await self._search(run, beams, archive)
        except _Abort as abort:
            status, abort_reason = "aborted", abort.reason
            beams = [b if b.finished else replace(b, finished="aborted") for b in run.beams]
        except DecisionBackendError as exc:
            logger.warning("traversal failed: %s", exc)
            status, error = "error", str(exc)
            beams = [b if b.finished else replace(b, finished="aborted") for b in run.beams]
        for beam in beams:
            archive[beam.beam_id] = beam
        answers = self._answers(run, list(archive.values()))
        return TraversalResult(
            status=status,
            answers=tuple(answers),
            abort_reason=abort_reason,
            error=error,
            trace=self._trace(run, start_ids),
        )

    # -- search ---------------------------------------------------------------------------

    def _initial_beams(self, run: _Run, start: tuple[NodeId, ...]) -> list[_Beam]:
        cfg = self.config
        seeds = [start] if cfg.hop_mode == "relation" else [(node_id,) for node_id in start]
        if cfg.strategy == "sample":
            seeds = [seeds[i % len(seeds)] for i in range(cfg.n_samples)]
        beams: list[_Beam] = []
        for walk, frontier in enumerate(seeds):
            beam_id = run.new_id()
            beams.append(
                _Beam(
                    beam_id=beam_id,
                    parent_id=None,
                    order=beam_id,
                    walk=walk,
                    start=frontier,
                    frontier=frontier,
                    visited=frozenset(frontier),
                )
            )
            if cfg.strategy == "sample":
                material = f"{cfg.seed}\x1f{content_hash(run.query)}\x1f{walk}"
                run.rngs[walk] = random.Random(content_hash(material))  # noqa: S311
        return beams

    async def _search(
        self, run: _Run, beams: list[_Beam], archive: dict[int, _Beam]
    ) -> list[_Beam]:
        cfg = self.config
        run.beams = beams
        for depth in range(cfg.budget.max_depth):
            active = [b for b in beams if b.finished is None]
            if not active:
                break
            plans = [await self._plan(run, beam, depth) for beam in active]
            results = await self._ask(run, depth, [p.question for p in plans if p.question])
            candidates: list[_Beam] = []
            for plan in plans:
                candidates.extend(self._expand(run, plan, depth, results))
            finished = [b for b in beams if b.finished is not None]
            if cfg.strategy == "sample":
                beams = finished + candidates
            else:
                pool = sorted(finished + candidates, key=self._rank_key)
                beams = pool[: cfg.width]
                for beam in beams:
                    if beam.finished is not None:
                        archive[beam.beam_id] = beam
            run.beams = beams
            run.depth_reached = depth + 1
            limit = cfg.budget.max_input_tokens
            if limit is not None and sum(c.input_tokens for c in run.calls) > limit:
                raise _Abort(f"max_input_tokens ({limit}) exceeded")
        return [b if b.finished else replace(b, finished="max_depth") for b in beams]

    def _rank_key(self, beam: _Beam) -> tuple[float, int]:
        return (-self._score(beam), beam.order)

    def _score(self, beam: _Beam) -> float:
        return normalized_score(beam.sum_logp, beam.n_decisions, self.config.length_alpha)

    async def _plan(self, run: _Run, beam: _Beam, depth: int) -> _Plan:
        cfg = self.config
        moves, pruned = await self._moves(run, beam)
        frontier_nodes = [run.nodes[n] for n in beam.frontier]
        moves, more_pruned = await self._cap_options(run, beam, moves)
        pruned.extend(more_pruned)
        if cfg.hop_mode == "entity":
            names: list[str | None] = [run.nodes[m.targets[0]].name for m in moves]
        else:
            names = [None] * len(moves)
        labels = prompts.make_labels(cfg.label_style, moves, names)
        popts = self._prompt_options(run, depth)
        options: list[_Option] = []
        for label, move in zip(labels, moves, strict=True):
            if cfg.hop_mode == "entity":
                description: JSONContent = prompts.entity_option(
                    frontier_nodes[0], move, run.nodes[move.targets[0]], popts
                )
            else:
                targets = [run.nodes[t] for t in move.targets]
                description = prompts.relation_option(move, targets, popts)
            options.append(_Option(label=label, move=move, description=description))
        if (depth > 0 or cfg.allow_stop_at_start) and not self._gated(run, frontier_nodes):
            options.append(
                _Option(
                    label=STOP,
                    move=None,
                    description=prompts.stop_description(cfg.hop_mode, popts),
                )
            )
        plan = _Plan(beam=beam, options=options, pruned=pruned)
        if len(options) >= 2:  # noqa: PLR2004 - a real decision
            plan.question = ChoiceQuestion(
                key=f"b{beam.beam_id}",
                instructions=prompts.instructions_for(
                    cfg.hop_mode, frontier_nodes, beam.path_text, popts
                ),
                options={o.label: o.description for o in options},
            )
        return plan

    def _expected_type(self, run: _Run) -> tuple[str, float] | None:
        dist = run.answer_type
        if dist is None:
            return None
        best = max(dist, key=dist.__getitem__)  # ties: first type offered
        return best, dist[best]

    def _prompt_options(self, run: _Run, depth: int) -> prompts.PromptOptions:
        cfg = self.config
        expected = self._expected_type(run) if depth > 0 else None
        return prompts.PromptOptions(
            include_summary=cfg.include_summaries,
            preview=cfg.frontier_preview,
            show_types=cfg.show_types,
            stop_style=cfg.stop_style,
            relation_glosses=cfg.relation_glosses or None,
            expected_answer_type=None if expected is None else expected[0],
        )

    def _gated(self, run: _Run, frontier: Sequence[Node]) -> bool:
        """``answer_type="gate"``: withhold STOP when no current node has the confidently
        predicted answer type."""
        if self.config.answer_type != "gate":
            return False
        expected = self._expected_type(run)
        if expected is None or expected[1] < self.config.answer_type_gate_min_p:
            return False
        return all(node.type != expected[0] for node in frontier)

    async def _moves(self, run: _Run, beam: _Beam) -> tuple[list[Move], list[Pruned]]:
        cfg = self.config
        grouped: dict[tuple[str, Literal["out", "in"]], list[NodeId]] = defaultdict(list)
        moves: list[Move] = []
        for node_id in beam.frontier:
            for neighbor in await self.store.neighbors(node_id, direction=cfg.direction):
                target = neighbor.node.id
                if cfg.exclude_visited and target in beam.visited:
                    continue
                run.nodes.setdefault(target, neighbor.node)
                if cfg.hop_mode == "entity":
                    moves.append(Move(neighbor.edge.type, neighbor.direction, (target,)))
                else:
                    targets = grouped[(neighbor.edge.type, neighbor.direction)]
                    if target not in targets:
                        targets.append(target)
        pruned: list[Pruned] = []
        if cfg.hop_mode == "relation":
            for (relation, direction), targets in grouped.items():
                kept, dropped = await self._cap_frontier(run, targets)
                if dropped:
                    reason = "frontier_prefilter" if self._cache else "frontier_truncated"
                    pruned.append(Pruned(reason=reason, relation=relation, targets=tuple(dropped)))
                moves.append(Move(relation, direction, tuple(kept)))
            moves.sort(key=lambda m: (m.direction != "out", m.relation))
        return moves, pruned

    async def _cap_frontier(
        self, run: _Run, targets: list[NodeId]
    ) -> tuple[list[NodeId], list[NodeId]]:
        cap = self.config.max_frontier
        if len(targets) <= cap:
            return targets, []
        if self._cache is None:
            return targets[:cap], targets[cap:]
        texts = [prompts.node_text(run.nodes[t]) for t in targets]
        ranked = await rank_by_similarity(self._cache, run.query, texts)
        keep = sorted(i for i, _ in ranked[:cap])
        keep_set = set(keep)
        return [targets[i] for i in keep], [t for i, t in enumerate(targets) if i not in keep_set]

    async def _cap_options(
        self, run: _Run, beam: _Beam, moves: list[Move]
    ) -> tuple[list[Move], list[Pruned]]:
        """Prefilter high-degree option sets; always respect the backend's option limit."""
        cfg = self.config
        hard_cap = self.decider.max_options - 1  # one slot for STOP
        if self._cache is not None and len(moves) > cfg.prefilter_threshold:
            cap = min(cfg.prefilter_top_n, hard_cap)
            texts = [self._move_text(run, move) for move in moves]
            ranked = await rank_by_similarity(self._cache, run.query, texts)
            keep = {i for i, _ in ranked[:cap]}
            pruned = [
                Pruned(
                    reason="prefilter",
                    relation=moves[i].relation,
                    targets=moves[i].targets,
                    similarity=sim,
                )
                for i, sim in ranked[cap:]
            ]
            return [m for i, m in enumerate(moves) if i in keep], pruned
        if len(moves) > hard_cap:
            logger.warning(
                "beam %d: %d options exceed the backend limit; truncating (no embedder)",
                beam.beam_id,
                len(moves),
            )
            pruned = [
                Pruned(reason="truncated", relation=m.relation, targets=m.targets)
                for m in moves[hard_cap:]
            ]
            return moves[:hard_cap], pruned
        return moves, []

    def _move_text(self, run: _Run, move: Move) -> str:
        # Entity mode ranks by the target node's text alone, so vectors can be precomputed
        # once per graph (see preload_embeddings) instead of embedded per step.
        if self.config.hop_mode == "entity":
            return prompts.node_text(run.nodes[move.targets[0]])
        return move.relation

    async def _ask(
        self, run: _Run, depth: int, questions: list[ChoiceQuestion]
    ) -> dict[str, tuple[ChoiceResult, int]]:
        """Send the depth's questions; return key -> (result, call_id)."""
        cfg = self.config
        if depth == 0 and cfg.answer_type != "off":
            # Speculative fan-out: rides along with the first hop, so it adds no latency.
            questions = [
                ChoiceQuestion(
                    key=prompts.ANSWER_TYPE_KEY,
                    instructions=prompts.ANSWER_TYPE_TASK,
                    options=dict(prompts.answer_type_options(self.node_types)),
                ),
                *questions,
            ]
        if not questions:
            return {}
        if cfg.beam_batching == "per_beam":
            groups = [[q] for q in questions]
        else:
            groups = split_questions(
                run.state,
                questions,
                max_request_tokens=cfg.max_request_tokens,
                max_state_plus_question_tokens=cfg.max_state_plus_question_tokens,
                safety_margin=cfg.token_safety_margin,
            )
        limit = cfg.budget.max_decision_calls
        if limit is not None and len(run.calls) + len(groups) > limit:
            raise _Abort(f"max_decision_calls ({limit}) would be exceeded")
        state: JSONContent = run.state
        responses = await asyncio.gather(
            *(
                self.decider.decide(DecisionRequest(state=state, questions=tuple(group)))
                for group in groups
            )
        )
        out: dict[str, tuple[ChoiceResult, int]] = {}
        for group, response in zip(groups, responses, strict=True):
            call_id = len(run.calls)
            run.calls.append(
                Call(
                    call_id=call_id,
                    depth=depth,
                    n_questions=len(group),
                    latency_s=response.latency_s,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cost_usd=response.usage.cost_usd,
                    request_id=response.request_id,
                    model=response.model,
                )
            )
            for question in group:
                result = response.results.get(question.key)
                if result is None:
                    msg = f"backend returned no result for question {question.key!r}"
                    raise DecisionBackendError(msg)
                out[question.key] = (result, call_id)
        answer_type = out.pop(prompts.ANSWER_TYPE_KEY, None)
        if answer_type is not None:
            run.answer_type = dict(answer_type[0].probabilities)
        return out

    def _expand(
        self,
        run: _Run,
        plan: _Plan,
        depth: int,
        results: dict[str, tuple[ChoiceResult, int]],
    ) -> list[_Beam]:
        cfg = self.config
        beam = plan.beam
        by_label = {o.label: o for o in plan.options}
        option_infos = tuple(self._option_info(o) for o in plan.options)
        if plan.question is None:  # forced: zero or one legal option
            chosen = plan.options[0] if plan.options else None
            run.steps.append(
                Step(
                    depth=depth,
                    beam_id=beam.beam_id,
                    parent_beam_id=beam.parent_id,
                    frontier=beam.frontier,
                    options=option_infos,
                    pruned=tuple(plan.pruned),
                    call_id=None,
                    chosen=() if chosen is None else (chosen.label,),
                )
            )
            if chosen is None or chosen.move is None:
                # Counted as a certain STOP (p = 1), so a walk that ends at a leaf is scored
                # like one that chose STOP there; otherwise mean log-prob would penalize it.
                return [
                    self._child(
                        run, beam, None, logp=0.0, confidence=None, end="dead_end", decision=True
                    )
                ]
            return [self._child(run, beam, chosen.move, logp=0.0, confidence=None)]
        result, call_id = results[plan.question.key]
        used = result.probabilities
        if cfg.strategy == "sample":
            used = reshape(used, temperature=cfg.temperature, top_p=cfg.top_p)
            labels = [draw(used, run.rngs[beam.walk])]
        elif cfg.strategy == "greedy":
            labels = [result.top]
        else:
            labels = list(result.probabilities)
        run.steps.append(
            Step(
                depth=depth,
                beam_id=beam.beam_id,
                parent_beam_id=beam.parent_id,
                frontier=beam.frontier,
                options=option_infos,
                pruned=tuple(plan.pruned),
                call_id=call_id,
                question_key=plan.question.key,
                distribution=dict(result.probabilities),
                distribution_used=dict(used),
                confidence=result.confidence,
                chosen=tuple(labels),
            )
        )
        children: list[_Beam] = []
        for label in labels:
            option = by_label[label]
            logp = step_logp(result.probabilities[label], cfg.prob_floor)
            end: Termination | None = "stop" if option.move is None else None
            children.append(
                self._child(
                    run, beam, option.move, logp=logp, confidence=result.confidence, end=end
                )
            )
        return children

    def _child(
        self,
        run: _Run,
        beam: _Beam,
        move: Move | None,
        *,
        logp: float,
        confidence: float | None,
        end: Termination | None = None,
        decision: bool | None = None,
    ) -> _Beam:
        """Extend ``beam`` by ``move`` (``None`` = stop here). ``decision`` defaults to
        whether a backend answer (with a confidence) was involved."""
        beam_id = run.new_id()
        decided = confidence is not None if decision is None else decision
        min_conf = beam.min_confidence
        if confidence is not None:
            min_conf = confidence if min_conf is None else min(min_conf, confidence)
        child = replace(
            beam,
            beam_id=beam_id,
            parent_id=beam.beam_id,
            order=beam_id,
            sum_logp=beam.sum_logp + logp,
            n_decisions=beam.n_decisions + int(decided),
            min_confidence=min_conf,
            finished=end,
        )
        if move is None:
            return child
        names = [run.nodes[t].name for t in move.targets]
        if self.config.hop_mode == "entity":
            text = prompts.edge_text(
                run.nodes[beam.frontier[0]].name, move.relation, move.direction, names[0]
            )
        else:
            here = prompts.names_text([run.nodes[n].name for n in beam.frontier], 3)
            there = prompts.names_text(names, 3)
            text = (
                f"{here} --{move.relation}--> {there}"
                if move.direction == "out"
                else (f"{there} --{move.relation}--> {here}")
            )
        return replace(
            child,
            frontier=move.targets,
            path=(
                *beam.path,
                Hop(relation=move.relation, direction=move.direction, targets=move.targets),
            ),
            path_text=(*beam.path_text, text),
            visited=beam.visited | set(move.targets),
        )

    @staticmethod
    def _option_info(option: _Option) -> OptionInfo:
        if option.move is None:
            return OptionInfo(label=option.label, kind="stop", description=option.description)
        return OptionInfo(
            label=option.label,
            kind="move",
            relation=option.move.relation,
            direction=option.move.direction,
            targets=option.move.targets,
            description=option.description,
        )

    # -- results --------------------------------------------------------------------------

    def _answers(self, run: _Run, beams: list[_Beam]) -> list[Answer]:
        """Group hypotheses by answer set. Walks that ended on their own (STOP chosen, or a
        dead end) rank first; then by votes (sample) or score, then creation order."""
        groups: dict[tuple[NodeId, ...], list[_Beam]] = defaultdict(list)
        for beam in beams:
            groups[tuple(sorted(beam.frontier))].append(beam)
        answers: list[tuple[tuple[int, int, float, int], Answer]] = []
        for members in groups.values():
            best = min(members, key=lambda b: (not _ended(b), *self._rank_key(b)))
            votes = len(members)
            answer = Answer(
                beam_id=best.beam_id,
                node_ids=best.frontier,
                names=tuple(run.nodes[n].name for n in best.frontier),
                start=best.start,
                path=best.path,
                score=self._score(best),
                sum_logp=best.sum_logp,
                n_decisions=best.n_decisions,
                min_confidence=best.min_confidence,
                terminated_by=best.finished or "max_depth",
                votes=votes,
            )
            stopped = int(not _ended(best))
            vote_key = -votes if self.config.strategy == "sample" else 0
            answers.append(((stopped, vote_key, -answer.score, best.order), answer))
        answers.sort(key=lambda item: item[0])
        return [answer for _, answer in answers]

    def _trace(self, run: _Run, start: tuple[NodeId, ...]) -> Trace:
        calls = run.calls
        costs = [c.cost_usd for c in calls if c.cost_usd is not None]
        totals = Totals(
            decision_calls=len(calls),
            questions=sum(c.n_questions for c in calls),
            input_tokens=sum(c.input_tokens for c in calls),
            output_tokens=sum(c.output_tokens for c in calls),
            cost_usd=sum(costs) if costs else (0.0 if not calls else None),
            decision_latency_s=sum(c.latency_s for c in calls),
            wall_s=time.perf_counter() - run.started,
            depth_reached=run.depth_reached,
        )
        embedder = self.embedder
        return Trace(
            query=run.query,
            config=self.config.model_dump(mode="json"),
            decision_model=self.decider.model_id,
            embedder_model=None if embedder is None else embedder.model_id,
            start=start,
            steps=tuple(run.steps),
            calls=tuple(calls),
            answer_type=run.answer_type,
            totals=totals,
        )
