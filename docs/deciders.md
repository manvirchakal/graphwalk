# Deciders

graphwalk turns each hop of a walk into a classification: the current node's relations
(plus `STOP`) are the options, and a **decider** returns a probability for each. The
walk's confidence is built from those probabilities, so it is only as meaningful as
they are. graphwalk provides the machinery (options, walks, confidence, escalation, the
MCP tools); the decider is yours to choose.

## Built-in deciders

| `GRAPHWALK_DECIDER` | What decides | Confidence | Speed (measured) |
|---|---|---|---|
| `jev` (default) | TypeSafe's Jev, a classification model | Informative (AUROC 0.92–0.97 on MetaQA 2–3 hop and WebQSP) | Fastest: 0.7–1.1 s per MetaQA walk |
| `logprob` | Any chat model that returns token log-probabilities, at any OpenAI-compatible endpoint | Depends on the model: as informative as Jev's with Qwen3.8-27B (P1); informative but weaker with Gemma 4 31B and DeepSeek V4 Flash | ~10× Jev's latency through OpenRouter; local serving not measured |
| `llm` | A chat model that *states* a score per option | Barely informative (AUROC 0.50–0.69): scores, not probabilities | Slowest |

The evidence is in [P1](results-paper.md#p1-is-the-confidence-jevs-or-the-framings):
the same walks with Qwen3.8-27B reading its own token probabilities matched Jev's
accuracy and the informativeness of its confidence on every dataset tried, and beat it
on 2Wiki and CWQ. **The model matters**, though (MetaQA 3-hop):

| decider | EM | AUROC of confidence |
|---|---|---|
| Jev | 0.80 | 0.95 |
| Qwen3.8-27B | 0.80 | 0.96 |
| Gemma 4 31B | 0.82 | 0.82 (overconfident) |
| DeepSeek V4 Flash | 0.44 | 0.84 |

What Jev adds is speed, and not having to choose. With `logprob`, start from
Qwen3.8-27B (the default) and check any other model on labeled questions.

## The `logprob` decider

The model sees the options lettered A, B, C, …, answers with one letter, and the
distribution is read from that token's `top_logprobs` (one call per question,
`max_tokens=1`, temperature 0). It takes at most 20 options per question: wider nodes
are ranked down to 20, by the embedder if the index has one, otherwise by the words they
share with the question. Pass an embedder (`Index.open(..., embedder=...)`, the
`embeddings` extra) on graphs with many relations per node.

=== "OpenRouter (default)"

    ```bash
    export OPENROUTER_API_KEY=sk-or-...
    export GRAPHWALK_DECIDER=logprob                       # default model: qwen/qwen3.8-27b
    export GRAPHWALK_DECIDER_MODEL=deepseek/deepseek-v4-flash   # optional
    ```

=== "Your own server (vLLM, llama.cpp, SGLang)"

    ```bash
    vllm serve <model> --port 8000                         # any OpenAI-compatible server
    export GRAPHWALK_DECIDER=logprob
    export GRAPHWALK_DECIDER_BASE_URL=http://localhost:8000/v1
    export GRAPHWALK_DECIDER_MODEL=<model>
    # GRAPHWALK_DECIDER_API_KEY=... if the server wants one
    ```

=== "Python"

    ```python
    from graphwalk import Index, LogprobDecider

    decider = LogprobDecider("<model>", base_url="http://localhost:8000/v1")
    async with Index.open("kg.db", decider=decider) as index:
        result = await index.walk("Where was the director of Inception born?")
    ```

Things to know:

- **The endpoint must return log-probabilities.** One that answers without them raises
  an error instead of deciding blindly. Through OpenRouter, providers differ for the
  same model, and some silently drop the parameters: graphwalk asks for providers that
  honor every parameter and retries when a reply has none, but a model with no such
  provider will fail. In a spot check through OpenRouter (October 2026), Qwen3.8-27B,
  Gemma 4 31B, DeepSeek V4 Flash, Granite 4.2 8B, and Nemotron 3.5 Lightning returned
  them; a model whose endpoints require reasoning (GLM 5.3 Flash) did not.
- **Reasoning models must answer without reasoning.** Through OpenRouter, graphwalk
  turns reasoning off; endpoints that make reasoning mandatory cannot serve this decider.
- **Only Qwen3.8-27B was evaluated in full** (P1, six datasets); two other models were
  checked on one (above), and one of them failed. Check a different model on 50–100
  labeled questions from your graph before relying on it.

As a fallback or for escalation only:

```bash
export GRAPHWALK_DECISION_FALLBACK=logprob     # Jev when its key is set, else logprob
export GRAPHWALK_ESCALATION_DECIDER=logprob    # escalated walks also carry probabilities
```

## Bring your own decider

Anything with this shape is a decider (`graphwalk.DecisionBackend`, a protocol: no
subclassing needed):

```python
from graphwalk import (
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    Usage,
    normalize_distribution,
)


class MyDecider:
    model_id = "my-classifier-v1"   # recorded in every trace
    max_options = 64                # graphwalk ranks wider nodes down to this

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        results = {}
        for question in request.questions:
            # question.options: label -> description; request.state: the walk so far.
            scores = await my_model_probabilities(request.state, question)  # yours
            results[question.key] = normalize_distribution(question, scores)
        return DecisionResponse(
            results=results,
            usage=Usage(input_tokens=0, output_tokens=0, cost_usd=None),
            latency_s=0.0,
            model=self.model_id,
        )

    async def aclose(self) -> None:
        pass


async with Index.open("kg.db", decider=MyDecider()) as index:
    ...
```

The contract: return a probability for **every offered label** of each question
(`normalize_distribution` validates and renormalizes), and make them your model's real
probabilities. The confidence graphwalk reports, and any escalation threshold, mean
only what your probabilities mean. A fine-tuned classifier, a cross-encoder over
(question, path, relation), or a local LLM's token probabilities all fit.
`graphwalk.decisions.FakeDecisionBackend` is a scripted decider for tests.
