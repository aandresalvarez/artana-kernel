# Examples

Run examples from the repository root.

## 01 - Durable Chat Replay

Demonstrates:
- tenant context + middleware
- capability-scoped tool execution
- durable event log in SQLite
- replay-safe model/tool behavior on repeated `chat` with the same `run_id`

Run:

```bash
uv run python examples/01_durable_chat_replay.py
```

## 02 - Real LiteLLM Chat (OpenAI)

Uses `LiteLLMAdapter` with a real model call.
This example uses kernel-issued `run_id` and proves replay invariants on a second call.

Run:

```bash
set -a; source .env; set +a
uv run python examples/02_real_litellm_chat.py
```

## 03 - Fact Extraction (Triplets)

Single-step fact extraction from articles as subject–predicate–object triplets.
Uses one model call with structured output (`ExtractedFacts` / `Triplet`).

Run:

```bash
set -a; source .env; set +a
uv run python examples/03_fact_extraction_triplets.py
```

## Golden Example

Canonical production-leaning example with:
- kernel-issued `run_id` (`start_run`)
- mandatory middleware stack
- replay assertions (no duplicate events on second call)
- unknown tool outcome handling + `reconcile_tool(...)`
- post-reconcile replay assertions for tool results

Run:

```bash
set -a; source .env; set +a
uv run python examples/golden_example.py
```

## 04 - Autonomous Agent

Demonstrates the autonomous while-loop runtime:
- model/tool replay-safe conversation memory in `AutonomousAgent`
- durable `StepModelResult` replay with deterministic `step_key`s
- automatic tool execution loop managed by the `AutonomousAgent`

Run:

```bash
uv run python examples/04_autonomous_agent_research.py
```

## 05 - Hard Triplets Workflow

Demonstrates strict workflow control:
- explicit workflow steps with `WorkflowContext`
- human-in-the-loop pause/resume
- deterministic Python graph logic outside the model loop

Run:

```bash
uv run python examples/05_hard_triplets_workflow.py
```

## 06 - Triplets Swarm (Sub-Agent Runtime)

Demonstrates the sub-agent runtime pattern:
- lead model orchestrates extractor and adjudicator sub-agents
- deterministic `run_graph_math` Python tool for inferred relations
- capability-scoped execution with durable replay

Run:

```bash
set -a; source .env; set +a
uv run python examples/06_triplets_swarm.py
```

## 07 - Adaptive Agent Learning (Inter-Run Experience)

Demonstrates Experience Engine behavior:
- tenant/task-scoped `SQLiteExperienceStore` rule persistence
- `ContextBuilder` injection of `[PAST LEARNINGS FOR THIS TASK]`
- `AutonomousAgent(auto_reflect=True)` post-run reflection that writes reusable rules
- second run succeeding with fewer tool retries due to injected learnings

Run:

```bash
uv run python examples/07_adaptive_agent_learning.py
```
