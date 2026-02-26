# Examples

Run examples from the repository root.

Use `artana init <path>` for a generated local-first starter (`app.py`) with `MockModelPort`.
Use explicit `StepKey(...)` in that scaffold when you want deterministic workflow-style key control.

## Runtime Profiles

- Local-first (no external API key): `01_durable_chat_replay.py`, `04_autonomous_agent_research.py`, `05_hard_triplets_workflow.py`, `07_adaptive_agent_learning.py`, `09_harness_engineering_dx.py`
- Live model required (`OPENAI_API_KEY`): `02_real_litellm_chat.py`, `03_fact_extraction_triplets.py`, `06_triplets_swarm.py`, `08_responses_mode.py`, `golden_example.py`

All live examples fail fast with a troubleshooting message if `OPENAI_API_KEY` is not loaded.

## Model Overrides

- `ARTANA_MODEL`: override model for `02`, `03`, and `golden_example`.
- `ARTANA_RESPONSES_MODEL`: override model for `08_responses_mode.py`.
- `ARTANA_MODEL_LEAD`: override lead model in `06_triplets_swarm.py`.
- `ARTANA_MODEL_EXTRACTOR`: override extractor sub-agent model in `06_triplets_swarm.py`.
- `ARTANA_MODEL_ADJUDICATOR`: override adjudicator sub-agent model in `06_triplets_swarm.py`.

## Chapter Path Mapping

Use this mapping to follow the Chapter 1 → 6 learning path with runnable scripts.

| Chapter | Primary examples |
| --- | --- |
| Chapter 1 (first success + primitives) | `01_durable_chat_replay.py` (local), `03_fact_extraction_triplets.py` (live), `05_hard_triplets_workflow.py` (local) |
| Chapter 2 (harness discipline + supervision) | `09_harness_engineering_dx.py` |
| Chapter 3 (failure/replay/recovery) | `golden_example.py`, `05_hard_triplets_workflow.py` |
| Chapter 4 (advanced orchestration) | `06_triplets_swarm.py`, `09_harness_engineering_dx.py` |
| Chapter 5 (operations/distributed posture) | `02_real_litellm_chat.py`, `08_responses_mode.py` |
| Chapter 6 (safety and governance) | `golden_example.py`, `09_harness_engineering_dx.py` |

## 01 - Durable Chat Replay

Demonstrates:
- tenant context + middleware
- durable event log in SQLite
- replay-safe model step behavior on repeated prompts
- replay-safe tool step behavior via deterministic `step_key`

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

## 08 - Responses Mode (OpenAI Responses API)

Demonstrates Responses-native model calls through Artana:
- explicit `ModelCallOptions(api_mode="responses")` controls
- reasoning and verbosity options
- `previous_response_id` chaining across turns
- surfaced `api_mode_used`, `response_id`, and response metadata

Run:

```bash
set -a; source .env; set +a
uv run python examples/08_responses_mode.py
```

Normal production baseline remains `ModelCallOptions(api_mode="auto")` (automatic Responses/chat routing).

## 09 - Harness Engineering DX

Demonstrates a complete harness-oriented flow:
- deterministic `StepKey` generation
- `run_draft_model(...)` / `run_verify_model(...)`
- `@kernel.tool(side_effect=True)` registration guardrails with `ToolExecutionContext`
- autonomous acceptance gating via `AcceptanceSpec` + `ToolGate`
- coding bundle usage via `CodingHarnessTools`

Run:

```bash
uv run python examples/09_harness_engineering_dx.py
```
