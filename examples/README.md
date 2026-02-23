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
