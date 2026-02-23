# Kernel Contracts

This document is the operational contract for replay, tool compatibility, and policy behavior.
The machine-generated behavior index lives at `docs/kernel_behavior_index.json` and is
validated in CI.

## Replay Policy

`step_model` and `KernelModelClient.step` support:

- `strict` (default): exact replay requires matching prompt, messages, model, tool signatures, and `step_key`.
- `allow_prompt_drift`: if prompt/messages drift for the same `(model, step_key, tool signatures)`, replay the prior completion and append `replayed_with_drift`.
- `fork_on_drift`: if drift is detected for the same `(model, step_key, tool signatures)`, fork into `run_id::fork::<hash>` and execute there.

## Model Request Invariants

Each `model_requested` event stores:

- `allowed_tools`: sorted tool names
- `allowed_tool_signatures`: `name + tool_version + schema_version + schema_hash`
- `allowed_tools_hash`: hash of tool signatures (not just tool names)
- `context_version`:
  - `system_prompt_hash`
  - `context_builder_version`
  - `compaction_version`

Replay validates signatures first. Name-only hash fallback is used only for legacy events.

## Tool Determinism Invariants

- Tool arguments are canonicalized as sorted JSON objects before matching, storage, and idempotency-key derivation.
- Tool idempotency key input is canonical arguments plus `(run_id, tool_name, step_key)`.
- Tool request events persist `tool_version` and `schema_version`.

## Tool IO Policy Hooks

Kernel middleware now includes tool hooks:

- `prepare_tool_request(run_id, tenant, tool_name, arguments_json)`
- `prepare_tool_result(run_id, tenant, tool_name, result_json)`

Hooks run in middleware order, enabling policy enforcement on tool input/output in addition to model prompt/messages.

## Agent Observability

The autonomous agent emits `run_summary` events for model and tool steps.
These summaries are queryable via `query_event_history`.
Kernel model steps also emit `run_summary` entries with `summary_type=capability_decision`
that explain why each tool was allowed or filtered.
