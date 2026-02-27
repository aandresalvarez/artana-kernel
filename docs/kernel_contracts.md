# Kernel Contracts

This document is the operational contract for replay, tool compatibility, and policy behavior.
The machine-generated behavior index lives at `docs/kernel_behavior_index.json` and is
validated in CI.

## Replay Policy

`step_model` and `KernelModelClient.step` support:

- `strict` (default): exact replay requires matching prompt, messages, model, model options, Responses input items, tool signatures, and `step_key`.
- `allow_prompt_drift`: if prompt/messages/model options drift for the same `(model, step_key, tool signatures)`, replay the prior completion and append `replayed_with_drift`.
- `fork_on_drift`: if drift is detected for the same `(model, step_key, tool signatures)`, fork into `run_id::fork::<hash>` and execute there.

`step_key` is required for drift detection semantics. Without a stable `step_key`, strict replay still works for exact repeats but drift workflows are not activated.

High-level ergonomics note:

- `KernelModelClient.step(...)` / `SingleStepModelClient.step(...)` generate a deterministic step key when `step_key` is omitted.
- For long-lived workflows where you intentionally evolve prompts/options, explicit `StepKey(...)` values are still recommended.
- `KernelModelClient.capabilities()` exposes whether the bound kernel supports `replay_policy` and `context_version`.
- Mixed-version compatibility: if unsupported kwargs are detected, the client retries once without unsupported kwargs and emits a warning.

## Kernel Policy Modes

- `permissive`: no required middleware.
- `enforced`: requires `PIIScrubberMiddleware`, `QuotaMiddleware`, and `CapabilityGuardMiddleware`.
- `enforced_v2`: requires all `enforced` middleware plus `SafetyPolicyMiddleware`.

## Run Lifecycle Contracts

Kernel orchestration syscalls now include:

- `get_run_status(run_id)`
- `get_run_progress(run_id)`
- `stream_run_progress(run_id, since_seq=0, follow=False, ...)`
- `list_active_runs(tenant_id, ...)`
- `resume_point(run_id)`
- `block_run(...)`
- `unblock_run(...)`

Status semantics:

- `active`: run is not terminal and not currently paused.
- `paused`: latest unresolved `pause_requested` exists.
- `failed`: harness-level failure recorded.
- `completed`: harness sleep recorded with completed status.

Run progress semantics:

- `status`: `running|completed|failed` (`queued` and `cancelled` are reserved for future lifecycle support).
- `percent`: deterministic integer in `[0, 100]`.
- `current_stage`: best-known active stage from `task_progress` summaries, otherwise `explain_run().last_stage`.
- `completed_stages`: ordered stage ids from `task_progress` entries with `state=done`.
- `eta_seconds`: provided only when deterministic signal is sufficient; otherwise `null`.

Event-loop ownership semantics:

- `ArtanaKernel` and `PostgresStore` are loop-affine and are expected to be used from one event loop.
- Repeated sync access patterns must bridge onto the owning loop instead of repeatedly creating fresh loops with `asyncio.run(...)`.
- Close the kernel/store once at shutdown after in-flight operations drain.

Postgres read retry semantics:

- `PostgresStore` read APIs retry transient connection-lifecycle failures up to `max_retry_attempts`.
- On retryable read failure, the store invalidates the current pool and reconnects before retrying.
- Retry classification includes `asyncpg.PostgresConnectionError` (including `ConnectionDoesNotExistError`) and connection-closed interface errors.

## Model Request Invariants

Each `model_requested` event stores:

- `api_mode` (`auto|responses|chat`)
- `reasoning_effort` (optional)
- `verbosity` (optional)
- `previous_response_id` (optional)
- `responses_input_items` (optional, canonicalized)
- `allowed_tools`: sorted tool names
- `allowed_tool_signatures`: `name + tool_version + schema_version + schema_hash`
- `allowed_tools_hash`: hash of tool signatures (not just tool names)
- `context_version`:
  - `system_prompt_hash`
  - `context_builder_version`
  - `compaction_version`

Replay validates tool signatures plus model input identity (prompt/messages/options/Responses items).

## Model Completion Metadata

Each `model_completed` event stores:

- `api_mode_used` (`responses|chat`)
- `response_id` (optional)
- `responses_output_items` (full normalized output items)
- `tool_calls` (canonicalized arguments JSON)

This keeps audit and replay fidelity for both chat-completions and Responses-native providers.

## Model API Mode Defaults

Artana model calls default to `ModelCallOptions(api_mode="auto")`:

- use Responses when supported by provider/model routing
- fallback to chat-completions when Responses is unsupported
- use `api_mode="responses"` for strict Responses-only behavior

## Tool Determinism Invariants

- Tool arguments are canonicalized as sorted JSON objects before matching and storage.
- Tool idempotency key input is `(run_id, tool_name, seq)`.
- `@kernel.tool(side_effect=True)` requires signature parameter
  `artana_context: ToolExecutionContext`; registration fails fast otherwise.
- Tool request events persist `tool_version` and `schema_version`.
- `tool_requested` payload optionally persists:
  - `semantic_idempotency_key`
  - `intent_id`
  - `amount_usd`

Tool gateway metadata is part of tool definitions:

- `risk_level` (`low|medium|high|critical`)
- `sandbox_profile` (optional string)

Kernel exposes:

- `canonicalize_tool_args(tool_name, args)`
- `tool_fingerprint(tool_name)`

Capability visibility helpers:

- `describe_capabilities(tenant=...)` for allow/filter reasoning payloads
- `list_tools_for_tenant(tenant=...)` for effective visible tool definitions

## Safety Policy Invariants

When `SafetyPolicyMiddleware` is configured for a tool, `prepare_tool_request` evaluates rules in this order:

1. intent requirement
2. semantic idempotency
3. tool limits/rate checks
4. approval gates
5. deterministic invariants

Each decision appends `run_summary` with:
- `summary_type=policy_decision`
- JSON payload containing:
  - `tool_name`
  - `fingerprint`
  - `outcome` (`allow` or `deny`)
  - `rule_id`
  - `reason`

Built-in deterministic invariants:

- `required_arg_true`
- `email_domain_allowlist`
- `recipient_must_be_verified`
- `custom_json_rule`
- `ast_validation_passed` (valid Python syntax via AST parse on `field`)
- `linter_passed` (deterministic lint checks on `field`)

### Semantic Idempotency

- Semantic keys are derived deterministically from configured templates.
- Duplicate prior `success` outcomes are blocked (`semantic_duplicate`).
- Prior `unknown_outcome` for the same semantic key is blocked until reconciliation (`semantic_requires_reconciliation`).

### Approval Gates

- Approval records are run summaries under `summary_type=policy::approval::<approval_key>`.
- Human mode raises `ApprovalRequiredError` until approval is recorded.
- Critic mode uses deterministic kernel model steps with key prefix `critic::<tool>::`.
- Critic denials raise `PolicyViolationError(code="critic_denied")`.

### Intent Plans

- Intent records are run summaries under `summary_type=policy::intent_plan`.
- `record_intent_plan(...)` stores typed `IntentPlanRecord` payloads.
- Missing or stale intent plans for configured tools raise
  `PolicyViolationError(code="missing_intent_plan")`.

## Tool IO Policy Hooks

Kernel middleware now includes tool hooks:

- `prepare_tool_request(run_id, tenant, tool_name, arguments_json)`
- `prepare_tool_result(run_id, tenant, tool_name, result_json)`

Hooks run in middleware order, enabling policy enforcement on tool input/output in addition to model prompt/messages.

## Checkpoints and Artifacts

Kernel-native checkpoint syscall:

- `checkpoint(run_id, tenant, name, payload, ...)`

Current persistence representation:

- checkpoint writes `run_summary` with `summary_type=checkpoint::<name>`

Kernel artifact syscalls:

- `set_artifact(run_id, tenant, key, value, ...)`
- `get_artifact(run_id, key)`
- `list_artifacts(run_id)`

Current persistence representation:

- artifact writes `run_summary` with `summary_type=artifact::<key>`
- payload envelope: `{"value": ...}` (optional schema metadata can be attached)

## Agent Observability

The autonomous agent emits `run_summary` events for model and tool steps.
These summaries are queryable via `query_event_history`.
Kernel model steps also emit `run_summary` entries with `summary_type=capability_decision`
that explain why each tool was allowed or filtered.

`ContextBuilder(workspace_context_path=...)` is additive context input only; it does not
change kernel replay/event contracts beyond standard model input identity tracking.

## Acceptance Gate Contracts

`AutonomousAgent.run(..., acceptance=AcceptanceSpec(...))` evaluates configured `ToolGate` items
after a draft candidate and before final completion.

Behavior contract:

- each gate executes as deterministic tool steps with stable key pattern
  `turn_{iteration}_accept_tool_{index}_{tool}`
- gate outcomes are persisted in `run_summary` with `summary_type=agent_acceptance_gate`
- if any `must_pass=True` gate fails, the run continues to another draft turn
- when all required gates pass, the run can proceed to final verification/return

## Harness and Artifact Contracts

Artana exposes first-class harness APIs:

- `HarnessContext`
- `BaseHarness`
- `IncrementalTaskHarness`
- `TestDrivenHarness`
- `TaskUnit`
- `SupervisorHarness`

Model helper wrappers in `BaseHarness`:

- `run_model(...)`
- `run_draft_model(...)`
- `run_verify_model(...)`
- `run_tool(...)`

`TestDrivenHarness.verify_and_commit(...)` enforces verification before `TaskUnit -> done`.

Artifacts in harnesses are persisted as run summaries with `summary_type=artifact::<key>`.
`set_artifact(...)` writes `{"value": ...}` payloads and `get_artifact(...)` resolves the latest value.

## Event Streaming and Leases

Store-agnostic event streaming contract:

- `stream_events(run_id, since_seq=0, follow=False, ...) -> AsyncIterator[KernelEvent]`

Run-lease contracts for multi-worker schedulers:

- `acquire_run_lease(run_id, worker_id, ttl_seconds)`
- `renew_run_lease(run_id, worker_id, ttl_seconds)`
- `release_run_lease(run_id, worker_id)`
- `get_run_lease(run_id)`

## Versioning and Compatibility Contracts

- Tagged `0.x` releases are tracked in `CHANGELOG.md`.
- Runtime and store compatibility matrix is published in `docs/compatibility_matrix.md`.
- Store backends expose schema metadata via `get_schema_info()` returning:
  - `backend`: `sqlite|postgres`
  - `schema_version`: backend schema contract version string

## CLI Operational Contracts

CLI run inspection commands:

- `artana run list --db ... | --dsn ...`
- `artana run tail <run_id> --db ... | --dsn ...`
- `artana run verify-ledger <run_id> --db ... | --dsn ...`
- `artana run status <run_id> --db ... | --dsn ...`
- `artana run summaries <run_id> [--type ...] [--limit ...] --db ... | --dsn ...`
- `artana run artifacts <run_id> --db ... | --dsn ...`
- `artana init [path] [--profile enforced|dev] [--force]`

Output contract:

- `run list`: run ids, one per line
- `run tail`: tab-separated `seq timestamp event_type parent_step_key`
- `verify-ledger`: prints `valid` or `invalid`; returns exit code `0` on valid, `1` on invalid
- `status`: run lifecycle summary; `--json` emits machine-readable payload
- `summaries`: latest run summaries; supports `--type` and `--limit`
- `artifacts`: latest artifact key/value snapshots
- `--json` is supported across run inspection commands (`list`, `tail`, `verify-ledger`, `status`, `summaries`, `artifacts`)
