Below is a clean set of **Jira-ready stories**, grouped into epics, with clear acceptance criteria and implementation notes. You can paste these directly into Jira (one per story).

---

# EPIC 1 — Replay Determinism Hardening

---

## 🟦 Story 1.1 — Make `allowed_tools` Ordering Deterministic in Model Replay

**Summary**
Ensure replay consistency does not depend on tool registration order.

**Description**
Currently `allowed_tools` is stored and compared as an ordered list. Changes in registration order may break replay even when the logical tool set is unchanged.

We must normalize tool ordering before persisting and comparing in replay.

**Acceptance Criteria**

* `allowed_tool_names` is sorted before:

  * persisting in `ModelRequestedPayload`
  * comparison in `find_matching_model_cycle`
* Replays succeed when tool registration order changes but tool set remains identical.
* Add unit test:

  * Register tools in different order across runs
  * Replay still succeeds
* No behavior change for existing examples.

**Implementation Notes**

* Replace:

  ```python
  [tool.name for tool in scoped_invocation.allowed_tools]
  ```

  with:

  ```python
  sorted(tool.name for tool in scoped_invocation.allowed_tools)
  ```

---

## 🟦 Story 1.2 — Add `allowed_tools_hash` to ModelRequestedPayload

**Summary**
Add a deterministic hash to guard against tool set drift.

**Description**
Store a SHA256 hash of the sorted tool names to make replay validation faster and more explicit.

**Acceptance Criteria**

* `ModelRequestedPayload` includes:

  ```python
  allowed_tools_hash: str
  ```
* Hash is computed as:

  ```
  sha256(",".join(sorted_names))
  ```
* Replay compares hash first before full list comparison.
* Backward compatibility preserved (legacy runs still load).

---

# EPIC 2 — Capability Enforcement Consolidation

---

## 🟦 Story 2.1 — Remove Duplicate Capability Filtering

**Summary**
Unify capability filtering logic to avoid drift between middleware and policy enforcement.

**Description**
Currently capability filtering happens in both:

* `CapabilityGuardMiddleware`
* `enforce_capability_scope()`

This duplication creates risk of divergence.

We must consolidate enforcement into a single mechanism.

**Acceptance Criteria**

* Only one capability enforcement path exists.
* `KernelPolicy(mode="enforced")` requires `CapabilityGuardMiddleware`.
* `enforce_capability_scope()` removed or converted to internal helper.
* All examples continue to function.
* Add test:

  * Tool requiring capability is hidden when capability missing.
  * Tool visible when capability present.

---

# EPIC 3 — ModelPort Message Support

---

## 🟦 Story 3.1 — Extend ModelRequest to Support Full Chat Messages

**Summary**
Allow model adapters to receive full message history.

**Description**
Current `ModelRequest` only includes `prompt`. Kernel supports `messages` but adapters ignore them.

We must extend the interface to support multi-turn semantics.

**Acceptance Criteria**

* `ModelRequest` includes:

  ```python
  messages: Sequence[ChatMessage]
  ```
* `LiteLLMAdapter.complete()` uses full message list.
* Existing prompt-only flows still work.
* Unit test:

  * ModelRequest created from `ModelInput.from_messages`
  * Adapter receives full message list.

---

# EPIC 4 — Workflow Integrity Improvements

---

## 🟦 Story 4.1 — Emit RUN_STARTED for Workflow Runs

**Summary**
Ensure workflow-based runs start with `RUN_STARTED`.

**Description**
`run_workflow()` currently does not emit a start event if the run does not exist.

This creates inconsistency with `start_run()`.

**Acceptance Criteria**

* If `get_events_for_run()` returns empty:

  * Append `RUN_STARTED`
* Workflow runs have identical invariants to manual runs.
* `verify_run_chain()` passes for workflow runs.

---

# EPIC 5 — Tool Idempotency Stability

---

## 🟦 Story 5.1 — Improve Idempotency Key Derivation for Tools

**Summary**
Make tool idempotency deterministic across concurrent executions.

**Description**
Current idempotency key is derived from `(run_id, seq)` when no `step_key` is provided. This is fragile under concurrency.

Replace with deterministic derivation based on tool identity.

**Acceptance Criteria**

* Idempotency key derived from:

  ```
  sha256(f"{run_id}:{tool_name}:{arguments_json}:{step_key}")
  ```
* `seq` no longer used for key derivation.
* Replays are stable across concurrent calls.
* Add test:

  * Same tool + args produces same idempotency key.
  * Different args produce different keys.

---

# EPIC 6 — Tool Schema & Validation

---

## 🟦 Story 6.1 — Improve Tool Argument JSON Schema Generation

**Summary**
Generate correct JSON schema types from function annotations.

**Description**
Current schema treats all arguments as strings. Improve schema mapping for:

* int
* float
* bool
* Decimal
* Optional
* Enum

**Acceptance Criteria**

* Generated JSON schema reflects correct types.
* Required parameters correctly inferred.
* ToolDefinition.arguments_schema_json matches annotation types.
* Add tests verifying schema correctness.

---

## 🟦 Story 6.2 — Validate Tool Arguments Before Execution

**Summary**
Ensure tool arguments match schema before execution.

**Description**
Arguments are currently passed directly to the tool function without schema validation.

Add validation step before invocation.

**Acceptance Criteria**

* Required arguments enforced.
* Unknown arguments rejected.
* Invalid types rejected.
* Clear error messages returned.
* Add negative test cases.

---

# EPIC 7 — Quota Efficiency Improvements

---

## 🟦 Story 7.1 — Optimize QuotaMiddleware Store-Based Aggregation

**Summary**
Avoid O(n) scan of events on every model call.

**Description**
Quota currently sums costs by iterating all events per run.

Replace with aggregated SQL query.

**Acceptance Criteria**

* SQLite implementation uses:

  ```
  SUM(cost_usd) WHERE event_type='model_completed'
  ```
* Performance improvement measurable in benchmark.
* Behavior unchanged.
* Add test verifying correct aggregation.

---

# EPIC 8 — ChatClient Naming & Responsibility Clarification

---

## 🟦 Story 8.1 — Rename ChatClient to Reflect Actual Responsibility

**Summary**
Rename `ChatClient` to better reflect that it is a thin model-step wrapper.

**Description**
Current name implies conversational agent. It only wraps `step_model()`.

Rename to:

* `KernelModelClient` or
* `StructuredModelClient`

**Acceptance Criteria**

* Class renamed.
* Examples updated.
* Documentation updated.
* No behavior change.

---

## 🟦 Story 8.2 — Create Dedicated Agent Runtime Abstraction (Optional Future Work)

**Summary**
Introduce a proper Agent runtime separate from kernel.

**Description**
Kernel provides replay-safe primitives. Agent runtime should manage:

* message accumulation
* tool execution loop
* memory summarization
* policy-driven reasoning

This story only defines the abstraction and basic skeleton.

**Acceptance Criteria**

* New `AgentRuntime` abstraction defined.
* Clear separation between:

  * Kernel (event-sourced)
  * Agent runtime (looping policy)
* No breaking changes to kernel.

---

# EPIC 9 — PII Scrubber Transparency

---

## 🟦 Story 9.1 — Document PIIScrubber as Demo-Level Middleware

**Summary**
Clarify that PIIScrubber is basic and not production-grade.

**Description**
Add documentation warning regarding limitations.

**Acceptance Criteria**

* README updated.
* Middleware docstring updated.
* No behavior change.

---

# Suggested Jira Priority Order

1. 🔴 Deterministic allowed_tools (1.1)
2. 🔴 Capability consolidation (2.1)
3. 🔴 ModelPort message support (3.1)
4. 🟠 Idempotency improvement (5.1)
5. 🟠 Workflow RUN_STARTED (4.1)
6. 🟡 Tool schema & validation (6.x)
7. 🟡 Quota optimization (7.1)
8. 🟢 Naming cleanup (8.1)

 