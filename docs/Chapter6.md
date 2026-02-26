# Chapter 6: OS-Grade Safety V2 and Harness Reality

This chapter covers what is now available in Artana:

* `KernelPolicy.enforced_v2()` for OS-grade safety
* Declarative per-tool policy with `SafetyPolicyMiddleware`
* Intent plans, semantic idempotency, limits, approvals, invariants
* Kernel orchestration syscalls (status/checkpoint/artifacts/blocking/leases/streaming)
* First-class harness APIs (`HarnessContext`, `TaskUnit`, artifacts)

## Chapter Metadata

- Audience: Engineers implementing or auditing high-governance, side-effect-sensitive Artana systems.
- Prerequisites: Chapters 1–5 complete; familiarity with policy middleware and replay semantics.
- Estimated time: 35 minutes.
- Expected outcome: You can configure `enforced_v2` safely and operate approval, intent, and invariant workflows end-to-end.

Code block contract for this chapter:

* `pycon` blocks are in-context snippets intended for existing kernel/harness setups.
* Runnable end-to-end scripts live in `examples/` and are listed in `examples/README.md`.

---

# Step 1 — Boot an Enforced V2 Kernel

```pycon
from artana.kernel import ArtanaKernel, KernelPolicy
from artana.middleware import SafetyPolicyMiddleware
from artana.safety import (
    IntentRequirement,
    SafetyPolicyConfig,
    SemanticIdempotencyRequirement,
    ToolLimitPolicy,
    ToolSafetyPolicy,
)
from artana.store import SQLiteStore

safety = SafetyPolicyMiddleware(
    config=SafetyPolicyConfig(
        tools={
            "send_invoice": ToolSafetyPolicy(
                intent=IntentRequirement(require_intent=True, max_age_seconds=3600),
                semantic_idempotency=SemanticIdempotencyRequirement(
                    template="send_invoice:{tenant_id}:{billing_period}",
                    required_fields=("billing_period",),
                ),
                limits=ToolLimitPolicy(
                    max_calls_per_run=2,
                    max_calls_per_tenant_window=5,
                    tenant_window_seconds=3600,
                    max_amount_usd_per_call=500.0,
                    amount_arg_path="amount_usd",
                ),
            )
        }
    )
)

kernel = ArtanaKernel(
    store=SQLiteStore("chapter6_safety.db"),
    model_port=DemoModelPort(),  # your ModelPort implementation
    middleware=ArtanaKernel.default_middleware_stack(safety=safety),
    policy=KernelPolicy.enforced_v2(),
)
```

What `enforced_v2` guarantees at boot:

* PII scrubber required
* quota middleware required
* capability guard required
* safety policy middleware required

---

# Step 2 — Record a Typed Intent Plan Before Side Effects

```pycon
from artana.safety import IntentPlanRecord

await kernel.record_intent_plan(
    run_id="billing_run",
    tenant=tenant,
    intent=IntentPlanRecord(
        intent_id="intent_2026_02",
        goal="Send February invoice",
        why="Monthly billing close",
        success_criteria="Invoice sent exactly once",
        assumed_state="Customer account is active and approved",
        applies_to_tools=("send_invoice",),
    ),
)
```

If a configured tool requires intent and none exists (or it is stale), the tool call is blocked.
Autonomous workflows can also write this via the runtime tool `record_intent_plan`.

---

# Step 2b — Side-Effect Registration Guardrails

For mutating tools, enforce idempotency context at registration time:

```pycon
import json

from artana.ports.tool import ToolExecutionContext

@kernel.tool(requires_capability="billing:write", side_effect=True)
async def charge_card(
    customer_id: str,
    amount_usd: float,
    artana_context: ToolExecutionContext,
) -> str:
    # Forward idempotency key to external API calls.
    return json.dumps(
        {
            "customer_id": customer_id,
            "amount_usd": amount_usd,
            "idempotency_key": artana_context.idempotency_key,
        }
    )
```

If `side_effect=True` is used without
`artana_context: ToolExecutionContext`, tool registration fails with `ValueError`.

---

# Step 3 — Semantic Idempotency Prevents Business-Level Duplicates

For policy-configured tools, semantic keys are derived deterministically from template fields.

Example key:

`send_invoice:{tenant_id}:{billing_period}`

If the same semantic key already completed successfully, Artana blocks the new call with policy violation (`semantic_duplicate`).
If the prior outcome is unknown, Artana blocks until reconciliation (`semantic_requires_reconciliation`).

---

# Step 4 — Limits, Amount Controls, and Deterministic Invariants

`ToolLimitPolicy` can enforce:

* max calls per run
* max calls per tenant in a UTC time window
* max amount per call using an argument path

`InvariantRule` supports built-ins:

* `required_arg_true`
* `email_domain_allowlist`
* `recipient_must_be_verified`
* `custom_json_rule`
* `ast_validation_passed`
* `linter_passed`

`ast_validation_passed` and `linter_passed` use a configured `field` that points
to the code payload in tool arguments.

All violations are hard blocks and are audited via `policy_decision` run summaries.

---

# Step 5 — Approval Gates (Human + Critic)

Human approval flow:

```pycon
from pydantic import BaseModel
from artana.kernel import ApprovalRequiredError

class SendInvoiceArgs(BaseModel):
    billing_period: str
    amount_usd: float

try:
    await kernel.step_tool(
        run_id="billing_run",
        tenant=tenant,
        tool_name="send_invoice",
        arguments=SendInvoiceArgs(billing_period="2026-02", amount_usd=120.0),
        step_key="invoice_send",
    )
except ApprovalRequiredError as exc:
    await kernel.approve_tool_call(
        run_id="billing_run",
        tenant=tenant,
        approval_key=exc.approval_key,
        mode="human",
        reason="Finance manager approved",
    )
    await kernel.step_tool(
        run_id="billing_run",
        tenant=tenant,
        tool_name="send_invoice",
        arguments=SendInvoiceArgs(billing_period="2026-02", amount_usd=120.0),
        step_key="invoice_send",
    )
```

Critic approval flow is kernel-managed and replay-safe. It runs a deterministic model step and either:

* records approval and continues
* blocks with `critic_denied`

---

# Step 6 — Harness APIs Are First-Class

Artana now has an explicit harness substrate:

* `HarnessContext` and `BaseHarness`
* `IncrementalTaskHarness` with typed `TaskUnit`
* `SupervisorHarness` for composition
* `TestDrivenHarness` for verify-before-done task progression
* `run_draft_model(...)` and `run_verify_model(...)` wrappers on `BaseHarness`
* built-in artifact helpers (`set_artifact`, `get_artifact`)
* `DraftVerifyLoopConfig` and `AcceptanceSpec` for deterministic autonomous completion

Example artifact usage:

```pycon
await harness.set_artifact(key="plan", value={"version": 2, "status": "approved"})
artifact = await harness.get_artifact(key="plan")
```

Artifacts are currently persisted as `run_summary` entries under `artifact::<key>`.
This gives durable retrieval without introducing a separate event type.

TDD verification pattern:

```pycon
from artana.harness import TestDrivenHarness

class MyTDDHarness(TestDrivenHarness):
    async def work_on(self, task):
        # update files first...
        await self.verify_and_commit(
            task_id=task.id,
            test_command="pytest -q",
        )
```

If verification fails, the task is moved back to `pending` and cannot transition to `done`.

---

# Step 7 — Co-Located Workspace Context

You can inject a repository-local plan file into the system context for every turn:

```pycon
from artana.agent import ContextBuilder

context_builder = ContextBuilder(
    workspace_context_path="docs/ACTIVE_PLAN.md",
)
```

When the file exists and is non-empty, it is appended as:
`Workspace Context / Active Plan: ...`

---

# Step 8 — Autonomous Draft/Verify With Deterministic Acceptance Gates

```pycon
from artana import AcceptanceSpec, ToolGate
from artana.agent import AutonomousAgent, DraftVerifyLoopConfig

agent = AutonomousAgent(
    kernel=kernel,
    loop=DraftVerifyLoopConfig(
        draft_model="gpt-5.3-codex-spark",
        verify_model="gpt-5.3-codex",
    ),
)

result = await agent.run(
    run_id="repair_run",
    tenant=tenant,
    model="openai/gpt-5.3-codex",
    prompt="Fix flaky tests and stop only when validated.",
    output_schema=FinalDecision,
    acceptance=AcceptanceSpec(
        gates=(ToolGate(tool="run_tests", must_pass=True),),
    ),
)
```

This makes completion deterministic:

* draft model proposes
* acceptance gate tool(s) must pass
* verify model adjudicates final completion

---

# Step 9 — Layer Selection Guide

Kernel orchestration syscalls for schedulers/workers:

```pycon
status = await kernel.get_run_status(run_id="billing_run")
resume_point = await kernel.resume_point(run_id="billing_run")
active_runs = await kernel.list_active_runs(tenant_id=tenant.tenant_id)
capabilities = await kernel.describe_capabilities(tenant=tenant)
visible_tools = kernel.list_tools_for_tenant(tenant=tenant)

await kernel.acquire_run_lease(
    run_id="billing_run",
    worker_id="worker_a",
    ttl_seconds=30,
)
```

Use this decision rule:

Model calls in all three layers use the same default flow: `ModelCallOptions(api_mode="auto")` (Responses when supported, chat-completions fallback otherwise).

* **Kernel**: when you want minimal deterministic primitives (`step_model`, `step_tool`, `run_workflow`)
* **Harness**: when you need structured long-running discipline and typed task progress
* **AutonomousAgent**: when model-led loops are desired, with kernel safety underneath

For formal contracts, see:

* `docs/kernel_contracts.md`
* `docs/deep_traceability.md`

---

## You Should Now Be Able To

- Configure and validate OS-grade safety policies with intent, dedupe, limits, approvals, and invariants.
- Debug policy and orchestration state with run status, resume points, capability views, and summaries.
- Select the right Artana layer for each workload while keeping deterministic safety guarantees.

## Where To Go Next

- Revisit [Chapter 1](./Chapter1.md) for onboarding refinements when training new teammates.
- Use [kernel_contracts.md](./kernel_contracts.md) as the operational source of truth.
- Use [deep_traceability.md](./deep_traceability.md) for observability and audit workflows.
- Use [examples/README.md](../examples/README.md) to run end-to-end scenarios.
