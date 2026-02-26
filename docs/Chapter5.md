# Chapter 5: Distributed Scaling & Multi-Tenant Deployment

This chapter demonstrates:

* Multi-tenant isolation
* Horizontal scaling patterns
* Worker architecture
* Queue-based execution
* Long-running harness recovery
* Deployment topology
* Production safety checklist

## Chapter Metadata

- Audience: Engineers operating Artana across teams, workers, and environments.
- Prerequisites: Chapters 1–4 complete; familiarity with Postgres-backed deployments and worker patterns.
- Estimated time: 35 minutes.
- Expected outcome: You can operate distributed Artana runs, inspect state via CLI, and ship replay-safe upgrades.

Code block contract for this chapter:

* `pycon` blocks are in-context snippets for existing worker/orchestrator contexts.
* Runnable end-to-end scripts live in `examples/` and are listed in `examples/README.md`.

---

# 🏗️ Step 1 — Multi-Tenant Isolation (First-Class Concept)

In Artana, tenants are explicit.

Every run is tied to:

```pycon
TenantContext(
    tenant_id="tenant_name",
    capabilities=frozenset({...}),
    budget_usd_limit=...
)
```

Example:

```pycon
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore

kernel = ArtanaKernel(
    store=SQLiteStore("multi_tenant.db"),
    model_port=DemoModelPort(),
)

tenant_a = TenantContext(
    tenant_id="tenant_a",
    capabilities=frozenset({"analytics"}),
    budget_usd_limit=10.0,
)

tenant_b = TenantContext(
    tenant_id="tenant_b",
    capabilities=frozenset(),
    budget_usd_limit=2.0,
)
```

Every run enforces:

* Budget
* Capabilities
* Policy
* Ledger separation

Isolation is guaranteed at the run level.

---

# ⚙️ Step 2 — Horizontal Scaling Pattern

Artana Kernel is stateless.

State lives in:

* EventStore
* MemoryStore
* ExperienceStore

This enables horizontal scaling.

### Worker Pattern

Each worker process:

```pycon
from artana import ArtanaKernel, KernelPolicy, PostgresStore
from artana.ports.model_adapter import LiteLLMAdapter

kernel = ArtanaKernel(
    store=PostgresStore("postgresql://user:pass@db:5432/artana"),  # shared DB
    model_port=LiteLLMAdapter(...),  # model calls default to ModelCallOptions(api_mode="auto")
    middleware=ArtanaKernel.default_middleware_stack(),
    policy=KernelPolicy.enforced(),
)
```

Workers can:

* Load any run
* Resume safely
* Replay deterministically
* Continue long-running harness

No in-memory coordination required.

`api_mode="auto"` is the normal production flow: workers use Responses when supported and automatically fall back to chat-completions when a provider/model does not support Responses.

---

# 🔁 Step 3 — Queue + Worker Architecture

Example using a simple async queue:

```pycon
import asyncio

task_queue = asyncio.Queue()

async def worker():
    while True:
        run_id, tenant = await task_queue.get()
        worker_id = "worker-1"
        leased = await kernel.acquire_run_lease(
            run_id=run_id,
            worker_id=worker_id,
            ttl_seconds=30,
        )
        if not leased:
            task_queue.task_done()
            continue
        harness = DeploymentHarness(kernel=kernel, tenant=tenant)
        try:
            await harness.run(run_id)
        finally:
            await kernel.release_run_lease(run_id=run_id, worker_id=worker_id)
            task_queue.task_done()
```

Key insight:

* Workers can crash
* On restart, they resume from durable state
* Harness enforces clean-state validation
* Kernel guarantees replay

This enables:

* Kubernetes auto-scaling
* Serverless execution
* Background job systems

---

# 🧠 Step 4 — Long-Running Harness Recovery

If a worker crashes mid-task:

```pycon
await harness.run("migration_run")
```

On restart:

```pycon
await harness.run("migration_run")
```

Because:

* TaskProgressSnapshot is persisted
* Tool resolutions are reconciled
* Partial states rejected
* step_key prevents duplication

Recovery is deterministic.

---

# 🗃️ Step 5 — Distributed Event Store (PostgresStore)

`PostgresStore` is implemented in the Artana library and can be used directly:

```pycon
from artana.store import PostgresStore

store = PostgresStore(
    dsn="postgresql://user:pass@db:5432/artana",
    min_pool_size=2,
    max_pool_size=20,
    command_timeout_seconds=30.0,
)
```

Then pass it into `ArtanaKernel`; all kernel logic remains identical.

Implementation notes:

* Event append is transactional
* Per-run sequencing is protected with advisory locks
* Hash-chain ledger semantics remain the same as SQLite
* Replay and idempotency behavior do not change across stores

Benefits:

* Multi-worker concurrency
* High durability
* Strong transactional guarantees
* Cloud-native deployments

EventStore is the only scaling boundary.

---

# 🌍 Step 6 — Multi-Region Deployment Strategy

Architecture recommendation:

```
[Load Balancer]
      |
[Stateless API Layer]
      |
[Worker Pool]
      |
[Shared Postgres Event Store]
      |
[Model Provider APIs]
```

Rules:

* API nodes stateless
* Workers stateless
* All state in DB
* Idempotency enforced
* Replay safe

This allows:

* Blue/green deploys
* Canary releases
* Zero-downtime upgrades

---

# 🔄 Step 7 — Rolling Upgrade Strategy (Replay Safe)

When deploying new code:

1. Old workers finish current runs
2. New workers start
3. replay_policy="allow_prompt_drift" during transition
4. Monitor REPLAYED_WITH_DRIFT events

If incompatible change:

Use:

```pycon
replay_policy="fork_on_drift"
```

Old runs remain untouched.

New runs fork cleanly.

---

# 📊 Step 8 — Observability & Monitoring

Monitor:

* Model cost aggregation
* Drift events
* Unknown tool outcomes
* BudgetExceededError
* ReplayConsistencyError

Example cost dashboard query:

```sql
SELECT
  tenant_id,
  COALESCE(SUM((payload_json::jsonb ->> 'cost_usd')::double precision), 0.0) AS spend
FROM kernel_events
WHERE event_type = 'model_completed'
GROUP BY tenant_id;
```

Production metrics to track:

* Avg cost per run
* Drift rate
* Fork rate
* Tool failure rate
* Harness clean-state violations

---

# 🔐 Step 9 — Security Hardening Checklist

Production kernel should:

* Use `KernelPolicy.enforced()` at minimum
* Use `KernelPolicy.enforced_v2()` + `SafetyPolicyMiddleware` for side-effect tools
* Enable PII scrubber
* Enforce quota
* Enforce capability guard
* Validate tool idempotency
* Configure safety policy (intent, semantic dedupe, limits, approvals, invariants) where needed
* Monitor drift

Optional:

* Encrypt EventStore at rest
* Encrypt tool payloads
* Sign event_hash externally
* Stream ledger to SIEM

---

# 📦 Step 10 — Production Deployment Template

Minimal production instantiation:

```pycon
from artana import ArtanaKernel, KernelPolicy
from artana.middleware import SafetyPolicyMiddleware
from artana.ports.model_adapter import LiteLLMAdapter
from artana.safety import SafetyPolicyConfig
from artana.store import PostgresStore

safety = SafetyPolicyMiddleware(config=SafetyPolicyConfig(tools={...}))

kernel = ArtanaKernel(
    store=PostgresStore("postgresql://..."),
    model_port=LiteLLMAdapter(...),  # auto Responses routing by default
    middleware=ArtanaKernel.default_middleware_stack(safety=safety),
    policy=KernelPolicy.enforced_v2(),
)
```

Workers:

```pycon
harness = MyHarness(kernel=kernel, tenant=tenant)
await harness.run(run_id)
```

That’s it.

Everything else is architecture.

---

# 🧰 Step 11 — CLI Operations for Run Inspection

Use the built-in CLI to inspect distributed runs without writing ad-hoc SQL.

```bash
# List recent run IDs
artana run list --db .state.db

# Tail events for one run
artana run tail run_123 --db .state.db

# Verify hash-chain ledger integrity
artana run verify-ledger run_123 --db .state.db

# Run lifecycle summary (human and machine friendly)
artana run status run_123 --db .state.db
artana run status run_123 --db .state.db --json

# Inspect summaries and artifacts
artana run summaries run_123 --db .state.db --limit 20
artana run artifacts run_123 --db .state.db --json

# Scaffold a local starter (enforced by default)
artana init ./starter --profile enforced
artana init ./starter-dev --profile dev
```

Output shape:

* `run list`: one run id per line
* `run tail`: tab-separated `seq timestamp event_type parent_step_key`
* `verify-ledger`: `valid` or `invalid` (exit code 0/1)
* `status`: run status with last event metadata (`--json` for structured output)
* `summaries`: recent run summaries (`--type` and `--limit` filters)
* `artifacts`: latest artifact key/value snapshots (`--json` supported)

Use `--dsn postgresql://...` for shared Postgres deployments.

---

# 🧠 Final Production Mental Model

In distributed production:

| Layer        | Responsibility                 |
| ------------ | ------------------------------ |
| EventStore   | Source of truth                |
| Kernel       | Deterministic execution        |
| Harness      | Discipline & incremental logic |
| Middleware   | Enforcement                    |
| Workers      | Stateless executors            |
| Orchestrator | Scheduling                     |

Artana is:

> A deterministic execution substrate that survives crashes, drift, and scaling.

---

# 🏁 Production Readiness Checklist

Before deploying:

* [ ] All tools idempotent
* [ ] step_key stable
* [ ] replay_policy chosen intentionally
* [ ] KernelPolicy.enforced/enforced_v2 configured intentionally
* [ ] Run lease strategy defined for multi-worker runners
* [ ] Budget limits configured
* [ ] Ledger verification tested
* [ ] Drift handling strategy decided
* [ ] Artifact schema versioned
* [ ] Observability dashboards configured

---

## You Should Now Be Able To

- Run Artana workers in distributed topologies with deterministic replay and leasing.
- Operate day-2 diagnostics with CLI status/summaries/artifacts flows.
- Plan safe rollout and upgrade strategies for multi-tenant deployments.

## Next Chapter

Continue to [Chapter 6: OS-Grade Safety V2 and Harness Reality](./Chapter6.md) for intent plans, approvals, invariants, and safety-policy operations.

 
