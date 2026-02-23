 Perfect.

Below is a clean, production-grade **Chapter 5: Distributed Scaling & Multi-Tenant Deployment** — aligned with your real architecture and written for copy-paste use.

This chapter assumes:

* Kernel is event-sourced
* Harness enforces discipline
* Middleware enforces policy
* Replay is deterministic
* Store is pluggable

---

# Chapter 5: Distributed Scaling & Multi-Tenant Deployment

This chapter demonstrates:

* Multi-tenant isolation
* Horizontal scaling patterns
* Worker architecture
* Queue-based execution
* Long-running harness recovery
* Deployment topology
* Production safety checklist

---

# 🏗️ Step 1 — Multi-Tenant Isolation (First-Class Concept)

In Artana, tenants are explicit.

Every run is tied to:

```python
TenantContext(
    tenant_id="tenant_name",
    capabilities=frozenset({...}),
    budget_usd_limit=...
)
```

Example:

```python
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

```python
kernel = ArtanaKernel(
    store=PostgresStore(...),   # shared DB
    model_port=LiteLLMAdapter(...),
    middleware=ArtanaKernel.default_middleware_stack(),
)
```

Workers can:

* Load any run
* Resume safely
* Replay deterministically
* Continue long-running harness

No in-memory coordination required.

---

# 🔁 Step 3 — Queue + Worker Architecture

Example using a simple async queue:

```python
import asyncio

task_queue = asyncio.Queue()

async def worker():
    while True:
        run_id, tenant = await task_queue.get()
        harness = DeploymentHarness(kernel=kernel, tenant=tenant)
        await harness.run(run_id)
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

```python
await harness.run("migration_run")
```

On restart:

```python
await harness.run("migration_run")
```

Because:

* TaskProgressSnapshot is persisted
* Tool resolutions are reconciled
* Partial states rejected
* step_key prevents duplication

Recovery is deterministic.

---

# 🗃️ Step 5 — Distributed Event Store (Postgres Pattern)

Replace SQLite with Postgres implementation:

```python
class PostgresStore(EventStore):
    ...
```

All kernel logic remains identical.

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

```python
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
  SUM(CAST(json_extract(payload_json, '$.cost_usd') AS REAL)) AS spend
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

* Use KernelPolicy.enforced()
* Enable PII scrubber
* Enforce quota
* Enforce capability guard
* Validate tool idempotency
* Monitor drift

Optional:

* Encrypt EventStore at rest
* Encrypt tool payloads
* Sign event_hash externally
* Stream ledger to SIEM

---

# 📦 Step 10 — Production Deployment Template

Minimal production instantiation:

```python
kernel = ArtanaKernel(
    store=PostgresStore("postgres://..."),
    model_port=LiteLLMAdapter(...),
    middleware=ArtanaKernel.default_middleware_stack(),
    policy=KernelPolicy.enforced(),
)
```

Workers:

```python
harness = MyHarness(kernel=kernel, tenant=tenant)
await harness.run(run_id)
```

That’s it.

Everything else is architecture.

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
* [ ] KernelPolicy.enforced enabled
* [ ] Budget limits configured
* [ ] Ledger verification tested
* [ ] Drift handling strategy decided
* [ ] Artifact schema versioned
* [ ] Observability dashboards configured

---

 