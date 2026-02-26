# Documentation Guide

Artana docs are organized as one recommended path from first local run to production governance.

## Start Here (Recommended Order)

1. [Chapter 1](./Chapter1.md) — First local success, core primitives, and mental model.
2. [Chapter 2](./Chapter2.md) — Durable harness discipline and supervisor composition.
3. [Chapter 3](./Chapter3.md) — Failure handling, replay policies, and recovery patterns.
4. [Chapter 4](./Chapter4.md) — Advanced orchestration and custom loop architecture.
5. [Chapter 5](./Chapter5.md) — Distributed operations and deployment runbook.
6. [Chapter 6](./Chapter6.md) — OS-grade safety policies and governance workflows.

## Code Block Contract

- `python`: standalone runnable scripts.
- `pycon`: in-context snippets that assume surrounding state.

If you want end-to-end runnable scripts, use [examples/README.md](../examples/README.md).

## Coverage Matrix

| API / Capability | Primary Doc Location(s) |
| --- | --- |
| Minimal local-first onboarding (`MockModelPort`) | Chapter 1, `examples/README.md` |
| Optional deterministic keys (`StepKey`) + auto-step-key wrapper behavior | Chapter 1, `docs/kernel_contracts.md` |
| Constructor ergonomics (positional `kernel` supported in high-level wrappers) | Chapter 1, Chapters 3–4 |
| Capability visibility helpers (`describe_capabilities`, `list_tools_for_tenant`) | Chapter 2, Chapter 6, `docs/kernel_contracts.md` |
| `@kernel.tool(side_effect=True)` + `ToolExecutionContext` | Chapter 1, Chapter 6, `docs/kernel_contracts.md` |
| Replay policy modes (`strict`, `allow_prompt_drift`, `fork_on_drift`) | Chapter 3, Chapter 4, `docs/kernel_contracts.md` |
| Safety policy and approvals (`enforced_v2`) | Chapter 4, Chapter 6, `docs/kernel_contracts.md` |
| Traceability and run summaries | Chapter 5, `docs/deep_traceability.md` |
| CLI operations (`list`, `tail`, `verify-ledger`, `status`, `summaries`, `artifacts`, `--json`) | Chapter 5, `docs/kernel_contracts.md` |
| CLI starter scaffolding (`artana init`, `--profile enforced|dev`) | Chapter 1, Chapter 5, `docs/kernel_contracts.md` |

## Core Reference Docs

- [Kernel contracts](./kernel_contracts.md)
- [Deep traceability](./deep_traceability.md)
- [Behavior index (generated)](./kernel_behavior_index.json)
