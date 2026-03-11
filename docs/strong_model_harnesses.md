# Strong-Model Harnesses

This page shows the intended layering after the harness refresh:

```text
Task
↓
WorkspaceState
↓
Strong model decides
↓
Artana executes safely
↓
Verification / gates
↓
Artifacts / HarnessOutcome
```

## General vs Domain

### General harness runtime

Reusable across domains:

- `StrongModelHarness`
- `StrongModelAgentHarness`
- `WorkspaceState`
- `HarnessOutcome`
- replay-safe model/tool execution
- verification and acceptance gates
- artifact persistence
- pause/resume and approval workflows

### Domain harness templates

These specialize the workspace shape and tool posture:

- `ResearchHarness`
- `CodingHarness`
- `SupportHarness`
- `DataHarness`
- `ActionHarness`
- `ReviewHarness`
- `CurationHarness`

## Side By Side

| Layer | Question | Workspace focus | Typical tools | Example |
| --- | --- | --- | --- | --- |
| Generic strong-model runtime | "How should durable model-driven work run?" | generic artifacts, constraints, open tasks, allowed tools | whatever the tenant exposes | `StrongModelAgentHarness` |
| Coding | "How do I fix this repo issue?" | repo, files, failing tests, patches | read/edit/test tools | `examples/10_live_manual_agent_harness.py` |
| Research | "What does the evidence say?" | question, graph, evidence count, contradictions | literature, graph, scoring tools | `examples/12_research_strong_model_harness.py` |
| Support | "How do I resolve this ticket?" | customer profile, ticket history, policy, allowed actions | customer, policy, history tools | `examples/13_support_strong_model_harness.py` |
| Data | "Why did this pipeline fail?" | datasets, schema, quality rules, logs | logs, schema, metrics tools | `examples/14_data_diagnostic_harness.py` |
| Governed action | "Can I safely commit this side effect?" | subject, limits, approvals, intent plan | approval and side-effect tools | `examples/11_durable_release_harness.py` |

## Framing

Artana is not a research harness or a coding harness.

Artana provides:

- a generic strong-model harness runtime
- a generic workspace abstraction
- domain templates layered on top

That keeps the kernel general while still making the common operating modes opinionated and easy to adopt.
