# Strategy 4 Results (One-Call Durable Workflow)

## Scope

This file captures the current strategy 4 smoke run result for:

- Script: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/strategy4.py`
- Output: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy4_smoke/synthetic_article_1.strategy4.json`
- Gold: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/samples/gold_synthetic_1.json`

## Run Configuration

- Model: `gpt-5-mini`
- Reasoning effort: `medium`
- Verbosity: `medium`
- Pause: enabled (`--pause-before-finalize --auto-resume`)
- Replay checks: enabled (`--assert-replay`)
- Durable DB: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/.state_strategy4.db`

## Command Used

```bash
cd /Users/alvaro1/Documents/med13/artana/artana-kernel
set -a; source .env; set +a
uv run python evaluations/triplets/strategy4.py \
  --article evaluations/triplets/samples/synthetic_article_1.txt \
  --gold evaluations/triplets/samples/gold_synthetic_1.json \
  --output-dir evaluations/triplets/results_strategy4_smoke \
  --model gpt-5-mini \
  --reasoning-effort medium \
  --verbosity medium \
  --timeout-seconds 180 \
  --score-mode gold_keys \
  --pause-before-finalize \
  --auto-resume \
  --state-db evaluations/triplets/.state_strategy4.db \
  --reset-state-db \
  --assert-replay
```

## Metrics (Article 1 Smoke)

- Exact edge-set match: **1/1**
- Micro-average: **Precision=1.000, Recall=1.000, F1=1.000**

Per-key:

- `fact_edges`: 11/11 correct
- `derived_edges`: 4/4 correct

## Durability / Replay Evidence

For run id `triplets_strategy4::synthetic_article_1::a023eff07ca0` in the durable SQLite state:

- `run_started`: 1
- `model_requested`: 1
- `model_completed`: 1
- `workflow_step_requested`: 2
- `workflow_step_completed`: 2
- `pause_requested`: 1

This confirms the one-call model design with deterministic workflow steps and successful replay/invariant checks.

## Learnings

1. Architecture is stable, reproducible, and low-call-cost.
2. The article 1 derived-edge gap was fixed by tightening deterministic disease-reduction logic.
3. Remaining broader benchmark gaps are now mostly extraction/canonicalization drift on harder articles.

## Improvement Path

1. Adjust disease-derivation gating to recover `DRUG_R -> DERIVED_REDUCES -> DISEASE_D`.
2. Add source/path constraints to prevent `GENE_B -> DERIVED_REDUCES -> DISEASE_D`.
3. Add targeted unit tests for these two edge cases before broader reruns.
