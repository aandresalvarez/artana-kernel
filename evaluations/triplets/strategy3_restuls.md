# Strategy 3 Results (AutonomousAgent)

## Scope

This file captures the observed results for `strategy3.py` runs currently saved under:

- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy3_smoke`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy3_patch_smoke`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy3_patch_gpt5mini_mediummedium_smoke`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy3_gpt5mini_highhigh_smoke`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy3_gpt52_smoke`

All of the above are smoke evaluations on article 1 (`synthetic_article_1.txt`) against `gold_synthetic_1.json`.

## Snapshot Metrics

| Run folder | Precision | Recall | F1 | Exact match |
|---|---:|---:|---:|---:|
| `results_strategy3_smoke` | 0.938 | 1.000 | **0.968** | 0/1 |
| `results_strategy3_patch_smoke` | 0.933 | 0.933 | 0.933 | 0/1 |
| `results_strategy3_patch_gpt5mini_mediummedium_smoke` | 0.933 | 0.933 | 0.933 | 0/1 |
| `results_strategy3_gpt5mini_highhigh_smoke` | 0.933 | 0.933 | 0.933 | 0/1 |
| `results_strategy3_gpt52_smoke` | 1.000 | 0.867 | 0.929 | 0/1 |

## Best Observed Run

- Best F1 was in `results_strategy3_smoke`: **P=0.938, R=1.000, F1=0.968**
- It still failed exact match due to one extra derived edge:
  - `("GENE_B", "DERIVED_REDUCES", "DISEASE_D")`

## Repeated Failure Pattern

Across strategy 3 variants, fact edges were consistently correct; failures were concentrated in `derived_edges`:

- Extra derived edge appears frequently:
  - `("GENE_B", "DERIVED_REDUCES", "DISEASE_D")`
- Some variants missed expected disease-level associations:
  - `("GENE_A", "DERIVED_ASSOCIATED_WITH", "DISEASE_D")`
- In the `gpt5.2` smoke run, derived recall dropped further (2 missing derived edges).

## Learnings

1. Strategy 3 is strong on explicit extraction and canonicalization.
2. The main error source is the model-mediated derived adjudication step.
3. Derived-edge decisions are sensitive to model/config, so results are less stable than deterministic logic.

## Improvement Path

1. Make derived-edge acceptance deterministic after candidate generation.
2. Keep AI for explicit fact extraction, but not final derived inclusion/exclusion.
3. Add fixed evidence-path checks for each derived edge before output.
4. Re-run full 4-article evaluation after the deterministic derived update.
