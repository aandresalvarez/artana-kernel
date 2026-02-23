# Strategy 2 Results (AutonomousAgent)

## Scope

This document captures evaluation results for `strategy2.py`, which uses `AutonomousAgent` for:
- explicit edge extraction
- derived-edge adjudication over deterministic graph candidates

The key run requested was:
- model: `gpt-5-mini`
- reasoning effort: `medium`
- verbosity: `medium`

## Run Configuration

- Script: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/strategy2.py`
- Samples: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/samples`
- Output dir: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy2_gpt5mini_medium`
- Model timeout: `180s`
- Command:

```bash
cd /Users/alvaro1/Documents/med13/artana/artana-kernel
set -a; source .env; set +a
uv run python evaluations/triplets/strategy2.py \
  --samples-dir evaluations/triplets/samples \
  --output-dir evaluations/triplets/results_strategy2_gpt5mini_medium \
  --model gpt-5-mini \
  --reasoning-effort medium \
  --verbosity medium \
  --timeout-seconds 180
```

## Headline Results (gpt-5-mini, medium/medium)

- Exact edge-set matches: **0/4**
- Micro-average: **Precision=0.218, Recall=0.286, F1=0.247**

### Per-Article Scores

| Article | Precision | Recall | F1 |
|---|---:|---:|---:|
| synthetic_article_1.txt | 0.700 | 0.933 | 0.800 |
| synthetic_article_2.txt | 0.000 | 0.000 | 0.000 |
| synthetic_article_3.txt | 0.077 | 0.100 | 0.087 |
| synthetic_article_4.txt | 0.234 | 0.288 | 0.259 |

## Quick Comparison vs Prior Strategy 2 Run

Reference prior run (`gpt-4o-mini` default settings):
- micro P/R/F1 = **0.322 / 0.259 / 0.287**

Current run (`gpt-5-mini`, medium/medium):
- micro P/R/F1 = **0.218 / 0.286 / 0.247**

Interpretation:
- Recall increased slightly.
- Precision decreased materially.
- Net F1 decreased.

## Observed Failure Modes

1. **Canonical naming mismatch remains dominant**
- Example patterns:
  - `Aine` vs `AINE`
  - `Disorder D` vs `DISORDER_D`
  - `... (NTC)` forms vs canonical label without parenthetical alias

2. **Over-generation in derived edges**
- Additional plausible but non-gold derived edges are emitted.
- Derived set includes chain restatements not expected by benchmark.

3. **Epistemic boundary leakage**
- Hypothesis/unsupported statements often appear in contradiction lists.
- Contradicted edge formatting still drifts from expected canonical target strings.

4. **Long-article instability (articles 3/4)**
- Higher entity/alias density increases normalization drift.
- More relation surface variants produce additional false positives.

## Learnings for Strategy 3

1. **Canonicalization must be strict and deterministic**
- Introduce a dedicated canonical-entity normalization pass post-extraction.
- Normalize case, punctuation, parenthetical aliases, and spacing to one canonical token.

2. **Constrain derived generation more aggressively**
- Allow only disease- or key-process-level derived outputs.
- Add deterministic gating rules to suppress intermediate node derivations.

3. **Tighten contradiction/hypothesis classifier**
- Require explicit negative/refutation cues for `contradicted_edges`.
- Default uncertain claims to `hypothesis_only`.

4. **Use two-pass normalization before scoring/output**
- Pass 1: AI extraction (recall-oriented).
- Pass 2: deterministic ontology + canonical mapper (precision-oriented).

## Artifacts

Generated outputs for this run:
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy2_gpt5mini_medium/synthetic_article_1.strategy2.json`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy2_gpt5mini_medium/synthetic_article_2.strategy2.json`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy2_gpt5mini_medium/synthetic_article_3.strategy2.json`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_strategy2_gpt5mini_medium/synthetic_article_4.strategy2.json`

## Conclusion

Strategy 2 validates the `AutonomousAgent` architecture for this task but does not yet meet benchmark quality. The immediate priority is stricter canonicalization + stronger deterministic constraints on derived and epistemic classification outputs.
