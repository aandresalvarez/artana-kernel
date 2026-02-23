# Strategy 1 Results (AI-First Baseline)

## Scope

This document captures the first **AI-first** run of `strategy1.py` after removing sample-specific deterministic shortcuts.

Goal:
- Evaluate extraction quality on synthetic benchmark articles.
- Measure precision/recall/F1 against provided gold files.
- Capture practical learnings for the next strategy iteration.

## Run Configuration

- Date: 2026-02-22 (local run timestamp from generated files)
- Script: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/strategy1.py`
- Samples: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/samples`
- Output dir: `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_ai_first`
- Model: `gpt-4o-mini` (default from script)
- Command used:

```bash
cd /Users/alvaro1/Documents/med13/artana/artana-kernel
set -a; source .env; set +a
uv run python evaluations/triplets/strategy1.py \
  --samples-dir evaluations/triplets/samples \
  --output-dir evaluations/triplets/results_ai_first
```

## Headline Results

- Exact edge-set matches: **0/4**
- Micro-average: **Precision=0.476, Recall=0.348, F1=0.402**

### Per-Article Scores

| Article | Precision | Recall | F1 |
|---|---:|---:|---:|
| synthetic_article_1.txt | 0.700 | 0.933 | 0.800 |
| synthetic_article_2.txt | 0.000 | 0.000 | 0.000 |
| synthetic_article_3.txt | 0.476 | 0.333 | 0.392 |
| synthetic_article_4.txt | 0.600 | 0.288 | 0.390 |

## What Went Wrong

### 1) Canonicalization drift (largest impact)

The model extracted valid semantics but with non-canonical names, causing set mismatches:
- `Aine` vs `AINE`
- `BEX-β` vs `BEX`
- `IP signature` vs `INFLAMMATORY_PROGRAM`
- `P-axis` vs `PAXIS`
- `mTORC3` vs `mTORC3 branch`

### 2) Relation label drift

Some relations were semantically close but wrong per ontology:
- Used `DOWNREGULATES` where gold expects `INHIBITS`
- Used `REGULATES` or direct edges where `PART_OF` is expected
- Introduced invalid edge direction in some chains

### 3) Derived-edge over/under generation

- Article 1: Over-generated derived edges (extra unsupported generalizations).
- Articles 3/4: Under-generated multi-hop derived edges (missed expected chains).
- Root cause: second model step for derived inference was too free-form.

### 4) Contradiction vs hypothesis boundary confusion

Contradicted/unsubstantiated/speculative statements were mixed:
- Some speculative claims promoted into contradiction lists.
- Some contradiction candidates were phrased with alias drift and missed exact match.

### 5) Scale sensitivity on complex long text

Article 4 shows substantial drop in recall (0.288), indicating context/coverage degradation under longer, denser article structure.

## Learnings

1. **Need explicit canonical entity resolver**
- Build an alias-to-canonical map in a dedicated normalization phase.
- Enforce canonical rewrite for all edges before scoring/output.

2. **Need strict relation mapper**
- Add deterministic post-mapping from near-synonyms to allowed ontology labels.
- Reject or remap unsupported relation labels before final output.

3. **Derived reasoning should be graph-constrained**
- Keep AI for extraction of explicit facts.
- Compute derived edges with deterministic sign-propagation and path rules over the explicit fact graph.
- This should reduce hallucinated derived edges and improve repeatability.

4. **Add contradiction/hypothesis gating rules**
- Use explicit lexical cues and confidence thresholds.
- Only promote to `contradicted_edges` when text contains strong refutation markers.
- Keep weak/uncertain language in `hypothesis_only`.

5. **Long-context strategy is required**
- Chunk-aware extraction + merge/dedupe pass for long documents.
- Then run canonicalization and final graph consolidation.

## Artifacts

Predicted outputs from this run:
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_ai_first/synthetic_article_1.strategy1.json`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_ai_first/synthetic_article_2.strategy1.json`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_ai_first/synthetic_article_3.strategy1.json`
- `/Users/alvaro1/Documents/med13/artana/artana-kernel/evaluations/triplets/results_ai_first/synthetic_article_4.strategy1.json`

## Conclusion

Strategy 1 successfully moved to an AI-first, unknown-article-compatible architecture, but accuracy is not yet sufficient for benchmark-grade extraction. The next iteration should focus on canonicalization, deterministic derived reasoning, and stricter epistemic boundary handling.
