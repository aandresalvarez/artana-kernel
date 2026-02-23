from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from artana.agent import KernelModelClient
from artana.kernel import ArtanaKernel, KernelPolicy
from artana.models import TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.ports.model_types import LiteLLMCompletionFn
from artana.store import SQLiteStore

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES_DIR = ROOT_DIR / "samples"
EDGE_KEYS = ("fact_edges", "derived_edges", "contradicted_edges", "hypothesis_only")

FACT_RELATIONS = frozenset(
    {
        "ACTIVATES",
        "INHIBITS",
        "PART_OF",
        "UPREGULATES",
        "DOWNREGULATES",
        "ASSOCIATED_WITH",
        "REGULATES",
    }
)
DERIVED_RELATIONS = frozenset(
    {
        "DERIVED_REGULATES",
        "DERIVED_ASSOCIATED_WITH",
        "DERIVED_DOWNREGULATES",
        "DERIVED_REDUCES",
        "DERIVED_INHIBITS",
    }
)
HYPOTHESIS_RELATION = "POSSIBLE_ASSOCIATION"
ENTITY_TYPES = frozenset(
    {
        "GENE",
        "PROTEIN",
        "PATHWAY",
        "PROCESS",
        "DISEASE",
        "DRUG",
        "OTHER",
    }
)

RELATION_SYNONYMS = {
    "ACTIVATE": "ACTIVATES",
    "ACTIVATION": "ACTIVATES",
    "ACTIVATES_PHOSPHORYLATION_OF": "ACTIVATES",
    "INHIBIT": "INHIBITS",
    "INHIBITION": "INHIBITS",
    "UPREGULATE": "UPREGULATES",
    "UPREGULATION": "UPREGULATES",
    "UPREGULATES_TRANSCRIPTION_OF": "UPREGULATES",
    "DOWNREGULATE": "DOWNREGULATES",
    "DOWNREGULATION": "DOWNREGULATES",
    "ASSOCIATED": "ASSOCIATED_WITH",
    "ASSOCIATION_WITH": "ASSOCIATED_WITH",
    "ASSOCIATES_WITH": "ASSOCIATED_WITH",
    "ASSOCIATED_WITH_SEVERITY_OF": "ASSOCIATED_WITH",
    "PARTOF": "PART_OF",
    "BELONGS_TO": "PART_OF",
    "IN_PATHWAY": "PART_OF",
    "REGULATE": "REGULATES",
    "DERIVED_ASSOCIATED": "DERIVED_ASSOCIATED_WITH",
    "DERIVED_ASSOCIATION_WITH": "DERIVED_ASSOCIATED_WITH",
    "DERIVED_DOWNREGULATE": "DERIVED_DOWNREGULATES",
    "DERIVED_INHIBIT": "DERIVED_INHIBITS",
    "DERIVED_REDUCE": "DERIVED_REDUCES",
    "DERIVED_REGULATE": "DERIVED_REGULATES",
    "POSSIBLE_LINK": "POSSIBLE_ASSOCIATION",
    "POSSIBLE_ASSOCIATED_WITH": "POSSIBLE_ASSOCIATION",
    "MAY_INFLUENCE": "POSSIBLE_ASSOCIATION",
    "MAY_DIRECTLY_INFLUENCE": "POSSIBLE_ASSOCIATION",
}


class Triplet(BaseModel):
    src: str
    relation: str
    dst: str


class EvidenceTriplet(BaseModel):
    src: str
    relation: str
    dst: str
    evidence: str = ""
    confidence: float | None = None


class EntityRecord(BaseModel):
    canonical_name: str
    type: str = "OTHER"
    aliases: list[str] = Field(default_factory=list)
    note: str | None = None


class FreeExtractionResult(BaseModel):
    benchmark_name: str = ""
    entities: list[EntityRecord] = Field(default_factory=list)
    fact_edges: list[EvidenceTriplet] = Field(default_factory=list)
    derived_edges: list[EvidenceTriplet] = Field(default_factory=list)
    contradicted_edges: list[EvidenceTriplet] = Field(default_factory=list)
    hypothesis_only: list[EvidenceTriplet] = Field(default_factory=list)


class ReviewedExtractionResult(BaseModel):
    benchmark_name: str = ""
    entities: list[EntityRecord] = Field(default_factory=list)
    fact_edges: list[EvidenceTriplet] = Field(default_factory=list)
    derived_edges: list[EvidenceTriplet] = Field(default_factory=list)
    contradicted_edges: list[EvidenceTriplet] = Field(default_factory=list)
    hypothesis_only: list[EvidenceTriplet] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class KeyMetrics:
    found: int
    expected: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    missing: tuple[tuple[str, str, str], ...]
    extra: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True, slots=True)
class EvalResult:
    article_path: Path
    gold_path: Path | None
    per_key: dict[str, KeyMetrics]
    micro: KeyMetrics
    predicted: dict[str, object]

    @property
    def passed(self) -> bool:
        return self.micro.fp == 0 and self.micro.fn == 0


def _build_completion_fn(
    *,
    reasoning_effort: str | None,
    verbosity: str | None,
) -> LiteLLMCompletionFn:
    from litellm import acompletion

    async def _completion_fn(
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        kwargs: dict[str, object] = {}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if verbosity is not None:
            kwargs["verbosity"] = verbosity
        return await acompletion(
            model=model,
            messages=messages,
            response_format=response_format,
            tools=tools,
            **kwargs,
        )

    return _completion_fn


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _strip_parenthetical(value: str) -> str:
    return _normalize_text(re.sub(r"\([^)]*\)", "", value))


def _normalize_entity_name(value: str) -> str:
    cleaned = _normalize_text(value)
    if not cleaned:
        return ""
    cleaned = cleaned.strip(" \"'`.,;:")
    return _normalize_text(cleaned)


def _entity_alias_key(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).upper()


def _normalize_relation(value: str) -> str:
    base = _normalize_text(value).upper()
    if not base:
        return ""
    base = base.replace("-", "_").replace("/", "_").replace(" ", "_")
    base = re.sub(r"_+", "_", base)
    return RELATION_SYNONYMS.get(base, base)


def _normalized_entity_type(value: str) -> str:
    normalized = _normalize_relation(value)
    if normalized in ENTITY_TYPES:
        return normalized
    return "OTHER"


def _canonicalize_entity_name_for_type(value: str, *, entity_type: str) -> str:
    base = _strip_parenthetical(_normalize_entity_name(value))
    if not base:
        return ""

    if entity_type in {"GENE", "PROTEIN", "DRUG"}:
        if re.fullmatch(r"[A-Za-z]{2,5}", base):
            return base.upper()
        return base

    if entity_type == "PATHWAY":
        axis_loop_match = re.fullmatch(r"([A-Za-z]{1,6})[- ]?(axis|loop)", base, flags=re.I)
        if axis_loop_match is not None:
            return f"{axis_loop_match.group(1).upper()}{axis_loop_match.group(2).upper()}"
        if base.lower().endswith(" cascade"):
            candidate = base[:-8].strip()
            if candidate:
                return candidate
        return base

    if entity_type == "DISEASE":
        disease_match = re.fullmatch(r"(disease|disorder)\s+([A-Za-z0-9]+)", base, flags=re.I)
        if disease_match is not None:
            stem = disease_match.group(2).upper()
            if len(stem) <= 4:
                return f"{disease_match.group(1).upper()}_{stem}"
        if base == base.lower():
            return " ".join(piece.capitalize() for piece in base.split())
        return base

    if entity_type == "PROCESS":
        upper = base.upper()
        if "IP SIGNATURE" in upper or "INFLAMMATORY PROGRAM" in upper:
            return "INFLAMMATORY_PROGRAM"
        if "METABOLIC STRESS" in upper:
            return "METABOLIC_STRESS"
        return base

    return base


def _normalize_entities(records: Sequence[EntityRecord]) -> dict[str, dict[str, object]]:
    entities: dict[str, dict[str, object]] = {}
    for record in records:
        entity_type = _normalized_entity_type(record.type)
        raw_canonical = _normalize_entity_name(record.canonical_name)
        canonical = _canonicalize_entity_name_for_type(
            raw_canonical,
            entity_type=entity_type,
        )
        if not canonical:
            continue

        aliases: list[str] = []
        seen_aliases: set[str] = set()
        alias_candidates: list[str] = list(record.aliases)
        if raw_canonical and raw_canonical != canonical:
            alias_candidates.append(raw_canonical)

        for alias in alias_candidates:
            alias_name = _normalize_entity_name(alias)
            if not alias_name or alias_name == canonical:
                continue
            if alias_name in seen_aliases:
                continue
            seen_aliases.add(alias_name)
            aliases.append(alias_name)
        entry: dict[str, object] = {"type": entity_type, "aliases": aliases}
        if record.note:
            entry["note"] = _normalize_text(record.note)
        entities[canonical] = entry
    return entities


def _build_alias_to_canonical(
    entities: dict[str, dict[str, object]],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    mapping: dict[str, str] = {}
    for canonical, entry in entities.items():
        for variant in {
            canonical,
            _strip_parenthetical(canonical),
            canonical.upper(),
            canonical.lower(),
        }:
            key = _entity_alias_key(variant)
            if key:
                mapping[key] = canonical

        aliases_obj = entry.get("aliases", [])
        if not isinstance(aliases_obj, list):
            continue
        for alias in aliases_obj:
            if isinstance(alias, str):
                for variant in {
                    alias,
                    _strip_parenthetical(alias),
                    alias.upper(),
                    alias.lower(),
                }:
                    key = _entity_alias_key(variant)
                    if key:
                        mapping[key] = canonical
    ranked = sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)
    return mapping, ranked


def _resolve_canonical_entity(
    raw_name: str,
    *,
    alias_to_canonical: dict[str, str],
    ranked_alias_keys: list[tuple[str, str]],
) -> str:
    normalized = _normalize_entity_name(raw_name)
    no_parens = _strip_parenthetical(normalized)
    if not no_parens:
        return ""

    for candidate in (normalized, no_parens):
        key = _entity_alias_key(candidate)
        mapped = alias_to_canonical.get(key)
        if mapped is not None:
            return mapped

    key = _entity_alias_key(no_parens)
    for alias_key, canonical in ranked_alias_keys:
        if alias_key in key or key in alias_key:
            return canonical
    return no_parens


def _normalize_triplets(
    triplets: Sequence[EvidenceTriplet],
    *,
    alias_to_canonical: dict[str, str],
    ranked_alias_keys: list[tuple[str, str]],
    allowed_relations: frozenset[str] | None = None,
    default_relation: str | None = None,
    coerce_to_derived_prefix: bool = False,
    evidence_required: bool = True,
    min_evidence_chars: int = 24,
    min_confidence: float = 0.55,
) -> list[list[str]]:
    seen: set[tuple[str, str, str]] = set()
    normalized: list[list[str]] = []

    for edge in triplets:
        evidence = _normalize_text(edge.evidence)
        if evidence_required and len(evidence) < min_evidence_chars:
            continue
        if edge.confidence is not None and edge.confidence < min_confidence:
            continue

        src = _resolve_canonical_entity(
            edge.src,
            alias_to_canonical=alias_to_canonical,
            ranked_alias_keys=ranked_alias_keys,
        )
        dst = _resolve_canonical_entity(
            edge.dst,
            alias_to_canonical=alias_to_canonical,
            ranked_alias_keys=ranked_alias_keys,
        )
        if not src or not dst:
            continue

        relation = _normalize_relation(edge.relation)
        if coerce_to_derived_prefix and relation and not relation.startswith("DERIVED_"):
            relation = _normalize_relation(f"DERIVED_{relation}")

        if (
            default_relation is not None
            and allowed_relations is not None
            and relation not in allowed_relations
        ):
            relation = default_relation
        if allowed_relations is not None and relation not in allowed_relations:
            continue
        if not relation:
            continue

        key = (src, relation, dst)
        if key in seen:
            continue
        seen.add(key)
        normalized.append([src, relation, dst])
    return normalized


def _assemble_payload(
    *,
    article_name: str,
    extracted: ReviewedExtractionResult,
    evidence_required: bool,
    min_evidence_chars: int,
    min_confidence: float,
) -> dict[str, object]:
    entities = _normalize_entities(extracted.entities)
    alias_to_canonical, ranked_alias_keys = _build_alias_to_canonical(entities)

    fact_edges = _normalize_triplets(
        extracted.fact_edges,
        alias_to_canonical=alias_to_canonical,
        ranked_alias_keys=ranked_alias_keys,
        allowed_relations=FACT_RELATIONS,
        evidence_required=evidence_required,
        min_evidence_chars=min_evidence_chars,
        min_confidence=min_confidence,
    )
    derived_edges = _normalize_triplets(
        extracted.derived_edges,
        alias_to_canonical=alias_to_canonical,
        ranked_alias_keys=ranked_alias_keys,
        allowed_relations=DERIVED_RELATIONS,
        coerce_to_derived_prefix=True,
        evidence_required=evidence_required,
        min_evidence_chars=min_evidence_chars,
        min_confidence=min_confidence,
    )
    contradicted_edges = _normalize_triplets(
        extracted.contradicted_edges,
        alias_to_canonical=alias_to_canonical,
        ranked_alias_keys=ranked_alias_keys,
        allowed_relations=FACT_RELATIONS,
        evidence_required=evidence_required,
        min_evidence_chars=min_evidence_chars,
        min_confidence=min_confidence,
    )
    hypothesis_only = _normalize_triplets(
        extracted.hypothesis_only,
        alias_to_canonical=alias_to_canonical,
        ranked_alias_keys=ranked_alias_keys,
        allowed_relations=frozenset({HYPOTHESIS_RELATION}),
        default_relation=HYPOTHESIS_RELATION,
        evidence_required=evidence_required,
        min_evidence_chars=min_evidence_chars,
        min_confidence=min_confidence,
    )

    payload: dict[str, object] = {
        "benchmark_name": _normalize_text(extracted.benchmark_name) or article_name,
        "fact_edges": fact_edges,
        "derived_edges": derived_edges,
    }
    if entities:
        payload["entities"] = entities
    if contradicted_edges:
        payload["contradicted_edges"] = contradicted_edges
    if hypothesis_only:
        payload["hypothesis_only"] = hypothesis_only
    return payload


def _build_extract_prompt(article_text: str) -> str:
    return (
        "Extract a biomedical knowledge graph from the article.\n\n"
        "Return JSON matching the schema exactly.\n\n"
        "Guidance:\n"
        "1) You may use free relation labels in this first pass.\n"
        "2) Include evidence text for every edge (short quote or paraphrase).\n"
        "3) Include confidence in [0,1] for every edge.\n"
        "4) fact_edges: explicit supported statements only.\n"
        "5) derived_edges: multi-step implications inferred from explicit evidence.\n"
        "6) contradicted_edges: explicit refuted/retracted/disproven claims.\n"
        "7) hypothesis_only: speculative or weakly supported claims.\n"
        "8) Prefer precision over recall when uncertain.\n\n"
        f"ARTICLE:\n{article_text.strip()}"
    )


def _build_review_prompt(
    *,
    article_text: str,
    free_result: FreeExtractionResult,
) -> str:
    raw_payload = free_result.model_dump(mode="json")
    return (
        "Review and normalize the extracted graph.\n\n"
        "You MUST keep only supported edges.\n\n"
        "Rules:\n"
        "1) Drop any edge not supported by explicit evidence in article text.\n"
        "2) Keep evidence text for every retained edge.\n"
        "3) Normalize entity naming (canonical + aliases).\n"
        "4) Map FACT and CONTRADICTED relations to this set only:\n"
        "   ACTIVATES, INHIBITS, PART_OF, UPREGULATES, DOWNREGULATES,\n"
        "   ASSOCIATED_WITH, REGULATES.\n"
        "5) Map DERIVED relations to this set only:\n"
        "   DERIVED_REGULATES, DERIVED_ASSOCIATED_WITH,\n"
        "   DERIVED_DOWNREGULATES, DERIVED_REDUCES, DERIVED_INHIBITS.\n"
        "6) For hypothesis_only relation use POSSIBLE_ASSOCIATION.\n"
        "7) Keep derived edges conservative; require a plausible multi-step chain.\n"
        "8) Do not include study metadata edges (cohort size, assay names, binding constants).\n"
        "9) Prefer higher precision over higher recall.\n\n"
        f"ARTICLE:\n{article_text.strip()}\n\n"
        f"FREE_EXTRACTION_JSON:\n{json.dumps(raw_payload, ensure_ascii=False)}"
    )


def _read_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _discover_articles(samples_dir: Path) -> list[Path]:
    return sorted(samples_dir.glob("synthetic_article_*.txt"))


def _article_id_from_filename(article_path: Path) -> int | None:
    match = re.search(r"synthetic_article_(\d+)\.txt$", article_path.name)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_gold_path(
    *,
    article_path: Path,
    samples_dir: Path,
    explicit_gold: Path | None,
) -> Path | None:
    if explicit_gold is not None:
        return explicit_gold
    article_id = _article_id_from_filename(article_path)
    if article_id is None:
        return None
    candidate = samples_dir / f"gold_synthetic_{article_id}.json"
    if candidate.exists():
        return candidate
    return None


def _triplet_set(payload: dict[str, object], key: str) -> set[tuple[str, str, str]]:
    raw = payload.get(key, [])
    if not isinstance(raw, list):
        return set()
    parsed: set[tuple[str, str, str]] = set()
    for item in raw:
        if not isinstance(item, list) or len(item) != 3:
            continue
        src, relation, dst = item
        if all(isinstance(part, str) for part in (src, relation, dst)):
            parsed.add((src, relation, dst))
    return parsed


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _compare_with_gold(
    *,
    article_path: Path,
    predicted: dict[str, object],
    gold_path: Path | None,
    score_mode: str,
) -> EvalResult:
    if gold_path is None:
        empty = KeyMetrics(
            found=0,
            expected=0,
            tp=0,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            missing=(),
            extra=(),
        )
        per_key: dict[str, KeyMetrics] = {}
        for key in EDGE_KEYS:
            found = len(_triplet_set(predicted, key))
            per_key[key] = KeyMetrics(
                found=found,
                expected=0,
                tp=0,
                fp=0,
                fn=0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                missing=(),
                extra=(),
            )
        return EvalResult(
            article_path=article_path,
            gold_path=None,
            per_key=per_key,
            micro=empty,
            predicted=predicted,
        )

    gold = _read_json(gold_path)
    if score_mode == "all":
        keys_to_score: tuple[str, ...] = EDGE_KEYS
    else:
        dynamic = tuple(key for key in EDGE_KEYS if key in gold)
        keys_to_score = dynamic if dynamic else ("fact_edges", "derived_edges")

    per_key: dict[str, KeyMetrics] = {}
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for key in keys_to_score:
        found_set = _triplet_set(predicted, key)
        expected_set = _triplet_set(gold, key)
        tp = len(found_set & expected_set)
        fp = len(found_set - expected_set)
        fn = len(expected_set - found_set)
        precision, recall, f1 = _prf(tp, fp, fn)
        per_key[key] = KeyMetrics(
            found=len(found_set),
            expected=len(expected_set),
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            missing=tuple(sorted(expected_set - found_set)),
            extra=tuple(sorted(found_set - expected_set)),
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision, recall, f1 = _prf(tp_total, fp_total, fn_total)
    micro = KeyMetrics(
        found=tp_total + fp_total,
        expected=tp_total + fn_total,
        tp=tp_total,
        fp=fp_total,
        fn=fn_total,
        precision=precision,
        recall=recall,
        f1=f1,
        missing=(),
        extra=(),
    )
    return EvalResult(
        article_path=article_path,
        gold_path=gold_path,
        per_key=per_key,
        micro=micro,
        predicted=predicted,
    )


def _print_eval(result: EvalResult, *, max_diff_preview: int) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(
        f"[{status}] {result.article_path.name} | "
        f"P={result.micro.precision:.3f} R={result.micro.recall:.3f} F1={result.micro.f1:.3f}"
    )
    for key in EDGE_KEYS:
        metrics = result.per_key.get(key)
        if metrics is None:
            continue
        print(
            f"  - {key}: found={metrics.found} expected={metrics.expected} "
            f"tp={metrics.tp} fp={metrics.fp} fn={metrics.fn} "
            f"P={metrics.precision:.3f} R={metrics.recall:.3f} F1={metrics.f1:.3f}"
        )
        if metrics.missing:
            print(f"    missing: {list(metrics.missing[:max_diff_preview])}")
        if metrics.extra:
            print(f"    extra: {list(metrics.extra[:max_diff_preview])}")


async def _predict_article_with_atomic_calls(
    *,
    article_path: Path,
    extract_model: str,
    review_model: str,
    tenant: TenantContext,
    timeout_seconds: float,
    reasoning_effort: str | None,
    verbosity: str | None,
    evidence_required: bool,
    min_evidence_chars: int,
    min_confidence: float,
) -> dict[str, object]:
    article_text = article_path.read_text(encoding="utf-8")
    tmp_root = Path(tempfile.mkdtemp(prefix="artana_triplets_strategy7_"))
    db_path = tmp_root / "state.db"
    kernel = ArtanaKernel(
        store=SQLiteStore(str(db_path)),
        model_port=LiteLLMAdapter(
            completion_fn=_build_completion_fn(
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            ),
            timeout_seconds=timeout_seconds,
            max_retries=2,
            fail_on_unknown_cost=False,
        ),
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )
    chat = KernelModelClient(kernel=kernel)

    try:
        run = await kernel.start_run(tenant=tenant)
        run_id = run.run_id

        # Atomic LLM call #1: free extraction pass
        free_step = await chat.step(
            run_id=run_id,
            tenant=tenant,
            model=extract_model,
            prompt=_build_extract_prompt(article_text),
            output_schema=FreeExtractionResult,
            step_key="extract",
        )
        free_result = free_step.output

        # Atomic LLM call #2: review/normalization pass
        review_step = await chat.step(
            run_id=run_id,
            tenant=tenant,
            model=review_model,
            prompt=_build_review_prompt(
                article_text=article_text,
                free_result=free_result,
            ),
            output_schema=ReviewedExtractionResult,
            step_key="review",
        )
        reviewed_result = review_step.output

        return _assemble_payload(
            article_name=article_path.stem,
            extracted=reviewed_result,
            evidence_required=evidence_required,
            min_evidence_chars=min_evidence_chars,
            min_confidence=min_confidence,
        )
    finally:
        await kernel.close()
        shutil.rmtree(tmp_root, ignore_errors=True)


async def _run(
    *,
    article_paths: Sequence[Path],
    extract_model: str,
    review_model: str,
    tenant: TenantContext,
    samples_dir: Path,
    explicit_gold: Path | None,
    output_dir: Path | None,
    max_diff_preview: int,
    timeout_seconds: float,
    reasoning_effort: str | None,
    verbosity: str | None,
    score_mode: str,
    evidence_required: bool,
    min_evidence_chars: int,
    min_confidence: float,
) -> tuple[EvalResult, ...]:
    results: list[EvalResult] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for article_path in article_paths:
        predicted = await _predict_article_with_atomic_calls(
            article_path=article_path,
            extract_model=extract_model,
            review_model=review_model,
            tenant=tenant,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            evidence_required=evidence_required,
            min_evidence_chars=min_evidence_chars,
            min_confidence=min_confidence,
        )
        if output_dir is not None:
            output_path = output_dir / f"{article_path.stem}.strategy7.json"
            output_path.write_text(
                json.dumps(predicted, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        gold_path = _resolve_gold_path(
            article_path=article_path,
            samples_dir=samples_dir,
            explicit_gold=explicit_gold if len(article_paths) == 1 else None,
        )
        result = _compare_with_gold(
            article_path=article_path,
            predicted=predicted,
            gold_path=gold_path,
            score_mode=score_mode,
        )
        _print_eval(result, max_diff_preview=max_diff_preview)
        results.append(result)
    return tuple(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strategy7: two atomic chat calls (extract + review) with evidence gating."
    )
    parser.add_argument(
        "--article",
        type=Path,
        help="Single article .txt to analyze. If omitted, runs all synthetic samples.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        help="Optional explicit gold .json (only valid with --article).",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help=f"Samples directory. Default: {DEFAULT_SAMPLES_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to persist predicted JSON outputs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional fallback model id for both passes when --extract-model/"
            "--review-model are not set."
        ),
    )
    parser.add_argument(
        "--extract-model",
        type=str,
        default=os.getenv("ARTANA_EXTRACT_MODEL"),
        help=(
            "Model id for extraction pass (default: ARTANA_EXTRACT_MODEL, "
            "else ARTANA_MODEL, else gpt-5.2)."
        ),
    )
    parser.add_argument(
        "--review-model",
        type=str,
        default=os.getenv("ARTANA_REVIEW_MODEL"),
        help=(
            "Model id for review pass (default: ARTANA_REVIEW_MODEL, "
            "else ARTANA_MODEL, else gpt-5.2)."
        ),
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default="triplets-eval",
        help="Tenant id for Artana run context.",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=10.0,
        help="Tenant budget ceiling in USD.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.getenv("ARTANA_REASONING_EFFORT", "medium"),
        help="Reasoning effort forwarded to model provider.",
    )
    parser.add_argument(
        "--verbosity",
        choices=("low", "medium", "high"),
        default=os.getenv("ARTANA_VERBOSITY", "medium"),
        help="Verbosity forwarded to model provider.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("ARTANA_TIMEOUT_SECONDS", "180")),
        help="Model call timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--score-mode",
        choices=("all", "gold_keys"),
        default=os.getenv("ARTANA_SCORE_MODE", "gold_keys"),
        help=(
            "Scoring mode: 'gold_keys' only scores edge groups present in gold file; "
            "'all' scores all edge groups."
        ),
    )
    parser.add_argument(
        "--evidence-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only edges that include non-trivial evidence text.",
    )
    parser.add_argument(
        "--min-evidence-chars",
        type=int,
        default=24,
        help="Minimum evidence text length used by evidence gate.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum confidence used by evidence gate (if confidence is present).",
    )
    parser.add_argument(
        "--max-diff-preview",
        type=int,
        default=5,
        help="How many missing/extra triplets to print per edge group.",
    )
    return parser.parse_args()


async def _main_async() -> int:
    args = _parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for strategy7 execution. "
            "Load environment first (e.g. `set -a; source .env; set +a`)."
        )
    if args.gold is not None and args.article is None:
        raise ValueError("--gold requires --article.")
    if args.budget_usd <= 0:
        raise ValueError("--budget-usd must be > 0.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0.")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1.")
    if args.min_evidence_chars < 0:
        raise ValueError("--min-evidence-chars must be >= 0.")

    fallback_model = args.model or os.getenv("ARTANA_MODEL")
    extract_model = args.extract_model or fallback_model or "gpt-5.2"
    review_model = args.review_model or fallback_model or "gpt-5.2"
    print(f"Using models: extract={extract_model} review={review_model}")

    samples_dir = args.samples_dir.resolve()
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory does not exist: {samples_dir}")

    if args.article is not None:
        article_paths = [args.article.resolve()]
    else:
        article_paths = _discover_articles(samples_dir)
        if not article_paths:
            raise FileNotFoundError(f"No synthetic_article_*.txt files found in {samples_dir}")

    tenant = TenantContext(
        tenant_id=args.tenant_id,
        capabilities=frozenset(),
        budget_usd_limit=args.budget_usd,
    )
    results = await _run(
        article_paths=article_paths,
        extract_model=extract_model,
        review_model=review_model,
        tenant=tenant,
        samples_dir=samples_dir,
        explicit_gold=args.gold.resolve() if args.gold is not None else None,
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        max_diff_preview=max(1, args.max_diff_preview),
        timeout_seconds=args.timeout_seconds,
        reasoning_effort=cast(str, args.reasoning_effort),
        verbosity=cast(str, args.verbosity),
        score_mode=cast(str, args.score_mode),
        evidence_required=args.evidence_required,
        min_evidence_chars=args.min_evidence_chars,
        min_confidence=args.min_confidence,
    )

    with_gold = [result for result in results if result.gold_path is not None]
    if not with_gold:
        print("\nNo gold files provided; metrics are counts only.")
        return 0

    tp = sum(result.micro.tp for result in with_gold)
    fp = sum(result.micro.fp for result in with_gold)
    fn = sum(result.micro.fn for result in with_gold)
    precision, recall, f1 = _prf(tp, fp, fn)
    passed_count = sum(1 for result in with_gold if result.passed)
    print(
        "\nSummary: "
        f"{passed_count}/{len(with_gold)} exact edge-set matches | "
        f"micro P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
    )
    return 0 if passed_count == len(with_gold) else 1


def main() -> None:
    raise SystemExit(asyncio.run(_main_async()))


if __name__ == "__main__":
    main()
