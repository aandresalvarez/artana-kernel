from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

from artana.agent import KernelModelClient
from artana.events import (
    EventType,
    KernelEvent,
    PauseRequestedPayload,
    WorkflowStepCompletedPayload,
)
from artana.kernel import (
    ArtanaKernel,
    KernelPolicy,
    WorkflowContext,
    json_step_serde,
    pydantic_step_serde,
)
from artana.models import TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.ports.model_types import LiteLLMCompletionFn
from artana.store import SQLiteStore

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES_DIR = ROOT_DIR / "samples"
DEFAULT_STATE_DB = ROOT_DIR / ".state_strategy4.db"
STRATEGY_VERSION = "v2"
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
    {"GENE", "PROTEIN", "PATHWAY", "PROCESS", "DISEASE", "DRUG", "OTHER"}
)

RELATION_SYNONYMS = {
    "ACTIVATE": "ACTIVATES",
    "ACTIVATION": "ACTIVATES",
    "INHIBIT": "INHIBITS",
    "INHIBITION": "INHIBITS",
    "UPREGULATE": "UPREGULATES",
    "UPREGULATION": "UPREGULATES",
    "DOWNREGULATE": "DOWNREGULATES",
    "DOWNREGULATION": "DOWNREGULATES",
    "ASSOCIATED": "ASSOCIATED_WITH",
    "ASSOCIATION_WITH": "ASSOCIATED_WITH",
    "ASSOCIATES_WITH": "ASSOCIATED_WITH",
    "PARTOF": "PART_OF",
    "PART_OF_PATHWAY": "PART_OF",
    "BELONGS_TO": "PART_OF",
    "REGULATE": "REGULATES",
    "DERIVED_ASSOCIATED": "DERIVED_ASSOCIATED_WITH",
    "DERIVED_ASSOCIATION_WITH": "DERIVED_ASSOCIATED_WITH",
    "DERIVED_DOWNREGULATE": "DERIVED_DOWNREGULATES",
    "DERIVED_INHIBIT": "DERIVED_INHIBITS",
    "DERIVED_REDUCE": "DERIVED_REDUCES",
    "DERIVED_REGULATE": "DERIVED_REGULATES",
    "POSSIBLE_LINK": "POSSIBLE_ASSOCIATION",
    "POSSIBLE_ASSOCIATED_WITH": "POSSIBLE_ASSOCIATION",
}

SIGN_BY_RELATION = {
    "ACTIVATES": 1,
    "UPREGULATES": 1,
    "PART_OF": 1,
    "INHIBITS": -1,
    "DOWNREGULATES": -1,
}


class Triplet(BaseModel):
    src: str
    relation: str
    dst: str


class EntityRecord(BaseModel):
    canonical_name: str
    type: Literal["GENE", "PROTEIN", "PATHWAY", "PROCESS", "DISEASE", "DRUG", "OTHER"] = (
        "OTHER"
    )
    aliases: list[str] = Field(default_factory=list)
    note: str | None = None


class ExtractionResult(BaseModel):
    benchmark_name: str = ""
    entities: list[EntityRecord] = Field(default_factory=list)
    fact_edges: list[Triplet] = Field(default_factory=list)
    contradicted_edges: list[Triplet] = Field(default_factory=list)
    hypothesis_only: list[Triplet] = Field(default_factory=list)


class MathResult(BaseModel):
    benchmark_name: str = ""
    entities: dict[str, dict[str, object]] = Field(default_factory=dict)
    fact_edges: list[list[str]] = Field(default_factory=list)
    contradicted_edges: list[list[str]] = Field(default_factory=list)
    hypothesis_only: list[list[str]] = Field(default_factory=list)
    derived_edges: list[list[str]] = Field(default_factory=list)


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
    reasoning_effort: Literal["low", "medium", "high"] | None,
    verbosity: Literal["low", "medium", "high"] | None,
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
    base = base.replace("-", "_").replace("/", "_").replace(" ", "_")
    base = re.sub(r"_+", "_", base)
    return RELATION_SYNONYMS.get(base, base)


def _canonicalize_entity_name_for_type(value: str, *, entity_type: str) -> str:
    base = _strip_parenthetical(_normalize_entity_name(value))
    if not base:
        return ""

    if entity_type in {"GENE", "PROTEIN", "DRUG", "PATHWAY"}:
        # Compact symbolic tokens are generally represented in uppercase.
        if re.fullmatch(r"[A-Za-z]{2,5}", base):
            return base.upper()
        axis_loop_match = re.fullmatch(r"([A-Za-z]{1,2})[- ]?(axis|loop)", base, flags=re.I)
        if axis_loop_match is not None:
            return f"{axis_loop_match.group(1).upper()}{axis_loop_match.group(2).upper()}"
        return base

    if entity_type == "DISEASE":
        disease_match = re.fullmatch(r"(disease|disorder)\s+([A-Za-z0-9]+)", base, flags=re.I)
        if disease_match is not None:
            stem = disease_match.group(2).upper()
            if len(stem) <= 3:
                return f"{disease_match.group(1).upper()}_{stem}"
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
        entity_type = record.type if record.type in ENTITY_TYPES else "OTHER"
        raw_canonical = _normalize_entity_name(record.canonical_name)
        canonical = _canonicalize_entity_name_for_type(
            raw_canonical,
            entity_type=entity_type,
        )
        if not canonical:
            continue
        aliases: list[str] = []
        seen_aliases: set[str] = set()

        # Keep the original extracted canonical label as alias when canonicalization rewrites it.
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
    triplets: Sequence[Triplet],
    *,
    alias_to_canonical: dict[str, str],
    ranked_alias_keys: list[tuple[str, str]],
    allowed_relations: frozenset[str] | None = None,
    default_relation: str | None = None,
    coerce_to_derived_prefix: bool = False,
) -> list[list[str]]:
    seen: set[tuple[str, str, str]] = set()
    normalized: list[list[str]] = []

    for edge in triplets:
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
        if coerce_to_derived_prefix and not relation.startswith("DERIVED_"):
            relation = _normalize_relation(f"DERIVED_{relation}")
        if (
            default_relation is not None
            and allowed_relations is not None
            and relation not in allowed_relations
        ):
            relation = default_relation
        if allowed_relations is not None and relation not in allowed_relations:
            continue

        key = (src, relation, dst)
        if key in seen:
            continue
        seen.add(key)
        normalized.append([src, relation, dst])
    return normalized


def _entity_type(name: str, entities: dict[str, dict[str, object]]) -> str:
    entry = entities.get(name)
    if entry is None:
        return "OTHER"
    entity_type = entry.get("type")
    if isinstance(entity_type, str) and entity_type in ENTITY_TYPES:
        return entity_type
    return "OTHER"


def _is_disease(entity_name: str, entities: dict[str, dict[str, object]]) -> bool:
    if _entity_type(entity_name, entities) == "DISEASE":
        return True
    upper = entity_name.upper()
    return "DISEASE" in upper or "DISORDER" in upper or "SYNDROME" in upper


def _is_process_like(entity_name: str, entities: dict[str, dict[str, object]]) -> bool:
    if _entity_type(entity_name, entities) == "PROCESS":
        return True
    upper = entity_name.upper()
    keywords = ("PROGRAM", "RESPONSE", "CASCADE", "MODULE", "SIGNATURE")
    return any(keyword in upper for keyword in keywords)


def _generate_derived_edges(
    *,
    fact_edges: list[list[str]],
    entities: dict[str, dict[str, object]],
    max_depth: int = 6,
) -> list[list[str]]:
    signed_adjacency: dict[str, list[tuple[str, int]]] = {}
    outgoing_by_source: dict[str, set[str]] = {}
    association_targets: dict[str, set[str]] = {}
    direct_negative_targets: dict[str, set[str]] = {}

    for src, relation, dst in fact_edges:
        outgoing_by_source.setdefault(src, set()).add(relation)
        if relation in SIGN_BY_RELATION:
            signed_adjacency.setdefault(src, []).append((dst, SIGN_BY_RELATION[relation]))
        if relation in {"INHIBITS", "DOWNREGULATES"}:
            direct_negative_targets.setdefault(src, set()).add(dst)
        if relation == "ASSOCIATED_WITH":
            association_targets.setdefault(src, set()).add(dst)

    source_nodes: set[str] = set()
    for source, relations in outgoing_by_source.items():
        if not any(
            rel in {"ACTIVATES", "INHIBITS", "UPREGULATES", "DOWNREGULATES"}
            for rel in relations
        ):
            continue
        if _entity_type(source, entities) in {"GENE", "DRUG", "PROCESS"}:
            source_nodes.add(source)

    derived: set[tuple[str, str, str]] = set()
    for source in source_nodes:
        source_type = _entity_type(source, entities)
        queue: deque[tuple[str, int, int]] = deque([(source, 1, 0)])
        best_depth: dict[tuple[str, int], int] = {(source, 1): 0}

        while queue:
            node, sign, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nxt, edge_sign in signed_adjacency.get(node, []):
                next_sign = sign * edge_sign
                next_depth = depth + 1
                key = (nxt, next_sign)
                previous = best_depth.get(key)
                if previous is not None and previous <= next_depth:
                    continue
                best_depth[key] = next_depth
                queue.append((nxt, next_sign, next_depth))

        for (target, sign), depth in best_depth.items():
            if target == source:
                continue

            if (
                depth >= 2
                and _is_process_like(target, entities)
                and not _is_disease(target, entities)
            ):
                if sign > 0:
                    derived.add((source, "DERIVED_REGULATES", target))
                else:
                    derived.add((source, "DERIVED_DOWNREGULATES", target))

            if target in association_targets:
                for disease in association_targets[target]:
                    source_direct_negatives = direct_negative_targets.get(source, set())
                    has_direct_negative_to_target = target in source_direct_negatives
                    if source_type == "GENE" and sign < 0 and not has_direct_negative_to_target:
                        continue
                    if sign < 0 and (depth >= 2 or has_direct_negative_to_target):
                        derived.add((source, "DERIVED_REDUCES", disease))
                        continue
                    if depth < 2:
                        continue
                    if sign > 0:
                        derived.add((source, "DERIVED_ASSOCIATED_WITH", disease))

    return [list(item) for item in sorted(derived)]


def _run_math(extraction: ExtractionResult) -> MathResult:
    entities = _normalize_entities(extraction.entities)
    alias_to_canonical, ranked_alias_keys = _build_alias_to_canonical(entities)
    fact_edges = _normalize_triplets(
        extraction.fact_edges,
        alias_to_canonical=alias_to_canonical,
        ranked_alias_keys=ranked_alias_keys,
        allowed_relations=FACT_RELATIONS,
    )
    contradicted_edges = _normalize_triplets(
        extraction.contradicted_edges,
        alias_to_canonical=alias_to_canonical,
        ranked_alias_keys=ranked_alias_keys,
        allowed_relations=FACT_RELATIONS,
    )
    hypothesis_only = _normalize_triplets(
        extraction.hypothesis_only,
        alias_to_canonical=alias_to_canonical,
        ranked_alias_keys=ranked_alias_keys,
        allowed_relations=frozenset({HYPOTHESIS_RELATION}),
        default_relation=HYPOTHESIS_RELATION,
    )
    derived_edges = _generate_derived_edges(
        fact_edges=fact_edges,
        entities=entities,
    )
    return MathResult(
        benchmark_name=_normalize_text(extraction.benchmark_name),
        entities=entities,
        fact_edges=fact_edges,
        contradicted_edges=contradicted_edges,
        hypothesis_only=hypothesis_only,
        derived_edges=derived_edges,
    )


def _assemble_payload(
    *,
    article_name: str,
    math_result: MathResult,
    include_epistemic_edges: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "benchmark_name": math_result.benchmark_name or article_name,
        "fact_edges": math_result.fact_edges,
        "derived_edges": math_result.derived_edges,
    }
    if math_result.entities:
        payload["entities"] = math_result.entities
    if include_epistemic_edges and math_result.contradicted_edges:
        payload["contradicted_edges"] = math_result.contradicted_edges
    if include_epistemic_edges and math_result.hypothesis_only:
        payload["hypothesis_only"] = math_result.hypothesis_only
    return payload


def _build_extraction_prompt(article_text: str) -> str:
    return (
        "Extract an explicit knowledge graph from this article in one pass.\n\n"
        "Return JSON that matches the schema exactly.\n\n"
        "Rules:\n"
        "1) fact_edges: explicit supported statements only.\n"
        "2) contradicted_edges: explicit refuted/retracted/disproven claims.\n"
        "3) hypothesis_only: speculative/unproven claims only.\n"
        "4) Keep contradicted and speculative claims out of fact_edges.\n"
        "5) Build entities with canonical_name + aliases.\n"
        "6) Use only fact relations:\n"
        "   ACTIVATES, INHIBITS, PART_OF, UPREGULATES, DOWNREGULATES,\n"
        "   ASSOCIATED_WITH, REGULATES.\n"
        "7) For hypothesis_only use POSSIBLE_ASSOCIATION when possible.\n"
        "8) Do not include multi-hop derived edges.\n"
        "9) Prefer precision over recall when ambiguous.\n\n"
        f"ARTICLE:\n{article_text.strip()}"
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
    score_mode: Literal["all", "gold_keys"],
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


def _workflow_run_id(
    *,
    article_path: Path,
    model: str,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    verbosity: Literal["low", "medium", "high"] | None,
    include_epistemic_edges: bool,
    pause_before_finalize: bool,
) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", article_path.stem)
    fingerprint = "|".join(
        (
            STRATEGY_VERSION,
            safe_stem,
            model,
            reasoning_effort or "none",
            verbosity or "none",
            f"epistemic={int(include_epistemic_edges)}",
            f"pause={int(pause_before_finalize)}",
        )
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"triplets_strategy4::{safe_stem}::{digest}"


def _count_event_types(events: Sequence[KernelEvent]) -> Counter[EventType]:
    return Counter(event.event_type for event in events)


def _completed_step_names(events: Sequence[KernelEvent]) -> set[str]:
    names: set[str] = set()
    for event in events:
        if event.event_type != EventType.WORKFLOW_STEP_COMPLETED:
            continue
        payload = event.payload
        if isinstance(payload, WorkflowStepCompletedPayload):
            names.add(payload.step_name)
    return names


def _assert_replay_invariants(
    *,
    run_id: str,
    first_output: dict[str, object],
    second_output: dict[str, object],
    events_after_first: Sequence[KernelEvent],
    events_after_second: Sequence[KernelEvent],
    pause_before_finalize: bool,
) -> None:
    if second_output != first_output:
        raise AssertionError(f"Replay output mismatch for run_id={run_id}.")
    if len(events_after_second) != len(events_after_first):
        raise AssertionError(
            "Replay should not append duplicate events "
            f"(run_id={run_id}, first={len(events_after_first)}, "
            f"second={len(events_after_second)})."
        )

    counts = _count_event_types(events_after_first)
    if counts[EventType.RUN_STARTED] != 1:
        raise AssertionError(f"Expected exactly one run_started event for run_id={run_id}.")
    if counts[EventType.MODEL_REQUESTED] < 1 or counts[EventType.MODEL_COMPLETED] < 1:
        raise AssertionError(f"Expected at least one model cycle for run_id={run_id}.")
    if counts[EventType.MODEL_REQUESTED] != counts[EventType.MODEL_COMPLETED]:
        raise AssertionError(
            f"Model requested/completed mismatch for run_id={run_id}: "
            f"{counts[EventType.MODEL_REQUESTED]} vs {counts[EventType.MODEL_COMPLETED]}."
        )
    if counts[EventType.WORKFLOW_STEP_REQUESTED] != counts[EventType.WORKFLOW_STEP_COMPLETED]:
        raise AssertionError(
            f"Workflow step requested/completed mismatch for run_id={run_id}: "
            f"{counts[EventType.WORKFLOW_STEP_REQUESTED]} vs "
            f"{counts[EventType.WORKFLOW_STEP_COMPLETED]}."
        )
    if counts[EventType.WORKFLOW_STEP_COMPLETED] < 2:
        raise AssertionError(
            f"Expected at least two completed workflow steps (math/finalize) for run_id={run_id}."
        )
    if pause_before_finalize and counts[EventType.PAUSE_REQUESTED] < 1:
        raise AssertionError(
            f"Expected at least one pause_requested event for run_id={run_id} when pause enabled."
        )
    completed_steps = _completed_step_names(events_after_first)
    if "math" not in completed_steps or "finalize" not in completed_steps:
        raise AssertionError(
            f"Expected completed workflow steps {{'math', 'finalize'}} for run_id={run_id}; "
            f"got {sorted(completed_steps)}."
        )


def _build_kernel(
    *,
    db_path: Path,
    timeout_seconds: float,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    verbosity: Literal["low", "medium", "high"] | None,
) -> tuple[ArtanaKernel, SQLiteStore]:
    store = SQLiteStore(str(db_path))
    kernel = ArtanaKernel(
        store=store,
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
    return kernel, store


async def _run_workflow_once(
    *,
    kernel: ArtanaKernel,
    store: SQLiteStore,
    run_id: str,
    tenant: TenantContext,
    article_path: Path,
    article_text: str,
    model: str,
    pause_before_finalize: bool,
    auto_resume: bool,
    include_epistemic_edges: bool,
) -> tuple[dict[str, object], list[KernelEvent]]:
    chat = KernelModelClient(kernel=kernel)

    async def my_workflow(ctx: WorkflowContext) -> dict[str, object]:
        # 1) Atomic LLM call (single extraction pass)
        extraction_step = await chat.step(
            run_id=ctx.run_id,
            tenant=tenant,
            model=model,
            prompt=_build_extraction_prompt(article_text),
            output_schema=ExtractionResult,
            step_key="extract",
        )
        extraction = extraction_step.output

        # 2) Deterministic Python logic (cached via workflow step events)
        async def run_math() -> MathResult:
            return _run_math(extraction)

        math = await ctx.step(
            name="math",
            action=run_math,
            serde=pydantic_step_serde(MathResult),
        )

        # 3) Durable pause (optional)
        if pause_before_finalize:
            history = await store.get_events_for_run(ctx.run_id)
            already_review_paused = any(
                event.event_type == EventType.PAUSE_REQUESTED
                and isinstance(event.payload, PauseRequestedPayload)
                and event.payload.step_key == "review"
                for event in history
            )
            if not already_review_paused:
                await ctx.pause(
                    reason="Review deterministic graph before finalize.",
                    context=math,
                    step_key="review",
                )

        async def finalize() -> dict[str, object]:
            return _assemble_payload(
                article_name=article_path.stem,
                math_result=math,
                include_epistemic_edges=include_epistemic_edges,
            )

        payload = await ctx.step(
            name="finalize",
            action=finalize,
            serde=json_step_serde(),
        )
        if not isinstance(payload, dict):
            raise TypeError("finalize step returned non-dict payload.")
        return cast(dict[str, object], payload)

    run_result = await kernel.run_workflow(
        run_id=run_id,
        tenant=tenant,
        workflow=my_workflow,
    )
    if run_result.status == "paused":
        if not auto_resume:
            raise RuntimeError(
                "Workflow paused. Re-run with --auto-resume to continue automatically."
            )
        run_result = await kernel.run_workflow(
            run_id=run_result.run_id,
            tenant=tenant,
            workflow=my_workflow,
        )
    if run_result.status != "complete" or run_result.output is None:
        raise RuntimeError("Workflow did not complete successfully.")

    output = run_result.output
    if not isinstance(output, dict):
        raise TypeError("Workflow output must be a dict payload.")
    events = await store.get_events_for_run(run_id)
    if not await store.verify_run_chain(run_id):
        raise AssertionError(f"Event chain verification failed for run_id={run_id}.")
    return cast(dict[str, object], output), events


async def _predict_article_with_workflow(
    *,
    article_path: Path,
    model: str,
    tenant: TenantContext,
    state_db: Path,
    timeout_seconds: float,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    verbosity: Literal["low", "medium", "high"] | None,
    pause_before_finalize: bool,
    auto_resume: bool,
    include_epistemic_edges: bool,
    assert_replay: bool,
) -> dict[str, object]:
    article_text = article_path.read_text(encoding="utf-8")
    run_id = _workflow_run_id(
        article_path=article_path,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        include_epistemic_edges=include_epistemic_edges,
        pause_before_finalize=pause_before_finalize,
    )

    first_kernel, first_store = _build_kernel(
        db_path=state_db,
        timeout_seconds=timeout_seconds,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )
    try:
        first_output, events_after_first = await _run_workflow_once(
            kernel=first_kernel,
            store=first_store,
            run_id=run_id,
            tenant=tenant,
            article_path=article_path,
            article_text=article_text,
            model=model,
            pause_before_finalize=pause_before_finalize,
            auto_resume=auto_resume,
            include_epistemic_edges=include_epistemic_edges,
        )
    finally:
        await first_kernel.close()

    if not assert_replay:
        return first_output

    second_kernel, second_store = _build_kernel(
        db_path=state_db,
        timeout_seconds=timeout_seconds,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )
    try:
        second_output, events_after_second = await _run_workflow_once(
            kernel=second_kernel,
            store=second_store,
            run_id=run_id,
            tenant=tenant,
            article_path=article_path,
            article_text=article_text,
            model=model,
            pause_before_finalize=pause_before_finalize,
            auto_resume=auto_resume,
            include_epistemic_edges=include_epistemic_edges,
        )
    finally:
        await second_kernel.close()

    _assert_replay_invariants(
        run_id=run_id,
        first_output=first_output,
        second_output=second_output,
        events_after_first=events_after_first,
        events_after_second=events_after_second,
        pause_before_finalize=pause_before_finalize,
    )
    return first_output


async def _run(
    *,
    article_paths: Sequence[Path],
    model: str,
    tenant: TenantContext,
    state_db: Path,
    samples_dir: Path,
    explicit_gold: Path | None,
    output_dir: Path | None,
    max_diff_preview: int,
    timeout_seconds: float,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    verbosity: Literal["low", "medium", "high"] | None,
    pause_before_finalize: bool,
    auto_resume: bool,
    include_epistemic_edges: bool,
    assert_replay: bool,
    score_mode: Literal["all", "gold_keys"],
) -> tuple[EvalResult, ...]:
    results: list[EvalResult] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for article_path in article_paths:
        predicted = await _predict_article_with_workflow(
            article_path=article_path,
            model=model,
            tenant=tenant,
            state_db=state_db,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            pause_before_finalize=pause_before_finalize,
            auto_resume=auto_resume,
            include_epistemic_edges=include_epistemic_edges,
            assert_replay=assert_replay,
        )
        if output_dir is not None:
            output_path = output_dir / f"{article_path.stem}.strategy4.json"
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
        description="Strategy4: one-call workflow + deterministic math in durable workflow runtime."
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
        "--state-db",
        type=Path,
        default=DEFAULT_STATE_DB,
        help=(
            "Durable SQLite DB path used for workflow replay checks "
            f"(default: {DEFAULT_STATE_DB})."
        ),
    )
    parser.add_argument(
        "--reset-state-db",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete --state-db before execution to force fresh runs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("ARTANA_MODEL", "gpt-5-mini"),
        help="Model id for LiteLLMAdapter (default: ARTANA_MODEL or gpt-5-mini).",
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
        "--pause-before-finalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pause workflow before finalize step for durable manual review.",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If workflow pauses, automatically resume and complete.",
    )
    parser.add_argument(
        "--assert-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each article workflow twice and assert deterministic replay invariants.",
    )
    parser.add_argument(
        "--include-epistemic-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include contradicted_edges/hypothesis_only in output payload.",
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
            "OPENAI_API_KEY is required for strategy4 execution. "
            "Load environment first (e.g. `set -a; source .env; set +a`)."
        )
    if args.gold is not None and args.article is None:
        raise ValueError("--gold requires --article.")
    if args.budget_usd <= 0:
        raise ValueError("--budget-usd must be > 0.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0.")

    samples_dir = args.samples_dir.resolve()
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory does not exist: {samples_dir}")
    state_db = args.state_db.resolve()
    if state_db.exists() and state_db.is_dir():
        raise IsADirectoryError(f"--state-db points to a directory, expected a file: {state_db}")
    state_db.parent.mkdir(parents=True, exist_ok=True)
    if args.reset_state_db and state_db.exists():
        state_db.unlink()
    print(f"Using durable state DB: {state_db}")

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
        model=args.model,
        tenant=tenant,
        state_db=state_db,
        samples_dir=samples_dir,
        explicit_gold=args.gold.resolve() if args.gold is not None else None,
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        max_diff_preview=max(1, args.max_diff_preview),
        timeout_seconds=args.timeout_seconds,
        reasoning_effort=cast(
            Literal["low", "medium", "high"],
            args.reasoning_effort,
        ),
        verbosity=cast(
            Literal["low", "medium", "high"],
            args.verbosity,
        ),
        pause_before_finalize=args.pause_before_finalize,
        auto_resume=args.auto_resume,
        include_epistemic_edges=args.include_epistemic_edges,
        assert_replay=args.assert_replay,
        score_mode=cast(
            Literal["all", "gold_keys"],
            args.score_mode,
        ),
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
