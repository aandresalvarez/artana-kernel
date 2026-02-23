from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

from artana.harness import HarnessContext, IncrementalTaskHarness, TaskUnit
from artana.kernel import ArtanaKernel, KernelPolicy, ReplayPolicy
from artana.models import TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.ports.model_types import LiteLLMCompletionFn
from artana.store import SQLiteStore

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES_DIR = ROOT_DIR / "samples"
DEFAULT_STATE_DB = ROOT_DIR / ".state_strategy9.db"
STRATEGY_VERSION = "v1"
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


ClaimBucket = Literal[
    "fact_edges",
    "derived_edges",
    "contradicted_edges",
    "hypothesis_only",
]


class ClaimRecord(BaseModel):
    key: str = ""
    bucket: ClaimBucket = "fact_edges"
    src: str
    relation: str
    dst: str
    evidence: str = ""
    confidence: float | None = None


class DebateState(BaseModel):
    claims: list[ClaimRecord] = Field(default_factory=list)
    counterclaims: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0


class Proposal(BaseModel):
    claims: list[ClaimRecord] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class Critique(BaseModel):
    invalid_claims: list[str] = Field(default_factory=list)
    missing_assumptions: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    requires_tool_check: list[str] = Field(default_factory=list)


class VerifyClaimArgs(BaseModel):
    article_text: str
    claim_key: str
    src: str
    relation: str
    dst: str
    evidence: str = ""


class Repair(BaseModel):
    revised_claims: list[ClaimRecord] = Field(default_factory=list)
    dropped_claims: list[str] = Field(default_factory=list)
    confidence_delta: float = 0.0


class Evaluation(BaseModel):
    convergence_score: float = 0.0
    unresolved_issues: list[str] = Field(default_factory=list)
    should_continue: bool = False


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


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _dedupe_texts(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in values:
        normalized = _normalize_text(raw)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _normalize_claim_bucket(value: str) -> ClaimBucket:
    if value in EDGE_KEYS:
        return cast(ClaimBucket, value)
    return "fact_edges"


def _claim_key(*, bucket: str, src: str, relation: str, dst: str) -> str:
    normalized_bucket = _normalize_claim_bucket(bucket)
    normalized_src = _normalize_entity_name(src)
    normalized_relation = _normalize_relation(relation)
    normalized_dst = _normalize_entity_name(dst)
    return "|".join((normalized_bucket, normalized_src, normalized_relation, normalized_dst))


def _normalize_claim(record: ClaimRecord) -> ClaimRecord:
    bucket = _normalize_claim_bucket(record.bucket)
    src = _normalize_entity_name(record.src)
    relation = _normalize_relation(record.relation)
    dst = _normalize_entity_name(record.dst)
    evidence = _normalize_text(record.evidence)
    key = _normalize_text(record.key) or _claim_key(
        bucket=bucket,
        src=src,
        relation=relation,
        dst=dst,
    )
    confidence = _clamp01(record.confidence) if record.confidence is not None else None
    return ClaimRecord(
        key=key,
        bucket=bucket,
        src=src,
        relation=relation,
        dst=dst,
        evidence=evidence,
        confidence=confidence,
    )


def _claims_to_map(claims: Sequence[ClaimRecord]) -> dict[str, ClaimRecord]:
    mapping: dict[str, ClaimRecord] = {}
    for claim in claims:
        normalized = _normalize_claim(claim)
        if not normalized.src or not normalized.relation or not normalized.dst:
            continue
        mapping[normalized.key] = normalized
    return mapping


def _merge_claims(state: DebateState, claims: Sequence[ClaimRecord]) -> None:
    merged = _claims_to_map(state.claims)
    merged.update(_claims_to_map(claims))
    state.claims = list(merged.values())


def _replace_claims(state: DebateState, claims: Sequence[ClaimRecord]) -> None:
    state.claims = list(_claims_to_map(claims).values())


def _drop_claims(state: DebateState, dropped_keys: Sequence[str]) -> None:
    if not dropped_keys:
        return
    drop_set = {_normalize_text(key) for key in dropped_keys if _normalize_text(key)}
    if not drop_set:
        return
    state.claims = [claim for claim in state.claims if claim.key not in drop_set]


def _safe_step_fragment(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return compact[:24] if compact else "claim"


def _normalize_for_match(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[\u2010-\u2015]", "-", lowered)
    lowered = re.sub(r"[^a-z0-9\s-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _phrase_in_text(*, phrase: str, text: str) -> bool:
    normalized_phrase = _normalize_for_match(phrase)
    normalized_text = _normalize_for_match(text)
    return bool(normalized_phrase) and normalized_phrase in normalized_text


def _verify_claim_against_article(
    *,
    article_text: str,
    claim: ClaimRecord,
) -> dict[str, object]:
    src_present = _phrase_in_text(phrase=claim.src, text=article_text)
    dst_present = _phrase_in_text(phrase=claim.dst, text=article_text)
    evidence_present = True
    if claim.evidence:
        evidence_present = _phrase_in_text(phrase=claim.evidence, text=article_text)
    supported = src_present and dst_present and evidence_present
    return {
        "claim_key": claim.key,
        "src_present": src_present,
        "dst_present": dst_present,
        "evidence_present": evidence_present,
        "supported": supported,
    }


def _parse_tool_result_json(result_json: str) -> dict[str, object]:
    try:
        parsed = json.loads(result_json)
    except json.JSONDecodeError:
        return {"raw_result": result_json}
    if isinstance(parsed, dict):
        return cast(dict[str, object], parsed)
    return {"raw_result": parsed}


def _strategy9_run_id(
    *,
    article_path: Path,
    proposer_model: str,
    critic_model: str,
    repair_model: str,
    judge_model: str,
    max_rounds: int,
    replay_policy: ReplayPolicy,
) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", article_path.stem)
    fingerprint = "|".join(
        (
            STRATEGY_VERSION,
            safe_stem,
            proposer_model,
            critic_model,
            repair_model,
            judge_model,
            f"rounds={max_rounds}",
            f"replay={replay_policy}",
        )
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"triplets_strategy9::{safe_stem}::{digest}"


def _build_kernel(
    *,
    db_path: Path,
    timeout_seconds: float,
    reasoning_effort: str | None,
    verbosity: str | None,
) -> ArtanaKernel:
    return ArtanaKernel(
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


def _claims_to_reviewed_result(
    *,
    article_name: str,
    claims: Sequence[ClaimRecord],
) -> ReviewedExtractionResult:
    fact_edges: list[EvidenceTriplet] = []
    derived_edges: list[EvidenceTriplet] = []
    contradicted_edges: list[EvidenceTriplet] = []
    hypothesis_only: list[EvidenceTriplet] = []

    for claim in claims:
        normalized = _normalize_claim(claim)
        if not normalized.src or not normalized.relation or not normalized.dst:
            continue
        edge = EvidenceTriplet(
            src=normalized.src,
            relation=normalized.relation,
            dst=normalized.dst,
            evidence=normalized.evidence,
            confidence=normalized.confidence,
        )
        if normalized.bucket == "fact_edges":
            fact_edges.append(edge)
        elif normalized.bucket == "derived_edges":
            derived_edges.append(edge)
        elif normalized.bucket == "contradicted_edges":
            contradicted_edges.append(edge)
        else:
            hypothesis_only.append(edge)

    return ReviewedExtractionResult(
        benchmark_name=article_name,
        entities=[],
        fact_edges=fact_edges,
        derived_edges=derived_edges,
        contradicted_edges=contradicted_edges,
        hypothesis_only=hypothesis_only,
    )


def _build_proposer_prompt(
    *,
    article_text: str,
    state: DebateState,
    round_idx: int,
    max_rounds: int,
) -> str:
    state_json = json.dumps(state.model_dump(mode="json"), ensure_ascii=False)
    return (
        "Role: Proposer agent in a structured biomedical extraction debate.\n"
        "Return ONLY JSON matching the schema.\n\n"
        "Your job:\n"
        "1) Propose high-precision claim candidates.\n"
        "2) Each claim must include bucket, src, relation, dst, evidence, confidence.\n"
        "3) bucket must be one of: fact_edges, derived_edges, "
        "contradicted_edges, hypothesis_only.\n"
        "4) Use relation sets:\n"
        "   FACT/CONTRADICTED: ACTIVATES, INHIBITS, PART_OF, UPREGULATES, "
        "DOWNREGULATES, ASSOCIATED_WITH, REGULATES.\n"
        "   DERIVED: DERIVED_REGULATES, DERIVED_ASSOCIATED_WITH, "
        "DERIVED_DOWNREGULATES, DERIVED_REDUCES, DERIVED_INHIBITS.\n"
        "   HYPOTHESIS: POSSIBLE_ASSOCIATION.\n"
        "5) Keep only claims grounded in the article.\n"
        "6) Prefer precision over recall.\n\n"
        f"ROUND: {round_idx}/{max_rounds}\n"
        f"CURRENT_STATE_JSON:\n{state_json}\n\n"
        f"ARTICLE:\n{article_text.strip()}"
    )


def _build_critic_prompt(
    *,
    article_text: str,
    state: DebateState,
    proposal: Proposal,
    round_idx: int,
    max_rounds: int,
) -> str:
    state_json = json.dumps(state.model_dump(mode="json"), ensure_ascii=False)
    proposal_json = json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False)
    return (
        "Role: Critic agent.\n"
        "Return ONLY JSON matching the schema.\n\n"
        "Your job:\n"
        "1) Find invalid or weak claims by key.\n"
        "2) List missing assumptions and alternative explanations.\n"
        "3) In requires_tool_check include claim keys that require deterministic "
        "lexical verification.\n"
        "4) Do not invent claim keys.\n\n"
        f"ROUND: {round_idx}/{max_rounds}\n"
        f"CURRENT_STATE_JSON:\n{state_json}\n\n"
        f"LATEST_PROPOSAL_JSON:\n{proposal_json}\n\n"
        f"ARTICLE:\n{article_text.strip()}"
    )


def _build_repair_prompt(
    *,
    article_text: str,
    state: DebateState,
    critique: Critique,
    tool_results: Sequence[dict[str, object]],
    round_idx: int,
    max_rounds: int,
) -> str:
    state_json = json.dumps(state.model_dump(mode="json"), ensure_ascii=False)
    critique_json = json.dumps(critique.model_dump(mode="json"), ensure_ascii=False)
    tool_results_json = json.dumps(list(tool_results), ensure_ascii=False)
    return (
        "Role: Repair agent.\n"
        "Return ONLY JSON matching the schema.\n\n"
        "Your job:\n"
        "1) Produce revised_claims as the COMPLETE claim set that should remain "
        "after this round.\n"
        "2) Use dropped_claims for removed claim keys.\n"
        "3) If tool_results show supported=false for a claim, usually drop it "
        "unless evidence is strong.\n"
        "4) Keep high precision.\n"
        "5) confidence_delta adjusts global confidence in [-1,1].\n\n"
        f"ROUND: {round_idx}/{max_rounds}\n"
        f"CURRENT_STATE_JSON:\n{state_json}\n\n"
        f"CRITIQUE_JSON:\n{critique_json}\n\n"
        f"TOOL_RESULTS_JSON:\n{tool_results_json}\n\n"
        f"ARTICLE:\n{article_text.strip()}"
    )


def _build_judge_prompt(
    *,
    state: DebateState,
    critique: Critique,
    round_idx: int,
    max_rounds: int,
) -> str:
    state_json = json.dumps(state.model_dump(mode="json"), ensure_ascii=False)
    critique_json = json.dumps(critique.model_dump(mode="json"), ensure_ascii=False)
    return (
        "Role: Adjudicator.\n"
        "Return ONLY JSON matching the schema.\n\n"
        "Evaluate whether extraction has converged.\n"
        "Set should_continue=false when remaining issues are minor or no "
        "high-confidence gains expected.\n"
        "Use unresolved_issues for concrete blockers.\n\n"
        f"ROUND: {round_idx}/{max_rounds}\n"
        f"CURRENT_STATE_JSON:\n{state_json}\n\n"
        f"LATEST_CRITIQUE_JSON:\n{critique_json}"
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


class Strategy9Harness(IncrementalTaskHarness):
    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        tenant: TenantContext,
        article_text: str,
        proposer_model: str,
        critic_model: str,
        repair_model: str,
        judge_model: str,
        max_rounds: int,
        replay_policy: ReplayPolicy = "allow_prompt_drift",
    ) -> None:
        super().__init__(
            kernel=kernel,
            tenant=tenant,
            default_model=proposer_model,
            replay_policy=replay_policy,
        )
        self.article_text = article_text
        self.proposer_model = proposer_model
        self.critic_model = critic_model
        self.repair_model = repair_model
        self.judge_model = judge_model
        self.max_rounds = max_rounds

    async def define_tasks(self) -> list[TaskUnit]:
        return [
            TaskUnit(id=f"round_{idx}", description=f"Debate round {idx}")
            for idx in range(1, self.max_rounds + 1)
        ]

    async def on_initialize(self, *, context: HarnessContext) -> None:
        await super().on_initialize(context=context)
        await self.set_artifact(
            key="debate_state",
            value=DebateState().model_dump(mode="json"),
            step_key="init_debate_state",
        )
        await self.set_artifact(
            key="debate_halted",
            value=False,
            step_key="init_debate_halted",
        )
        await self.write_summary(
            "debate_state_snapshot",
            {"round": 0, "claim_count": 0, "halted": False},
            step_key="debate_state_snapshot_init",
        )

    async def on_wake(
        self,
        *,
        context: HarnessContext,
        reorientation: dict[str, object],
    ) -> None:
        await super().on_wake(context=context, reorientation=reorientation)
        progress = await self.get_task_progress(run_id=context.run_id)
        completed_rounds = (
            sum(1 for unit in progress if unit.state == "done")
            if progress is not None
            else 0
        )
        state_data = await self.get_artifact(key="debate_state")
        state = (
            DebateState.model_validate(state_data)
            if isinstance(state_data, dict)
            else DebateState()
        )
        await self.write_summary(
            "debate_wake",
            {
                "run_id": context.run_id,
                "claims": len(state.claims),
                "open_questions": len(state.open_questions),
            },
            step_key=f"wake_{completed_rounds + 1}",
        )

    async def on_sleep(
        self,
        *,
        context: HarnessContext,
        execution_error: Exception | None,
    ) -> None:
        state_data = await self.get_artifact(key="debate_state")
        state = (
            DebateState.model_validate(state_data)
            if isinstance(state_data, dict)
            else DebateState()
        )
        progress = await self.get_task_progress(run_id=context.run_id)
        completed_rounds = (
            sum(1 for unit in progress if unit.state == "done")
            if progress is not None
            else 0
        )
        await self.write_summary(
            "debate_state_snapshot",
            {
                "claims": len(state.claims),
                "evidence": len(state.evidence),
                "open_questions": len(state.open_questions),
                "confidence_score": state.confidence_score,
                "execution_error": None if execution_error is None else str(execution_error),
            },
            step_key=f"sleep_{completed_rounds}",
        )
        await super().on_sleep(context=context, execution_error=execution_error)

    async def work_on(self, task: TaskUnit, context: HarnessContext) -> None:
        halted = await self.get_artifact(key="debate_halted")
        if bool(halted):
            await self.write_summary(
                "debate_round",
                {
                    "task_id": task.id,
                    "status": "skipped",
                    "reason": "halted",
                },
                step_key=f"{task.id}.summary",
            )
            return

        state_data = await self.get_artifact(key="debate_state")
        state = (
            DebateState.model_validate(state_data)
            if isinstance(state_data, dict)
            else DebateState()
        )
        round_idx = int(task.id.split("_")[1])

        proposal = await self.run_model(
            model=self.proposer_model,
            output_schema=Proposal,
            prompt=_build_proposer_prompt(
                article_text=self.article_text,
                state=state,
                round_idx=round_idx,
                max_rounds=self.max_rounds,
            ),
            step_key=f"{task.id}.proposer",
        )
        _merge_claims(state, proposal.output.claims)
        state.evidence = _dedupe_texts([*state.evidence, *proposal.output.supporting_evidence])
        state.confidence_score = _clamp01(
            (state.confidence_score + _clamp01(proposal.output.confidence)) / 2.0
        )

        critique = await self.run_model(
            model=self.critic_model,
            output_schema=Critique,
            prompt=_build_critic_prompt(
                article_text=self.article_text,
                state=state,
                proposal=proposal.output,
                round_idx=round_idx,
                max_rounds=self.max_rounds,
            ),
            step_key=f"{task.id}.critic",
        )
        state.counterclaims = _dedupe_texts(
            [
                *state.counterclaims,
                *critique.output.invalid_claims,
                *critique.output.alternative_explanations,
            ]
        )
        state.open_questions = _dedupe_texts(
            [*state.open_questions, *critique.output.missing_assumptions]
        )

        claim_by_key = _claims_to_map(state.claims)
        tool_results: list[dict[str, object]] = []
        for index, claim_key in enumerate(
            _dedupe_texts(critique.output.requires_tool_check),
            start=1,
        ):
            candidate = claim_by_key.get(claim_key)
            if candidate is None:
                state.open_questions = _dedupe_texts(
                    [*state.open_questions, f"Unknown claim_key for tool check: {claim_key}"]
                )
                continue
            tool_step = await self.run_tool(
                tool_name="verify_claim_tool",
                arguments=VerifyClaimArgs(
                    article_text=self.article_text,
                    claim_key=candidate.key,
                    src=candidate.src,
                    relation=candidate.relation,
                    dst=candidate.dst,
                    evidence=candidate.evidence,
                ),
                step_key=f"{task.id}.tool_{index}_{_safe_step_fragment(claim_key)}",
            )
            parsed = _parse_tool_result_json(tool_step.result_json)
            tool_results.append(parsed)
            state.evidence = _dedupe_texts(
                [*state.evidence, json.dumps(parsed, ensure_ascii=False)]
            )

        repair = await self.run_model(
            model=self.repair_model,
            output_schema=Repair,
            prompt=_build_repair_prompt(
                article_text=self.article_text,
                state=state,
                critique=critique.output,
                tool_results=tool_results,
                round_idx=round_idx,
                max_rounds=self.max_rounds,
            ),
            step_key=f"{task.id}.repair",
        )
        if repair.output.revised_claims:
            _replace_claims(state, repair.output.revised_claims)
        _drop_claims(state, repair.output.dropped_claims)
        state.confidence_score = _clamp01(
            state.confidence_score + repair.output.confidence_delta
        )

        evaluation = await self.run_model(
            model=self.judge_model,
            output_schema=Evaluation,
            prompt=_build_judge_prompt(
                state=state,
                critique=critique.output,
                round_idx=round_idx,
                max_rounds=self.max_rounds,
            ),
            step_key=f"{task.id}.judge",
        )
        state.open_questions = _dedupe_texts(
            [*state.open_questions, *evaluation.output.unresolved_issues]
        )
        should_continue = bool(evaluation.output.should_continue)

        await self.set_artifact(
            key="debate_state",
            value=state.model_dump(mode="json"),
            step_key=f"{task.id}.persist_state",
        )
        await self.set_artifact(
            key="debate_halted",
            value=not should_continue,
            step_key=f"{task.id}.persist_halt",
        )
        await self.write_summary(
            "debate_round",
            {
                "task_id": task.id,
                "claims": len(state.claims),
                "tool_checks": len(tool_results),
                "should_continue": should_continue,
                "confidence_score": state.confidence_score,
            },
            step_key=f"{task.id}.summary",
        )


async def _predict_article_with_harness(
    *,
    kernel: ArtanaKernel,
    article_path: Path,
    proposer_model: str,
    critic_model: str,
    repair_model: str,
    judge_model: str,
    tenant: TenantContext,
    max_rounds: int,
    replay_policy: ReplayPolicy,
    evidence_required: bool,
    min_evidence_chars: int,
    min_confidence: float,
) -> dict[str, object]:
    article_text = article_path.read_text(encoding="utf-8")
    run_id = _strategy9_run_id(
        article_path=article_path,
        proposer_model=proposer_model,
        critic_model=critic_model,
        repair_model=repair_model,
        judge_model=judge_model,
        max_rounds=max_rounds,
        replay_policy=replay_policy,
    )
    harness = Strategy9Harness(
        kernel=kernel,
        tenant=tenant,
        article_text=article_text,
        proposer_model=proposer_model,
        critic_model=critic_model,
        repair_model=repair_model,
        judge_model=judge_model,
        max_rounds=max_rounds,
        replay_policy=replay_policy,
    )

    final_progress: tuple[TaskUnit, ...] = ()
    for _ in range(max_rounds):
        final_progress = await harness.run(run_id=run_id)
        halted = await harness.get_artifact(run_id=run_id, key="debate_halted")
        if bool(halted):
            break
        if final_progress and all(unit.state == "done" for unit in final_progress):
            break

    state_data = await harness.get_artifact(run_id=run_id, key="debate_state")
    final_state = (
        DebateState.model_validate(state_data)
        if isinstance(state_data, dict)
        else DebateState()
    )
    reviewed_result = _claims_to_reviewed_result(
        article_name=article_path.stem,
        claims=final_state.claims,
    )
    return _assemble_payload(
        article_name=article_path.stem,
        extracted=reviewed_result,
        evidence_required=evidence_required,
        min_evidence_chars=min_evidence_chars,
        min_confidence=min_confidence,
    )


async def _run(
    *,
    article_paths: Sequence[Path],
    proposer_model: str,
    critic_model: str,
    repair_model: str,
    judge_model: str,
    tenant: TenantContext,
    max_rounds: int,
    state_db: Path,
    samples_dir: Path,
    explicit_gold: Path | None,
    output_dir: Path | None,
    max_diff_preview: int,
    timeout_seconds: float,
    reasoning_effort: str | None,
    verbosity: str | None,
    replay_policy: ReplayPolicy,
    score_mode: str,
    evidence_required: bool,
    min_evidence_chars: int,
    min_confidence: float,
) -> tuple[EvalResult, ...]:
    results: list[EvalResult] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    state_db.parent.mkdir(parents=True, exist_ok=True)

    kernel = _build_kernel(
        db_path=state_db,
        timeout_seconds=timeout_seconds,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )

    @kernel.tool()
    async def verify_claim_tool(
        article_text: str,
        claim_key: str,
        src: str,
        relation: str,
        dst: str,
        evidence: str = "",
    ) -> str:
        claim = _normalize_claim(
            ClaimRecord(
                key=claim_key,
                bucket="fact_edges",
                src=src,
                relation=relation,
                dst=dst,
                evidence=evidence,
            )
        )
        verdict = _verify_claim_against_article(article_text=article_text, claim=claim)
        return json.dumps(verdict, sort_keys=True, ensure_ascii=False)

    try:
        for article_path in article_paths:
            predicted = await _predict_article_with_harness(
                kernel=kernel,
                article_path=article_path,
                proposer_model=proposer_model,
                critic_model=critic_model,
                repair_model=repair_model,
                judge_model=judge_model,
                tenant=tenant,
                max_rounds=max_rounds,
                replay_policy=replay_policy,
                evidence_required=evidence_required,
                min_evidence_chars=min_evidence_chars,
                min_confidence=min_confidence,
            )
            if output_dir is not None:
                output_path = output_dir / f"{article_path.stem}.strategy9.json"
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
    finally:
        await kernel.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strategy9: modern IncrementalTaskHarness debate loop with artifacts, "
            "run summaries, replay policy, and deterministic tool verification."
        )
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
        help=f"Durable SQLite DB path for harness runs. Default: {DEFAULT_STATE_DB}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional fallback model id for all loop roles.",
    )
    parser.add_argument(
        "--proposer-model",
        type=str,
        default=os.getenv("ARTANA_PROPOSER_MODEL"),
        help=(
            "Model id for proposer role (default: ARTANA_PROPOSER_MODEL, "
            "else ARTANA_MODEL, else gpt-5.2)."
        ),
    )
    parser.add_argument(
        "--critic-model",
        type=str,
        default=os.getenv("ARTANA_CRITIC_MODEL"),
        help=(
            "Model id for critic role (default: ARTANA_CRITIC_MODEL, "
            "else ARTANA_MODEL, else gpt-5.2)."
        ),
    )
    parser.add_argument(
        "--repair-model",
        type=str,
        default=os.getenv("ARTANA_REPAIR_MODEL"),
        help=(
            "Model id for repair role (default: ARTANA_REPAIR_MODEL, "
            "else ARTANA_MODEL, else gpt-5.2)."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.getenv("ARTANA_JUDGE_MODEL"),
        help=(
            "Model id for judge role (default: ARTANA_JUDGE_MODEL, "
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
        "--max-rounds",
        type=int,
        default=2,
        help="Maximum debate rounds for proposer/critic/repair/judge loop.",
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
        "--replay-policy",
        choices=("strict", "allow_prompt_drift", "fork_on_drift"),
        default=cast(
            str,
            os.getenv("ARTANA_REPLAY_POLICY", "allow_prompt_drift"),
        ),
        help=(
            "Replay behavior for model steps. "
            "Default: ARTANA_REPLAY_POLICY or allow_prompt_drift."
        ),
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
            "OPENAI_API_KEY is required for strategy9 execution. "
            "Load environment first (e.g. `set -a; source .env; set +a`)."
        )
    if args.gold is not None and args.article is None:
        raise ValueError("--gold requires --article.")
    if args.budget_usd <= 0:
        raise ValueError("--budget-usd must be > 0.")
    if args.max_rounds <= 0:
        raise ValueError("--max-rounds must be >= 1.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0.")
    if args.state_db.suffix != ".db":
        raise ValueError("--state-db must be a .db SQLite file path.")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1.")
    if args.min_evidence_chars < 0:
        raise ValueError("--min-evidence-chars must be >= 0.")

    fallback_model = args.model or os.getenv("ARTANA_MODEL")
    proposer_model = args.proposer_model or fallback_model or "gpt-5.2"
    critic_model = args.critic_model or fallback_model or "gpt-5.2"
    repair_model = args.repair_model or fallback_model or "gpt-5.2"
    judge_model = args.judge_model or fallback_model or "gpt-5.2"
    print(
        "Using models: "
        f"proposer={proposer_model} "
        f"critic={critic_model} "
        f"repair={repair_model} "
        f"judge={judge_model} "
        f"max_rounds={args.max_rounds} "
        f"replay_policy={args.replay_policy}"
    )

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
        proposer_model=proposer_model,
        critic_model=critic_model,
        repair_model=repair_model,
        judge_model=judge_model,
        tenant=tenant,
        max_rounds=args.max_rounds,
        state_db=args.state_db.resolve(),
        samples_dir=samples_dir,
        explicit_gold=args.gold.resolve() if args.gold is not None else None,
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        max_diff_preview=max(1, args.max_diff_preview),
        timeout_seconds=args.timeout_seconds,
        reasoning_effort=cast(str, args.reasoning_effort),
        verbosity=cast(str, args.verbosity),
        replay_policy=cast(ReplayPolicy, args.replay_policy),
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
