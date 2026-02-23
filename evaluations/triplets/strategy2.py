from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import tempfile
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

from artana.agent import AutonomousAgent, ContextBuilder
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


class DerivedResult(BaseModel):
    derived_edges: list[Triplet] = Field(default_factory=list)


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


def _normalize_relation(value: str) -> str:
    base = _normalize_text(value).upper()
    base = base.replace("-", "_").replace("/", "_").replace(" ", "_")
    base = re.sub(r"_+", "_", base)
    return RELATION_SYNONYMS.get(base, base)


def _normalize_entity_name(value: str) -> str:
    return _normalize_text(value)


def _entity_alias_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).upper()


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


def _build_extraction_prompt(article_text: str) -> str:
    return (
        "Extract an explicit knowledge graph from this article.\n\n"
        "You must return JSON matching the schema exactly.\n\n"
        "Rules:\n"
        "1) fact_edges = explicit supported statements only.\n"
        "2) contradicted_edges = explicit refuted/retracted/disproven claims.\n"
        "3) hypothesis_only = speculative/unproven claims only.\n"
        "4) Never put contradicted or speculative claims in fact_edges.\n"
        "5) Build entities list with canonical_name + aliases.\n"
        "6) Use only fact relation labels:\n"
        "   ACTIVATES, INHIBITS, PART_OF, UPREGULATES, DOWNREGULATES, "
        "ASSOCIATED_WITH, REGULATES.\n"
        "7) For hypothesis_only use POSSIBLE_ASSOCIATION when possible.\n"
        "8) Do not derive multi-hop edges in this step.\n"
        "9) Prefer conservative extraction over over-claiming.\n\n"
        f"ARTICLE:\n{article_text.strip()}"
    )


def _build_derived_adjudication_prompt(
    *,
    article_text: str,
    explicit_payload: dict[str, object],
    candidate_edges: list[list[str]],
) -> str:
    return (
        "You are deriving conservative multi-hop edges from explicit facts.\n\n"
        "Rules:\n"
        "1) Use only explicit fact_edges as proof.\n"
        "2) Ignore contradicted_edges and hypothesis_only as proof sources.\n"
        "3) Keep only high-confidence derived edges.\n"
        "4) Prefer biologically meaningful derived edges (program-level or disease-level),\n"
        "   not trivial intermediate-protein restatements.\n"
        "5) Use only relation labels:\n"
        "   DERIVED_REGULATES, DERIVED_ASSOCIATED_WITH, DERIVED_DOWNREGULATES,\n"
        "   DERIVED_REDUCES, DERIVED_INHIBITS.\n"
        "6) Do not output duplicates.\n\n"
        f"ARTICLE:\n{article_text.strip()}\n\n"
        f"EXPLICIT_GRAPH_JSON:\n{json.dumps(explicit_payload, ensure_ascii=False)}\n\n"
        f"DETERMINISTIC_CANDIDATES_JSON:\n{json.dumps(candidate_edges, ensure_ascii=False)}"
    )


def _normalize_entities(records: Sequence[EntityRecord]) -> dict[str, dict[str, object]]:
    entities: dict[str, dict[str, object]] = {}
    for record in records:
        canonical = _normalize_entity_name(record.canonical_name)
        if not canonical:
            continue
        entity_type = record.type if record.type in ENTITY_TYPES else "OTHER"
        aliases: list[str] = []
        seen_aliases: set[str] = set()
        for alias in record.aliases:
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


def _build_alias_to_canonical(entities: dict[str, dict[str, object]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, entry in entities.items():
        mapping[_entity_alias_key(canonical)] = canonical
        aliases_obj = entry.get("aliases", [])
        if not isinstance(aliases_obj, list):
            continue
        for alias in aliases_obj:
            if isinstance(alias, str):
                mapping[_entity_alias_key(alias)] = canonical
    return mapping


def _normalize_triplets(
    triplets: Sequence[Triplet],
    *,
    alias_to_canonical: dict[str, str],
    allowed_relations: frozenset[str] | None = None,
    default_relation: str | None = None,
    coerce_to_derived_prefix: bool = False,
) -> list[list[str]]:
    seen: set[tuple[str, str, str]] = set()
    normalized: list[list[str]] = []

    for edge in triplets:
        src = _normalize_entity_name(edge.src)
        dst = _normalize_entity_name(edge.dst)
        if not src or not dst:
            continue

        src = alias_to_canonical.get(_entity_alias_key(src), src)
        dst = alias_to_canonical.get(_entity_alias_key(dst), dst)
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


def _is_disease(entity_name: str, entities: dict[str, dict[str, object]]) -> bool:
    entry = entities.get(entity_name)
    if entry is not None and entry.get("type") == "DISEASE":
        return True
    upper = entity_name.upper()
    return "DISEASE" in upper or "DISORDER" in upper or "SYNDROME" in upper


def _is_process_like(entity_name: str, entities: dict[str, dict[str, object]]) -> bool:
    entry = entities.get(entity_name)
    if entry is not None and entry.get("type") == "PROCESS":
        return True
    upper = entity_name.upper()
    keywords = ("PROGRAM", "RESPONSE", "CASCADE", "MODULE", "SIGNATURE")
    return any(keyword in upper for keyword in keywords)


def _generate_deterministic_derived_candidates(
    *,
    fact_edges: list[list[str]],
    entities: dict[str, dict[str, object]],
    max_depth: int = 6,
) -> list[list[str]]:
    signed_adjacency: dict[str, list[tuple[str, int]]] = {}
    outgoing_by_source: dict[str, set[str]] = {}
    association_targets: dict[str, set[str]] = {}

    for src, relation, dst in fact_edges:
        outgoing_by_source.setdefault(src, set()).add(relation)
        if relation in SIGN_BY_RELATION:
            signed_adjacency.setdefault(src, []).append((dst, SIGN_BY_RELATION[relation]))
        if relation == "ASSOCIATED_WITH":
            association_targets.setdefault(src, set()).add(dst)

    source_nodes = {
        source
        for source, rels in outgoing_by_source.items()
        if any(rel in {"ACTIVATES", "INHIBITS", "UPREGULATES", "DOWNREGULATES"} for rel in rels)
    }

    candidates: set[tuple[str, str, str]] = set()
    for source in source_nodes:
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
                    candidates.add((source, "DERIVED_REGULATES", target))
                else:
                    candidates.add((source, "DERIVED_DOWNREGULATES", target))
                    candidates.add((source, "DERIVED_INHIBITS", target))

            if target in association_targets:
                for disease in association_targets[target]:
                    if depth + 1 < 2:
                        continue
                    if sign > 0:
                        candidates.add((source, "DERIVED_ASSOCIATED_WITH", disease))
                    else:
                        candidates.add((source, "DERIVED_REDUCES", disease))

    return [list(item) for item in sorted(candidates)]


def _assemble_payload(
    *,
    article_name: str,
    extracted: ExtractionResult,
    derived: DerivedResult,
) -> dict[str, object]:
    benchmark_name = _normalize_text(extracted.benchmark_name) or article_name
    entities = _normalize_entities(extracted.entities)
    alias_to_canonical = _build_alias_to_canonical(entities)

    fact_edges = _normalize_triplets(
        extracted.fact_edges,
        alias_to_canonical=alias_to_canonical,
        allowed_relations=FACT_RELATIONS,
    )
    contradicted_edges = _normalize_triplets(
        extracted.contradicted_edges,
        alias_to_canonical=alias_to_canonical,
        allowed_relations=FACT_RELATIONS,
    )
    hypothesis_edges = _normalize_triplets(
        extracted.hypothesis_only,
        alias_to_canonical=alias_to_canonical,
        allowed_relations=frozenset({HYPOTHESIS_RELATION}),
        default_relation=HYPOTHESIS_RELATION,
    )
    derived_edges = _normalize_triplets(
        derived.derived_edges,
        alias_to_canonical=alias_to_canonical,
        allowed_relations=DERIVED_RELATIONS,
        coerce_to_derived_prefix=True,
    )

    payload: dict[str, object] = {
        "benchmark_name": benchmark_name,
        "fact_edges": fact_edges,
        "derived_edges": derived_edges,
    }
    if entities:
        payload["entities"] = entities
    if contradicted_edges:
        payload["contradicted_edges"] = contradicted_edges
    if hypothesis_edges:
        payload["hypothesis_only"] = hypothesis_edges
    return payload


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
    per_key: dict[str, KeyMetrics] = {}
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for key in EDGE_KEYS:
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


async def _predict_article_with_autonomous_agent(
    *,
    article_path: Path,
    model: str,
    tenant: TenantContext,
    agent_iterations: int,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    verbosity: Literal["low", "medium", "high"] | None,
    timeout_seconds: float,
) -> dict[str, object]:
    article_text = article_path.read_text(encoding="utf-8")
    tmp_root = Path(tempfile.mkdtemp(prefix="artana_triplets_strategy2_"))
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
    context_builder = ContextBuilder(
        identity="You are a strict biomedical knowledge graph extraction agent.",
        progressive_skills=False,
    )
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=context_builder,
    )

    try:
        extraction = await agent.run(
            run_id=f"triplets_strategy2::{article_path.stem}::extract",
            tenant=tenant,
            model=model,
            system_prompt=(
                "Return only structured JSON matching the output schema. "
                "Do not call tools unless absolutely required."
            ),
            prompt=_build_extraction_prompt(article_text),
            output_schema=ExtractionResult,
            max_iterations=agent_iterations,
        )

        entities = _normalize_entities(extraction.entities)
        alias_to_canonical = _build_alias_to_canonical(entities)
        normalized_facts = _normalize_triplets(
            extraction.fact_edges,
            alias_to_canonical=alias_to_canonical,
            allowed_relations=FACT_RELATIONS,
        )
        normalized_contradictions = _normalize_triplets(
            extraction.contradicted_edges,
            alias_to_canonical=alias_to_canonical,
            allowed_relations=FACT_RELATIONS,
        )
        normalized_hypotheses = _normalize_triplets(
            extraction.hypothesis_only,
            alias_to_canonical=alias_to_canonical,
            allowed_relations=frozenset({HYPOTHESIS_RELATION}),
            default_relation=HYPOTHESIS_RELATION,
        )
        deterministic_candidates = _generate_deterministic_derived_candidates(
            fact_edges=normalized_facts,
            entities=entities,
        )
        explicit_payload = {
            "entities": entities,
            "fact_edges": normalized_facts,
            "contradicted_edges": normalized_contradictions,
            "hypothesis_only": normalized_hypotheses,
        }
        derived = await agent.run(
            run_id=f"triplets_strategy2::{article_path.stem}::derive",
            tenant=tenant,
            model=model,
            system_prompt=(
                "Return only structured JSON matching the output schema. "
                "Do not call tools unless absolutely required."
            ),
            prompt=_build_derived_adjudication_prompt(
                article_text=article_text,
                explicit_payload=explicit_payload,
                candidate_edges=deterministic_candidates,
            ),
            output_schema=DerivedResult,
            max_iterations=agent_iterations,
        )

        return _assemble_payload(
            article_name=article_path.stem,
            extracted=extraction,
            derived=derived,
        )
    finally:
        await kernel.close()
        shutil.rmtree(tmp_root, ignore_errors=True)


async def _run(
    *,
    article_paths: Sequence[Path],
    model: str,
    tenant: TenantContext,
    samples_dir: Path,
    explicit_gold: Path | None,
    output_dir: Path | None,
    max_diff_preview: int,
    agent_iterations: int,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    verbosity: Literal["low", "medium", "high"] | None,
    timeout_seconds: float,
) -> tuple[EvalResult, ...]:
    results: list[EvalResult] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for article_path in article_paths:
        predicted = await _predict_article_with_autonomous_agent(
            article_path=article_path,
            model=model,
            tenant=tenant,
            agent_iterations=agent_iterations,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            timeout_seconds=timeout_seconds,
        )
        if output_dir is not None:
            output_path = output_dir / f"{article_path.stem}.strategy2.json"
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
        )
        _print_eval(result, max_diff_preview=max_diff_preview)
        results.append(result)
    return tuple(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strategy2: AutonomousAgent-based triplet extraction and evaluation."
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
        default=os.getenv("ARTANA_MODEL", "gpt-4o-mini"),
        help="Model id for LiteLLMAdapter (default: ARTANA_MODEL or gpt-4o-mini).",
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
        "--agent-iterations",
        type=int,
        default=8,
        help="Max iterations for each AutonomousAgent run.",
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
            "OPENAI_API_KEY is required for AutonomousAgent strategy execution. "
            "Load environment first (e.g. `set -a; source .env; set +a`)."
        )
    if args.gold is not None and args.article is None:
        raise ValueError("--gold requires --article.")
    if args.budget_usd <= 0:
        raise ValueError("--budget-usd must be > 0.")
    if args.agent_iterations <= 0:
        raise ValueError("--agent-iterations must be >= 1.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0.")

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
        model=args.model,
        tenant=tenant,
        samples_dir=samples_dir,
        explicit_gold=args.gold.resolve() if args.gold is not None else None,
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        max_diff_preview=max(1, args.max_diff_preview),
        agent_iterations=args.agent_iterations,
        reasoning_effort=cast(
            Literal["low", "medium", "high"],
            args.reasoning_effort,
        ),
        verbosity=cast(
            Literal["low", "medium", "high"],
            args.verbosity,
        ),
        timeout_seconds=args.timeout_seconds,
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
