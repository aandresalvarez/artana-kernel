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
from typing import Literal, cast

from pydantic import BaseModel, Field

from artana.agent import KernelModelClient
from artana.kernel import (
    ArtanaKernel,
    KernelPolicy,
    WorkflowContext,
    json_step_serde,
    pydantic_step_serde,
)
from artana.models import TenantContext
from artana.ports.model import LiteLLMAdapter
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


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_relation(value: str) -> str:
    base = _normalize_text(value).upper()
    base = base.replace("-", "_").replace("/", "_").replace(" ", "_")
    base = re.sub(r"_+", "_", base)
    return RELATION_SYNONYMS.get(base, base)


def _normalize_entity_name(value: str) -> str:
    return _normalize_text(value)


def _normalize_triplets(
    triplets: Sequence[Triplet],
    *,
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
            alias_value = _normalize_entity_name(alias)
            if not alias_value or alias_value == canonical:
                continue
            if alias_value in seen_aliases:
                continue
            seen_aliases.add(alias_value)
            aliases.append(alias_value)
        entry: dict[str, object] = {"type": entity_type, "aliases": aliases}
        if record.note:
            entry["note"] = _normalize_text(record.note)
        entities[canonical] = entry
    return entities


def _build_extraction_prompt(article_text: str) -> str:
    return (
        "Extract a biomedical/technical knowledge graph from the article.\n\n"
        "Output MUST follow the provided JSON schema exactly.\n\n"
        "Rules:\n"
        "1) fact_edges: include only explicit supported statements from the article.\n"
        "2) contradicted_edges: include explicit refuted/retracted/disproven claims.\n"
        "3) hypothesis_only: include speculative/unproven claims only.\n"
        "4) Do not place speculative or contradicted claims into fact_edges.\n"
        "5) Canonicalize aliases to one canonical entity name per concept.\n"
        "6) Use relation labels from this ontology for facts:\n"
        "   ACTIVATES, INHIBITS, PART_OF, UPREGULATES, DOWNREGULATES, "
        "ASSOCIATED_WITH, REGULATES.\n"
        "7) For hypothesis_only, relation should be POSSIBLE_ASSOCIATION when possible.\n"
        "8) Do not infer multi-hop derived relations in this step.\n"
        "9) Be conservative: if uncertain, exclude from fact_edges and prefer hypothesis_only.\n\n"
        f"ARTICLE:\n{article_text.strip()}"
    )


def _build_derived_prompt(extracted: ExtractionResult) -> str:
    explicit_payload = {
        "fact_edges": [edge.model_dump() for edge in extracted.fact_edges],
        "contradicted_edges": [edge.model_dump() for edge in extracted.contradicted_edges],
        "hypothesis_only": [edge.model_dump() for edge in extracted.hypothesis_only],
    }
    return (
        "Derive multi-hop edges from explicit fact_edges only.\n\n"
        "Constraints:\n"
        "1) Use only explicit fact_edges as evidence.\n"
        "2) Never use contradicted_edges or hypothesis_only as proof.\n"
        "3) Derived edges should require at least a 2-hop path.\n"
        "4) Use only these relation labels:\n"
        "   DERIVED_REGULATES, DERIVED_ASSOCIATED_WITH, DERIVED_DOWNREGULATES, "
        "DERIVED_REDUCES, DERIVED_INHIBITS.\n"
        "5) Prefer conservative outputs; if support is weak, omit the edge.\n\n"
        f"INPUT_FACT_GRAPH_JSON:\n{json.dumps(explicit_payload, ensure_ascii=False)}"
    )


def _assemble_payload(
    *,
    article_name: str,
    extracted: ExtractionResult,
    derived: DerivedResult,
) -> dict[str, object]:
    benchmark_name = _normalize_text(extracted.benchmark_name) or article_name
    entities = _normalize_entities(extracted.entities)
    fact_edges = _normalize_triplets(extracted.fact_edges, allowed_relations=FACT_RELATIONS)
    contradicted_edges = _normalize_triplets(
        extracted.contradicted_edges,
        allowed_relations=FACT_RELATIONS,
    )
    hypothesis_edges = _normalize_triplets(
        extracted.hypothesis_only,
        allowed_relations=frozenset({HYPOTHESIS_RELATION}),
        default_relation=HYPOTHESIS_RELATION,
    )
    derived_edges = _normalize_triplets(
        derived.derived_edges,
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
        if not isinstance(item, list):
            continue
        if len(item) != 3:
            continue
        src, relation, dst = item
        if not all(isinstance(part, str) for part in (src, relation, dst)):
            continue
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
            preview = list(metrics.missing[:max_diff_preview])
            print(f"    missing: {preview}")
        if metrics.extra:
            preview = list(metrics.extra[:max_diff_preview])
            print(f"    extra: {preview}")


async def _predict_article_with_artana(
    *,
    article_path: Path,
    model: str,
    tenant: TenantContext,
) -> dict[str, object]:
    article_text = article_path.read_text(encoding="utf-8")
    tmp_root = Path(tempfile.mkdtemp(prefix="artana_triplets_strategy1_"))
    db_path = tmp_root / "state.db"
    kernel = ArtanaKernel(
        store=SQLiteStore(str(db_path)),
        model_port=LiteLLMAdapter(
            timeout_seconds=45.0,
            max_retries=2,
            fail_on_unknown_cost=False,
        ),
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )
    chat = KernelModelClient(kernel=kernel)

    async def workflow(context: WorkflowContext) -> dict[str, object]:
        async def extract_action() -> ExtractionResult:
            prompt = _build_extraction_prompt(article_text)
            result = await chat.step(
                run_id=context.run_id,
                tenant=tenant,
                model=model,
                prompt=prompt,
                output_schema=ExtractionResult,
                step_key="extract_explicit_graph",
            )
            return result.output

        extracted = await context.step(
            name="extract_explicit_graph",
            action=extract_action,
            serde=pydantic_step_serde(ExtractionResult),
        )

        async def derive_action() -> DerivedResult:
            prompt = _build_derived_prompt(extracted)
            result = await chat.step(
                run_id=context.run_id,
                tenant=tenant,
                model=model,
                prompt=prompt,
                output_schema=DerivedResult,
                step_key="derive_multi_hop_graph",
            )
            return result.output

        derived = await context.step(
            name="derive_multi_hop_graph",
            action=derive_action,
            serde=pydantic_step_serde(DerivedResult),
        )

        async def assemble_action() -> dict[str, object]:
            return _assemble_payload(
                article_name=article_path.stem,
                extracted=extracted,
                derived=derived,
            )

        assembled = await context.step(
            name="assemble_output",
            action=assemble_action,
            serde=json_step_serde(),
        )
        if not isinstance(assembled, dict):
            raise TypeError("assemble_output returned non-dict output.")
        return cast(dict[str, object], assembled)

    try:
        run_result = await kernel.run_workflow(
            run_id=f"triplets_strategy1::{article_path.stem}",
            tenant=tenant,
            workflow=workflow,
        )
        if run_result.status != "complete" or run_result.output is None:
            raise RuntimeError("Workflow did not complete successfully.")
        return cast(dict[str, object], run_result.output)
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
) -> tuple[EvalResult, ...]:
    results: list[EvalResult] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for article_path in article_paths:
        predicted = await _predict_article_with_artana(
            article_path=article_path,
            model=model,
            tenant=tenant,
        )
        if output_dir is not None:
            output_path = output_dir / f"{article_path.stem}.strategy1.json"
            output_path.write_text(
                json.dumps(predicted, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        gold_path = _resolve_gold_path(
            article_path=article_path,
            samples_dir=samples_dir,
            explicit_gold=explicit_gold if len(article_paths) == 1 else None,
        )
        eval_result = _compare_with_gold(
            article_path=article_path,
            predicted=predicted,
            gold_path=gold_path,
        )
        _print_eval(eval_result, max_diff_preview=max_diff_preview)
        results.append(eval_result)
    return tuple(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strategy1: AI-first triplet extraction/derivation using Artana workflow runtime."
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
        "--model",
        type=str,
        default=os.getenv("ARTANA_MODEL", "gpt-4o-mini"),
        help="Model identifier for LiteLLMAdapter (default: ARTANA_MODEL or gpt-4o-mini).",
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
            "OPENAI_API_KEY is required for AI-first extraction. "
            "Load your environment first (e.g. `set -a; source .env; set +a`)."
        )
    if args.gold is not None and args.article is None:
        raise ValueError("--gold requires --article.")
    if args.budget_usd <= 0:
        raise ValueError("--budget-usd must be > 0.")

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
