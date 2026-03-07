from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from artana.events import EventType, KernelEvent, RunSummaryPayload
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult
from artana.store import PostgresStore, SQLiteStore
from artana.store.base import EventStore, SupportsEventStreaming, SupportsRunIndexing

OutputT = TypeVar("OutputT", bound=BaseModel)


class _NoopModelPort:
    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        raise RuntimeError("Model calls are not available in CLI read commands.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    try:
        return asyncio.run(_run_command(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


async def _run_command(args: argparse.Namespace) -> int:
    command = getattr(args, "command", None)
    if command == "init":
        return _run_init(args)
    if command != "run":
        raise ValueError("Unsupported command.")

    subcommand = getattr(args, "run_command", None)
    if subcommand is None:
        raise ValueError("Missing run subcommand.")
    store = _open_store(db=args.db, dsn=args.dsn)
    try:
        if subcommand == "list":
            return await _run_list(args=args, store=store)
        if subcommand == "tail":
            return await _run_tail(args=args, store=store)
        if subcommand == "verify-ledger":
            return await _run_verify_ledger(args=args, store=store)
        if subcommand == "status":
            return await _run_status(args=args, store=store)
        if subcommand == "summaries":
            return await _run_summaries(args=args, store=store)
        if subcommand == "artifacts":
            return await _run_artifacts(args=args, store=store)
        raise ValueError(f"Unknown run subcommand: {subcommand!r}")
    finally:
        await store.close()


def _open_store(*, db: str | None, dsn: str | None) -> EventStore:
    if dsn is not None:
        if db not in (None, ".state.db"):
            raise ValueError("Provide either --db or --dsn, not both.")
        return PostgresStore(dsn)
    return SQLiteStore(db or ".state.db")


async def _run_list(args: argparse.Namespace, *, store: EventStore) -> int:
    tenant_id = _require_cli_tenant(args).tenant_id
    since_value = getattr(args, "since", None)
    json_output = bool(getattr(args, "json_output", False))
    since_dt: datetime | None = None
    if isinstance(since_value, str):
        since_dt = datetime.fromisoformat(since_value)
    if not isinstance(store, SupportsRunIndexing):
        raise RuntimeError("Configured store does not support run indexing.")
    run_ids = await store.list_run_ids(tenant_id=tenant_id, since=since_dt)
    if json_output:
        print(
            json.dumps(
                {"tenant": tenant_id, "since": since_value, "run_ids": list(run_ids)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    for run_id in run_ids:
        print(run_id)
    return 0


async def _run_tail(args: argparse.Namespace, *, store: EventStore) -> int:
    run_id = args.run_id
    follow = bool(getattr(args, "follow", False))
    since_seq = int(getattr(args, "since_seq", 0))
    json_output = bool(getattr(args, "json_output", False))
    tenant = _require_cli_tenant(args)
    if follow:
        if not isinstance(store, SupportsEventStreaming):
            raise RuntimeError("Configured store does not support streaming.")
        kernel = _kernel_for_reads(store=store)
        async for event in kernel.stream_events(
            run_id=run_id,
            tenant=tenant,
            since_seq=since_seq,
            follow=True,
            poll_interval_seconds=0.5,
        ):
            _print_event(event, json_output=json_output)
        return 0
    events = await store.get_events_for_run(run_id)
    _assert_cli_run_access(run_id=run_id, tenant=tenant, events=events)
    filtered = [event for event in events if event.seq > since_seq]
    if json_output:
        print(
            json.dumps(
                {"run_id": run_id, "events": [_serialize_event(event) for event in filtered]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    for event in events:
        if event.seq <= since_seq:
            continue
        _print_event(event, json_output=False)
    return 0


async def _run_verify_ledger(args: argparse.Namespace, *, store: EventStore) -> int:
    run_id = args.run_id
    json_output = bool(getattr(args, "json_output", False))
    tenant = _require_cli_tenant(args)
    events = await store.get_events_for_run(run_id)
    _assert_cli_run_access(run_id=run_id, tenant=tenant, events=events)
    valid = await store.verify_run_chain(run_id)
    if json_output:
        print(
            json.dumps(
                {"run_id": run_id, "valid": valid},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print("valid" if valid else "invalid")
    return 0 if valid else 1


async def _run_status(args: argparse.Namespace, *, store: EventStore) -> int:
    run_id = args.run_id
    json_output = bool(getattr(args, "json_output", False))
    kernel = _kernel_for_reads(store=store)
    status = await kernel.get_run_status(run_id=run_id, tenant=_require_cli_tenant(args))
    payload = {
        "run_id": status.run_id,
        "tenant_id": status.tenant_id,
        "status": status.status,
        "last_event_seq": status.last_event_seq,
        "last_event_type": status.last_event_type,
        "updated_at": status.updated_at.isoformat(),
        "blocked_on": status.blocked_on,
        "failure_reason": status.failure_reason,
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"{payload['run_id']}\t{payload['status']}\t"
            f"last_seq={payload['last_event_seq']}\t"
            f"updated_at={payload['updated_at']}"
        )
    return 0


async def _run_summaries(args: argparse.Namespace, *, store: EventStore) -> int:
    run_id = args.run_id
    summary_type = getattr(args, "summary_type", None)
    limit = int(getattr(args, "limit", 20))
    json_output = bool(getattr(args, "json_output", False))
    tenant = _require_cli_tenant(args)

    events = await store.get_events_for_run(run_id)
    _assert_cli_run_access(run_id=run_id, tenant=tenant, events=events)
    summaries: list[dict[str, object]] = []
    for event in events:
        if event.event_type != EventType.RUN_SUMMARY:
            continue
        payload = event.payload
        if not isinstance(payload, RunSummaryPayload):
            continue
        payload_summary_type = payload.summary_type
        if summary_type is not None and payload_summary_type != summary_type:
            continue
        try:
            parsed_summary: object = json.loads(payload.summary_json)
        except json.JSONDecodeError:
            parsed_summary = payload.summary_json
        summaries.append(
            {
                "seq": event.seq,
                "timestamp": event.timestamp.isoformat(),
                "summary_type": payload_summary_type,
                "step_key": payload.step_key,
                "parent_step_key": event.parent_step_key,
                "summary": parsed_summary,
            }
        )
    if limit > 0:
        summaries = summaries[-limit:]

    if json_output:
        print(
            json.dumps(
                {"run_id": run_id, "summaries": summaries},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    for summary in summaries:
        print(
            f"{summary['seq']}\t{summary['summary_type']}\t"
            f"{summary['step_key']}\t{json.dumps(summary['summary'], ensure_ascii=False)}"
        )
    return 0


async def _run_artifacts(args: argparse.Namespace, *, store: EventStore) -> int:
    run_id = args.run_id
    json_output = bool(getattr(args, "json_output", False))
    kernel = _kernel_for_reads(store=store)
    artifacts = await kernel.list_artifacts(run_id=run_id, tenant=_require_cli_tenant(args))
    if json_output:
        print(
            json.dumps(
                {"run_id": run_id, "artifacts": artifacts},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    for key, value in sorted(artifacts.items()):
        print(f"{key}\t{json.dumps(value, ensure_ascii=False, sort_keys=True)}")
    return 0


def _run_init(args: argparse.Namespace) -> int:
    target = Path(args.path).expanduser().resolve()
    force = bool(getattr(args, "force", False))
    profile = getattr(args, "profile", "enforced")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"Target path exists and is not a directory: {target}")
        if any(target.iterdir()) and not force:
            raise ValueError(
                f"Target directory is not empty: {target}. Use --force to continue."
            )
    else:
        target.mkdir(parents=True, exist_ok=True)

    app_path = target / "app.py"
    readme_path = target / "README.md"
    app_path.write_text(_render_init_app(profile=profile), encoding="utf-8")
    readme_path.write_text(_render_init_readme(profile=profile), encoding="utf-8")
    print(f"Initialized Artana project in {target}")
    return 0


def _render_init_app(*, profile: str) -> str:
    if profile == "dev":
        kernel_init = (
            '    kernel = ArtanaKernel(\n'
            '        store=SQLiteStore("state.db"),\n'
            '        model_port=MockModelPort(output={"message": "Hello from Artana!"}),\n'
            "    )\n"
        )
    else:
        kernel_init = (
            '    kernel = ArtanaKernel(\n'
            '        store=SQLiteStore("state.db"),\n'
            '        model_port=MockModelPort(output={"message": "Hello from Artana!"}),\n'
            "        middleware=ArtanaKernel.default_middleware_stack(),\n"
            "        policy=KernelPolicy.enforced(),\n"
            "    )\n"
        )

    return (
        "from __future__ import annotations\n\n"
        "import asyncio\n"
        "from pydantic import BaseModel\n\n"
        "from artana import (\n"
        "    ArtanaKernel,\n"
        "    KernelPolicy,\n"
        "    MockModelPort,\n"
        "    SingleStepModelClient,\n"
        "    SQLiteStore,\n"
        "    StepKey,\n"
        "    TenantContext,\n"
        ")\n\n\n"
        "class HelloResult(BaseModel):\n"
        "    message: str\n\n\n"
        "async def main() -> None:\n"
        f"{kernel_init}"
        "    tenant = TenantContext(\n"
        '        tenant_id="demo_user",\n'
        "        capabilities=frozenset(),\n"
        "        budget_usd_limit=1.0,\n"
        "    )\n"
        "    client = SingleStepModelClient(kernel)\n"
        '    step = StepKey(namespace="hello")\n'
        "    result = await client.step(\n"
        '        run_id="hello_run",\n'
        "        tenant=tenant,\n"
        '        model="demo-model",\n'
        '        prompt="Say hello",\n'
        "        output_schema=HelloResult,\n"
        '        step_key=step.next("model"),\n'
        "    )\n"
        "    print(result.output.model_dump())\n"
        "    await kernel.close()\n\n\n"
        'if __name__ == "__main__":\n'
        "    asyncio.run(main())\n"
    )


def _render_init_readme(*, profile: str) -> str:
    return (
        "# Artana Starter\n\n"
        f"Profile: `{profile}`\n\n"
        "Run:\n\n"
        "```bash\n"
        "uv run python app.py\n"
        "```\n"
    )


def _kernel_for_reads(*, store: EventStore) -> ArtanaKernel:
    return ArtanaKernel(store=store, model_port=_NoopModelPort())


def _serialize_event(event: KernelEvent) -> dict[str, object]:
    payload = event.payload
    serialized_payload: object
    if isinstance(payload, BaseModel):
        serialized_payload = payload.model_dump(mode="json")
    else:
        serialized_payload = str(payload)
    return {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "seq": event.seq,
        "tenant_id": event.tenant_id,
        "timestamp": event.timestamp.isoformat(),
        "event_type": event.event_type.value,
        "parent_step_key": event.parent_step_key,
        "payload": serialized_payload,
    }


def _print_event(event: KernelEvent, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(_serialize_event(event), ensure_ascii=False, sort_keys=True))
        return
    parent = event.parent_step_key if event.parent_step_key is not None else "-"
    print(
        f"{event.seq}\t{event.timestamp.isoformat()}\t"
        f"{event.event_type.value}\t{parent}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="artana")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)

    list_parser = run_subparsers.add_parser("list")
    _add_store_arguments(list_parser)
    list_parser.add_argument("--tenant", required=True)
    list_parser.add_argument(
        "--since",
        default=None,
        help="ISO-8601 timestamp filter (inclusive).",
    )
    _add_json_argument(list_parser)

    tail_parser = run_subparsers.add_parser("tail")
    tail_parser.add_argument("run_id")
    _add_store_arguments(tail_parser)
    tail_parser.add_argument("--tenant", required=True)
    tail_parser.add_argument("--follow", action="store_true")
    tail_parser.add_argument("--since-seq", type=int, default=0)
    _add_json_argument(tail_parser)

    verify_parser = run_subparsers.add_parser("verify-ledger")
    verify_parser.add_argument("run_id")
    _add_store_arguments(verify_parser)
    verify_parser.add_argument("--tenant", required=True)
    _add_json_argument(verify_parser)

    status_parser = run_subparsers.add_parser("status")
    status_parser.add_argument("run_id")
    _add_store_arguments(status_parser)
    status_parser.add_argument("--tenant", required=True)
    _add_json_argument(status_parser)

    summaries_parser = run_subparsers.add_parser("summaries")
    summaries_parser.add_argument("run_id")
    _add_store_arguments(summaries_parser)
    summaries_parser.add_argument("--tenant", required=True)
    summaries_parser.add_argument("--type", dest="summary_type", default=None)
    summaries_parser.add_argument("--limit", type=int, default=20)
    _add_json_argument(summaries_parser)

    artifacts_parser = run_subparsers.add_parser("artifacts")
    artifacts_parser.add_argument("run_id")
    _add_store_arguments(artifacts_parser)
    artifacts_parser.add_argument("--tenant", required=True)
    _add_json_argument(artifacts_parser)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.add_argument("--profile", choices=("enforced", "dev"), default="enforced")
    init_parser.add_argument("--force", action="store_true")

    return parser


def _add_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default=".state.db")
    parser.add_argument("--dsn", default=None)


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit JSON output.",
    )


def _require_cli_tenant(args: argparse.Namespace) -> TenantContext:
    tenant_id = getattr(args, "tenant", None)
    if not isinstance(tenant_id, str) or tenant_id.strip() == "":
        raise ValueError("--tenant is required for tenant-scoped run reads.")
    return TenantContext(
        tenant_id=tenant_id,
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


def _assert_cli_run_access(
    *,
    run_id: str,
    tenant: TenantContext,
    events: list[KernelEvent],
) -> None:
    if not events:
        raise ValueError(f"No events found for run_id={run_id!r}.")
    if events[0].tenant_id != tenant.tenant_id:
        raise ValueError(
            "Run tenant mismatch. "
            f"run tenant={events[0].tenant_id!r}, request tenant={tenant.tenant_id!r}."
        )


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
