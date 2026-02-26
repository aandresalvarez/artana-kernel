from __future__ import annotations

import json
from pathlib import Path

import pytest

from artana.ports.tool import ToolExecutionContext
from artana.tools import CodingHarnessTools, ObservabilityTools


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        run_id="run_tools",
        tenant_id="org_tools",
        idempotency_key="idemp_tools_1",
        request_event_id=None,
        tool_version="1.0.0",
        schema_version="1",
    )


@pytest.mark.asyncio
async def test_coding_harness_tools_bundle(tmp_path: Path) -> None:
    bundle = CodingHarnessTools(sandbox_root=str(tmp_path))
    registry = bundle.registry()
    names = {tool.name for tool in registry.to_all_tool_definitions()}
    assert names == {"apply_patch", "create_worktree", "git_diff", "read_file"}

    worktree_result = await registry.call("create_worktree", "{}", context=_context())
    worktree_payload = json.loads(worktree_result.result_json)
    assert worktree_payload["created"] is True

    write_result = await registry.call(
        "apply_patch",
        '{"path":"src/demo.txt","content":"hello"}',
        context=_context(),
    )
    write_payload = json.loads(write_result.result_json)
    assert write_payload["ok"] is True

    read_result = await registry.call(
        "read_file",
        '{"path":"src/demo.txt"}',
        context=_context(),
    )
    read_payload = json.loads(read_result.result_json)
    assert read_payload["ok"] is True
    assert read_payload["content"] == "hello"


@pytest.mark.asyncio
async def test_observability_tools_bundle(tmp_path: Path) -> None:
    logs_path = tmp_path / "service.log"
    logs_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"latency_ms": 42}', encoding="utf-8")

    bundle = ObservabilityTools(root=str(tmp_path))
    registry = bundle.registry()
    names = {tool.name for tool in registry.to_all_tool_definitions()}
    assert names == {"query_logs", "query_metrics"}

    logs_result = await registry.call(
        "query_logs",
        '{"file":"service.log","limit":2}',
        context=_context(),
    )
    logs_payload = json.loads(logs_result.result_json)
    assert logs_payload["ok"] is True
    assert logs_payload["lines"] == ["line2", "line3"]

    metrics_result = await registry.call(
        "query_metrics",
        '{"file":"metrics.json"}',
        context=_context(),
    )
    metrics_payload = json.loads(metrics_result.result_json)
    assert metrics_payload["ok"] is True
    assert metrics_payload["metrics"]["latency_ms"] == 42


@pytest.mark.asyncio
async def test_tool_bundles_handle_non_utf8_files(tmp_path: Path) -> None:
    binary_path = tmp_path / "raw.bin"
    binary_path.write_bytes(b"\xff\xfe\xfd")

    coding_registry = CodingHarnessTools(sandbox_root=str(tmp_path)).registry()
    coding_result = await coding_registry.call(
        "read_file",
        '{"path":"raw.bin"}',
        context=_context(),
    )
    coding_payload = json.loads(coding_result.result_json)
    assert coding_payload["ok"] is False
    assert coding_payload["error"] == "invalid_utf8"

    observability_registry = ObservabilityTools(root=str(tmp_path)).registry()
    logs_result = await observability_registry.call(
        "query_logs",
        '{"file":"raw.bin"}',
        context=_context(),
    )
    logs_payload = json.loads(logs_result.result_json)
    assert logs_payload["ok"] is False
    assert logs_payload["error"] == "invalid_utf8"

    metrics_result = await observability_registry.call(
        "query_metrics",
        '{"file":"raw.bin"}',
        context=_context(),
    )
    metrics_payload = json.loads(metrics_result.result_json)
    assert metrics_payload["ok"] is False
    assert metrics_payload["error"] == "invalid_utf8"
