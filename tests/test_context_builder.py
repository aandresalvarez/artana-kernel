from __future__ import annotations

from pathlib import Path

import pytest

from artana.agent.context import ContextBuilder
from artana.events import ChatMessage
from artana.models import TenantContext


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_context_builder",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_context_builder_ignores_non_utf8_workspace_context(tmp_path: Path) -> None:
    workspace_context = tmp_path / "ACTIVE_PLAN.bin"
    workspace_context.write_bytes(b"\xff\xfe\xfd")
    context_builder = ContextBuilder(
        progressive_skills=False,
        workspace_context_path=str(workspace_context),
    )

    messages = await context_builder.build_messages(
        run_id="run_workspace_context_non_utf8",
        tenant=_tenant(),
        short_term_messages=(ChatMessage(role="user", content="continue"),),
        system_prompt="You are the agent.",
        active_skills=frozenset(),
        available_skill_summaries=None,
        memory_text=None,
    )

    assert messages[0].role == "system"
    assert "Workspace Context / Active Plan:" not in messages[0].content
