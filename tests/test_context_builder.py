from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel, FilesystemSkillRegistry
from artana.agent.context import ContextBuilder, WorkspaceSnapshotContextBuilder
from artana.events import ChatMessage
from artana.harness import BaseHarness, HarnessContext, WorkspaceState
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_context_builder",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


def _write_skill_file(
    root: Path,
    *,
    slug: str,
    name: str,
    summary: str,
) -> Path:
    path = root / slug / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                "version: 1.0.0",
                f"summary: {summary}",
                "---",
                "Skill instructions.",
            ]
        ),
        encoding="utf-8",
    )
    return path


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


def test_context_builder_rejects_skill_names_without_registry() -> None:
    with pytest.raises(ValueError, match="allowed_skill_names requires skill_registry"):
        ContextBuilder(allowed_skill_names=("alpha",))

    with pytest.raises(ValueError, match="preload_skill_names requires skill_registry"):
        ContextBuilder(preload_skill_names=("alpha",))


def test_context_builder_validates_allowed_and_preloaded_skill_names(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_file(skills_root, slug="alpha", name="alpha", summary="Alpha skill")
    _write_skill_file(skills_root, slug="beta", name="beta", summary="Beta skill")
    registry = FilesystemSkillRegistry(skills_root)

    with pytest.raises(ValueError, match="Unknown allowed_skill_names: missing"):
        ContextBuilder(skill_registry=registry, allowed_skill_names=("missing",))

    with pytest.raises(ValueError, match="Unknown preload_skill_names: missing"):
        ContextBuilder(skill_registry=registry, preload_skill_names=("missing",))

    with pytest.raises(
        ValueError,
        match="preload_skill_names must be a subset of allowed_skill_names: beta",
    ):
        ContextBuilder(
            skill_registry=registry,
            allowed_skill_names=("alpha",),
            preload_skill_names=("beta",),
        )


class UnusedModelPort:
    async def complete(
        self,
        request: ModelRequest[OutputModelT],
    ) -> ModelResult[OutputModelT]:
        raise AssertionError("Model should not be called in context builder tests.")


class DummyOutput(BaseModel):
    ok: bool


class WorkspaceHarness(BaseHarness[WorkspaceState]):
    async def step(self, *, context: HarnessContext) -> WorkspaceState:
        raise AssertionError("WorkspaceHarness.step() should not be called in this test.")


@pytest.mark.asyncio
async def test_workspace_snapshot_context_builder_injects_workspace_state(tmp_path: Path) -> None:
    kernel = ArtanaKernel(
        store=SQLiteStore(str(tmp_path / "state.db")),
        model_port=UnusedModelPort(),
    )
    tenant = _tenant()
    run_id = "run_workspace_snapshot_context"
    harness = WorkspaceHarness(kernel=kernel, tenant=tenant)
    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await harness.set_workspace_state(
            run_id=run_id,
            tenant=tenant,
            workspace_state=WorkspaceState(
                domain="research",
                question="What evidence links MED13 to transcription control?",
                open_tasks=["Compare contradictory papers"],
                allowed_tools=["search_literature", "score_evidence"],
            ),
            step_key="workspace_state_seed",
        )

        context_builder = WorkspaceSnapshotContextBuilder(
            kernel=kernel,
            base=ContextBuilder(progressive_skills=False),
        )
        messages = await context_builder.build_messages(
            run_id=run_id,
            tenant=tenant,
            short_term_messages=(ChatMessage(role="user", content="continue"),),
            system_prompt="You are the agent.",
            active_skills=frozenset(),
            available_skill_summaries=None,
            memory_text=None,
        )

        assert messages[0].role == "system"
        assert "[WORKSPACE STATE SNAPSHOT]" in messages[0].content
        assert (
            "Question: What evidence links MED13 to transcription control?"
            in messages[0].content
        )
        assert "Allowed Tools:" in messages[0].content
        assert "- search_literature" in messages[0].content
    finally:
        await kernel.close()
