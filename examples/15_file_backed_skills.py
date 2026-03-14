from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

from artana import (
    ArtanaKernel,
    AutonomousAgent,
    ContextBuilder,
    FilesystemSkillRegistry,
    TenantContext,
)
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class SkillDemoResult(BaseModel):
    done: bool
    notes: list[str] = Field(default_factory=list)


class FileBackedSkillModelPort:
    def __init__(self) -> None:
        self.calls = 0
        self.allowed_tool_batches: list[list[str]] = []
        self.system_message_batches: list[list[str]] = []

    async def complete(
        self,
        request: ModelRequest[OutputModelT],
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        self.allowed_tool_batches.append([tool.name for tool in request.allowed_tools])
        self.system_message_batches.append(
            [message.content for message in request.messages if message.role == "system"]
        )

        tool_calls: tuple[ToolCall, ...]
        notes: list[str] = []
        if self.calls == 1:
            tool_calls = (
                ToolCall(
                    tool_name="load_skill",
                    arguments_json='{"skill_name":"writing_style"}',
                    tool_call_id="call_load_style_1",
                ),
            )
        elif self.calls == 2:
            tool_calls = (
                ToolCall(
                    tool_name="load_skill",
                    arguments_json='{"skill_name":"demo_reader"}',
                    tool_call_id="call_load_reader_2",
                ),
            )
        elif self.calls == 3:
            tool_calls = (
                ToolCall(
                    tool_name="read_demo_file",
                    arguments_json="{}",
                    tool_call_id="call_read_demo_file_3",
                ),
            )
        else:
            tool_calls = ()
            notes = [
                "Loaded instruction-only skill: writing_style",
                "Loaded bundled-tool skill: demo_reader",
                f"Allowed tool batches: {self.allowed_tool_batches}",
                (
                    "Observed active skill instructions on later turns: "
                    f"{any(
                        'Keep responses terse and grounded.' in message
                        for batch in self.system_message_batches
                        for message in batch
                    )}"
                ),
            ]

        return ModelResult(
            output=request.output_schema.model_validate(
                {
                    "done": not tool_calls,
                    "notes": notes,
                }
            ),
            usage=ModelUsage(prompt_tokens=8, completion_tokens=4, cost_usd=0.0),
            tool_calls=tool_calls,
        )


def _write_skill_file(
    root: Path,
    *,
    slug: str,
    name: str,
    summary: str,
    instructions: str,
    tools: tuple[str, ...] = (),
) -> None:
    path = root / slug / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {name}",
        "version: 1.0.0",
        f"summary: {summary}",
    ]
    if tools:
        lines.append("tools:")
        lines.extend(f"  - {tool}" for tool in tools)
    lines.extend(("---", instructions))
    path.write_text("\n".join(lines), encoding="utf-8")


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_file_backed_skills",
        capabilities=frozenset(),
        budget_usd_limit=2.0,
    )


async def main() -> None:
    database_path = Path("examples/.state_file_backed_skills.db")
    scratch_root = Path("examples/.tmp_file_backed_skills")

    if database_path.exists():
        database_path.unlink()
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)

    demo_file = scratch_root / "demo.txt"
    demo_file.write_text("skills-ready\n", encoding="utf-8")
    skills_root = scratch_root / "skills"
    _write_skill_file(
        skills_root,
        slug="writing_style",
        name="writing_style",
        summary="Instruction-only style guidance.",
        instructions="Keep responses terse and grounded.",
    )
    _write_skill_file(
        skills_root,
        slug="demo_reader",
        name="demo_reader",
        summary="Unlock the demo file reader.",
        instructions="Use read_demo_file once the demo reader skill is active.",
        tools=("read_demo_file",),
    )

    model_port = FileBackedSkillModelPort()
    kernel = ArtanaKernel(
        store=SQLiteStore(str(database_path)),
        model_port=model_port,
        middleware=ArtanaKernel.default_middleware_stack(),
    )

    @kernel.tool()
    async def read_demo_file() -> str:
        return json.dumps(
            {
                "path": str(demo_file),
                "content": demo_file.read_text(encoding="utf-8").strip(),
            }
        )

    registry = FilesystemSkillRegistry(skills_root)
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(
            progressive_skills=True,
            skill_registry=registry,
        ),
    )

    try:
        result = await agent.run(
            run_id="file_backed_skills_demo",
            tenant=_tenant(),
            model="local-file-backed-skills-demo",
            prompt=(
                "Load the style skill, then load the reader skill, then read the demo file."
            ),
            output_schema=SkillDemoResult,
            max_iterations=5,
        )

        events = await kernel.get_events(run_id="file_backed_skills_demo", tenant=_tenant())
        tool_results = [
            json.loads(event.payload.result_json)
            for event in events
            if event.event_type.value == "tool_completed"
            and getattr(event.payload, "kind", None) == "tool_completed"
        ]

        print("Filesystem-backed runtime skills demo:")
        print(result.model_dump_json(indent=2))
        print("Tool results:", json.dumps(tool_results, indent=2))
        print("Allowed tool batches:", json.dumps(model_port.allowed_tool_batches, indent=2))
        observed_skill_instructions = any(
            "Keep responses terse and grounded." in message
            for batch in model_port.system_message_batches
            for message in batch
        )
        print("Observed active skill instructions:", observed_skill_instructions)
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()
        if scratch_root.exists():
            shutil.rmtree(scratch_root)


if __name__ == "__main__":
    asyncio.run(main())
