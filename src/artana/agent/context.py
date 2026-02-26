from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from pathlib import Path

from artana.agent.experience import ExperienceRule, ExperienceStore
from artana.agent.memory import MemoryStore
from artana.events import ChatMessage
from artana.models import TenantContext


class ContextBuilder(ABC):
    VERSION = "context_builder.v1"

    def __init__(
        self,
        *,
        identity: str = "You are a helpful autonomous agent.",
        memory_store: MemoryStore | None = None,
        experience_store: ExperienceStore | None = None,
        task_category: str | None = None,
        progressive_skills: bool = True,
        workspace_context_path: str | None = None,
    ) -> None:
        self.identity = identity
        self.memory_store = memory_store
        self.experience_store = experience_store
        self.task_category = task_category
        self.progressive_skills = progressive_skills
        self.workspace_context_path = workspace_context_path

    @property
    def version(self) -> str:
        return self.VERSION

    async def build_messages(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        short_term_messages: tuple[ChatMessage, ...],
        system_prompt: str,
        active_skills: frozenset[str],
        available_skill_summaries: Mapping[str, str] | None,
        memory_text: str | None,
    ) -> tuple[ChatMessage, ...]:
        sections: list[str] = [self.identity, system_prompt]
        workspace_context = self._read_workspace_context()
        if workspace_context is not None:
            sections.append(f"Workspace Context / Active Plan:\n{workspace_context}")
        if memory_text:
            sections.append(f"Long-Term Memory:\n{memory_text}")
        experience_rules = await self._load_experience_rules(tenant_id=tenant.tenant_id)
        if experience_rules:
            sections.append(
                self._format_experience_panel(experience_rules=experience_rules)
            )
        if self.progressive_skills:
            sections.append(
                self._format_skill_panel(
                    active_skills=active_skills,
                    available_skill_summaries=available_skill_summaries,
                )
            )
        return (ChatMessage(role="system", content="\n\n".join(sections)),) + short_term_messages

    def _read_workspace_context(self) -> str | None:
        if self.workspace_context_path is None:
            return None
        path = Path(self.workspace_context_path).expanduser()
        try:
            if not path.exists() or not path.is_file():
                return None
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        if content == "":
            return None
        return content

    def _format_skill_panel(
        self,
        *,
        active_skills: frozenset[str],
        available_skill_summaries: Mapping[str, str] | None,
    ) -> str:
        loaded = ", ".join(sorted(active_skills)) or "(none)"
        available_names = (
            ", ".join(sorted(available_skill_summaries.keys()))
            if available_skill_summaries
            else ""
        )
        available_block = f"Available tools: [{available_names}]"
        if available_skill_summaries:
            summaries = ", ".join(
                f"{name}: {summary}"
                for name, summary in sorted(available_skill_summaries.items())
            )
        else:
            summaries = "(none)"
        return (
            f"{available_block}\n"
            f"Tool summaries: {summaries}\n"
            f"Loaded Skills: {loaded}\n"
            "Call load_skill(skill_name=\"<name>\") when you need full "
            "tool arguments and constraints."
        )

    async def _load_experience_rules(
        self,
        *,
        tenant_id: str,
    ) -> tuple[ExperienceRule, ...]:
        if self.experience_store is None:
            return ()
        if not self.task_category:
            return ()
        rules = await self.experience_store.get_rules(
            tenant_id=tenant_id,
            task_category=self.task_category,
        )
        return tuple(rules)

    def _format_experience_panel(
        self,
        *,
        experience_rules: tuple[ExperienceRule, ...],
    ) -> str:
        lines = ["[PAST LEARNINGS FOR THIS TASK]"]
        for rule in experience_rules:
            lines.append(f"{rule.rule_type.value.upper()}: {rule.content}")
        return "\n".join(lines)


__all__ = ["ContextBuilder"]
