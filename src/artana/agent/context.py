from __future__ import annotations

from abc import ABC
from collections.abc import Mapping

from artana.agent.memory import MemoryStore
from artana.events import ChatMessage
from artana.models import TenantContext


class ContextBuilder(ABC):
    def __init__(
        self,
        *,
        identity: str = "You are a helpful autonomous agent.",
        memory_store: MemoryStore | None = None,
        progressive_skills: bool = True,
    ) -> None:
        self.identity = identity
        self.memory_store = memory_store
        self.progressive_skills = progressive_skills

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
        if memory_text:
            sections.append(f"Long-Term Memory:\n{memory_text}")
        if self.progressive_skills:
            sections.append(
                self._format_skill_panel(
                    active_skills=active_skills,
                    available_skill_summaries=available_skill_summaries,
                )
            )
        return (ChatMessage(role="system", content="\n\n".join(sections)),) + short_term_messages

    def _format_skill_panel(
        self,
        *,
        active_skills: frozenset[str],
        available_skill_summaries: Mapping[str, str] | None,
    ) -> str:
        loaded = ", ".join(sorted(active_skills)) or "(none)"
        if available_skill_summaries:
            available = ", ".join(
                f"{name}: {summary}"
                for name, summary in sorted(available_skill_summaries.items())
            )
        else:
            available = "(none)"
        return (
            f"Available Skills: {available}\n"
            f"Loaded Skills: {loaded}\n"
            "Call load_skill(skill_name=\"<name>\") when you need full "
            "tool arguments and constraints."
        )


__all__ = ["ContextBuilder"]
