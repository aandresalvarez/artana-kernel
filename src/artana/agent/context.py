from __future__ import annotations

import json
from abc import ABC
from collections.abc import Iterable, Mapping
from pathlib import Path

from artana.agent.experience import ExperienceRule, ExperienceStore
from artana.agent.memory import MemoryStore
from artana.events import ChatMessage
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.skills import SkillRegistry


class ContextBuilder(ABC):
    VERSION = "context_builder.v2"

    def __init__(
        self,
        *,
        identity: str = "You are a helpful autonomous agent.",
        memory_store: MemoryStore | None = None,
        experience_store: ExperienceStore | None = None,
        task_category: str | None = None,
        progressive_skills: bool = True,
        workspace_context_path: str | None = None,
        skill_registry: SkillRegistry | None = None,
        allowed_skill_names: Iterable[str] | None = None,
        preload_skill_names: Iterable[str] | None = None,
    ) -> None:
        self.identity = identity
        self.memory_store = memory_store
        self.experience_store = experience_store
        self.task_category = task_category
        self.progressive_skills = progressive_skills
        self.workspace_context_path = workspace_context_path
        self.skill_registry = skill_registry
        self.allowed_skill_names = self._normalize_optional_skill_names(allowed_skill_names)
        self.preload_skill_names = self._normalize_skill_names(preload_skill_names)
        self._validate_skill_configuration()

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
        available_block = f"Available skills/tools: [{available_names}]"
        if available_skill_summaries:
            summaries = ", ".join(
                f"{name}: {summary}"
                for name, summary in sorted(available_skill_summaries.items())
            )
        else:
            summaries = "(none)"
        return (
            f"{available_block}\n"
            f"Skill summaries: {summaries}\n"
            f"Loaded Skills: {loaded}\n"
            "Call load_skill(skill_name=\"<name>\") when you need full "
            "skill instructions or hidden tool arguments and constraints."
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

    def _normalize_optional_skill_names(
        self,
        skill_names: Iterable[str] | None,
    ) -> frozenset[str] | None:
        if skill_names is None:
            return None
        return self._normalize_skill_names(skill_names)

    def _normalize_skill_names(self, skill_names: Iterable[str] | None) -> frozenset[str]:
        if skill_names is None:
            return frozenset()
        normalized: set[str] = set()
        for skill_name in skill_names:
            if not isinstance(skill_name, str):
                raise ValueError("Skill names must be strings.")
            stripped = skill_name.strip()
            if stripped == "":
                raise ValueError("Skill names must not be empty.")
            normalized.add(stripped)
        return frozenset(normalized)

    def _validate_skill_configuration(self) -> None:
        if self.skill_registry is None:
            if self.allowed_skill_names is not None:
                raise ValueError("allowed_skill_names requires skill_registry to be configured.")
            if self.preload_skill_names:
                raise ValueError("preload_skill_names requires skill_registry to be configured.")
            return

        known_skill_names = self.skill_registry.skill_names()
        if self.allowed_skill_names is not None:
            unknown_allowed = self.allowed_skill_names - known_skill_names
            if unknown_allowed:
                unknown = ", ".join(sorted(unknown_allowed))
                raise ValueError(f"Unknown allowed_skill_names: {unknown}")

        unknown_preloads = self.preload_skill_names - known_skill_names
        if unknown_preloads:
            unknown = ", ".join(sorted(unknown_preloads))
            raise ValueError(f"Unknown preload_skill_names: {unknown}")

        if self.allowed_skill_names is not None:
            disallowed_preloads = self.preload_skill_names - self.allowed_skill_names
            if disallowed_preloads:
                invalid = ", ".join(sorted(disallowed_preloads))
                raise ValueError(
                    "preload_skill_names must be a subset of allowed_skill_names: "
                    f"{invalid}"
                )


class WorkspaceSnapshotContextBuilder(ContextBuilder):
    VERSION = "context_builder.workspace_snapshot.v1"

    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        base: ContextBuilder | None = None,
        workspace_summary_type: str = "harness_workspace_state",
    ) -> None:
        self._kernel = kernel
        self._base = base or ContextBuilder()
        self._workspace_summary_type = workspace_summary_type
        super().__init__(
            identity=self._base.identity,
            memory_store=self._base.memory_store,
            experience_store=self._base.experience_store,
            task_category=self._base.task_category,
            progressive_skills=self._base.progressive_skills,
            workspace_context_path=self._base.workspace_context_path,
            skill_registry=self._base.skill_registry,
            allowed_skill_names=self._base.allowed_skill_names,
            preload_skill_names=self._base.preload_skill_names,
        )

    @property
    def version(self) -> str:
        return f"{self._base.version}+workspace_snapshot.v1"

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
        messages = await self._base.build_messages(
            run_id=run_id,
            tenant=tenant,
            short_term_messages=short_term_messages,
            system_prompt=system_prompt,
            active_skills=active_skills,
            available_skill_summaries=available_skill_summaries,
            memory_text=memory_text,
        )
        workspace_panel = await self._load_workspace_panel(run_id=run_id, tenant=tenant)
        if workspace_panel is None:
            return messages
        if not messages:
            return (ChatMessage(role="system", content=workspace_panel),)
        first = messages[0]
        if first.role != "system":
            return (ChatMessage(role="system", content=workspace_panel),) + messages
        return (
            ChatMessage(role="system", content=f"{first.content}\n\n{workspace_panel}"),
        ) + messages[1:]

    async def _load_workspace_panel(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
    ) -> str | None:
        summary = await self._kernel.get_latest_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type=self._workspace_summary_type,
        )
        if summary is None:
            return None
        try:
            payload_obj = json.loads(summary.summary_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload_obj, dict):
            return None
        payload = payload_obj
        return self._format_workspace_snapshot(payload=payload)

    def _format_workspace_snapshot(self, *, payload: Mapping[str, object]) -> str:
        lines = ["[WORKSPACE STATE SNAPSHOT]"]
        domain = payload.get("domain")
        if isinstance(domain, str) and domain:
            lines.append(f"Domain: {domain}")
        question = payload.get("question")
        if isinstance(question, str) and question:
            lines.append(f"Question: {question}")
        active_plan = payload.get("active_plan")
        if isinstance(active_plan, str) and active_plan:
            lines.append(f"Active Plan: {active_plan}")
        graph_summary = payload.get("graph_summary")
        if isinstance(graph_summary, str) and graph_summary:
            lines.append(f"Graph Summary: {graph_summary}")
        evidence_count = payload.get("evidence_count")
        if isinstance(evidence_count, int):
            lines.append(f"Evidence Count: {evidence_count}")
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, Mapping):
            artifact_names = ", ".join(sorted(str(key) for key in artifacts))
            if artifact_names:
                lines.append(f"Artifacts: {artifact_names}")
        self._append_list(lines, label="Constraints", value=payload.get("constraints"))
        self._append_list(lines, label="Open Tasks", value=payload.get("open_tasks"))
        self._append_list(
            lines,
            label="Unresolved Contradictions",
            value=payload.get("unresolved_contradictions"),
        )
        self._append_list(lines, label="Allowed Tools", value=payload.get("allowed_tools"))
        return "\n".join(lines)

    def _append_list(
        self,
        lines: list[str],
        *,
        label: str,
        value: object,
    ) -> None:
        if not isinstance(value, list):
            return
        rendered = [str(item).strip() for item in value if str(item).strip()]
        if not rendered:
            return
        lines.append(f"{label}:")
        lines.extend(f"- {item}" for item in rendered)


__all__ = ["ContextBuilder", "WorkspaceSnapshotContextBuilder"]
