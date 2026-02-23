from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from artana.agent.compaction import CompactionStrategy, CompactionSummary
from artana.agent.context import ContextBuilder
from artana.agent.experience import ExperienceRule, ReflectionResult
from artana.agent.memory import InMemoryMemoryStore, MemoryStore
from artana.agent.model_steps import execute_model_step
from artana.agent.runtime_tools import RuntimeToolManager, extract_loaded_skill_name
from artana.agent.tool_args import model_from_tool_arguments_json
from artana.events import ChatMessage
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import ToolCall

OutputT = TypeVar("OutputT", bound=BaseModel)


class AutonomousAgent:
    LOAD_SKILL_NAME = "load_skill"
    CORE_MEMORY_APPEND = "core_memory_append"
    CORE_MEMORY_REPLACE = "core_memory_replace"
    CORE_MEMORY_SEARCH = "core_memory_search"

    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        context_builder: ContextBuilder | None = None,
        compaction: CompactionStrategy | None = None,
        memory_store: MemoryStore | None = None,
        auto_reflect: bool = False,
        reflection_model: str = "gpt-4o-mini",
    ) -> None:
        if context_builder is None:
            context_builder = ContextBuilder()
        if memory_store is None:
            memory_store = context_builder.memory_store
        if memory_store is None:
            memory_store = InMemoryMemoryStore()
            context_builder.memory_store = memory_store
        if auto_reflect and context_builder.experience_store is None:
            raise ValueError(
                "auto_reflect requires context_builder.experience_store to be configured."
            )
        if auto_reflect and not context_builder.task_category:
            raise ValueError(
                "auto_reflect requires context_builder.task_category to be configured."
            )

        self._kernel = kernel
        self._context_builder = context_builder
        self._compaction = compaction
        self._memory_store = memory_store
        self._auto_reflect = auto_reflect
        self._reflection_model = reflection_model
        self._progressive_skills = context_builder.progressive_skills
        self._runtime_tools = RuntimeToolManager(
            kernel=kernel,
            memory_store=memory_store,
            progressive_skills=self._progressive_skills,
            load_skill_name=self.LOAD_SKILL_NAME,
            core_memory_append=self.CORE_MEMORY_APPEND,
            core_memory_replace=self.CORE_MEMORY_REPLACE,
            core_memory_search=self.CORE_MEMORY_SEARCH,
        )
        self._runtime_tools.ensure_registered()

    async def run(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        system_prompt: str = "You are a helpful autonomous agent.",
        prompt: str,
        output_schema: type[OutputT],
        max_iterations: int = 15,
    ) -> OutputT:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be >= 1.")
        await self._ensure_run_exists(run_id=run_id, tenant=tenant)

        short_term_messages: list[ChatMessage] = [ChatMessage(role="user", content=prompt)]
        loaded_skills: set[str] = set()

        for iteration in range(1, max_iterations + 1):
            short_term_messages = list(
                await self._compact_if_needed(
                    run_id=run_id,
                    tenant=tenant,
                    model=model,
                    iteration=iteration,
                    short_term_messages=tuple(short_term_messages),
                )
            )
            visible_tool_names = self._runtime_tools.visible_tool_names(
                loaded_skills=loaded_skills,
                tenant_capabilities=tenant.capabilities,
            )
            memory_text = await self._memory_store.load(run_id=run_id)
            available_skill_summaries = self._runtime_tools.available_skill_summaries(
                tenant_capabilities=tenant.capabilities
            )

            context_messages = await self._context_builder.build_messages(
                run_id=run_id,
                tenant=tenant,
                short_term_messages=tuple(short_term_messages),
                system_prompt=system_prompt,
                active_skills=frozenset(loaded_skills),
                available_skill_summaries=available_skill_summaries,
                memory_text=memory_text,
            )
            model_result = await execute_model_step(
                kernel=self._kernel,
                run_id=run_id,
                tenant=tenant,
                model=model,
                messages=context_messages,
                output_schema=output_schema,
                step_key=f"turn_{iteration}_model",
                visible_tool_names=visible_tool_names,
            )
            if not model_result.tool_calls:
                if self._auto_reflect:
                    await self._run_reflection(
                        run_id=run_id,
                        tenant=tenant,
                        iteration=iteration,
                        messages=(
                            *context_messages,
                            ChatMessage(
                                role="assistant",
                                content=(
                                    "Final structured output: "
                                    + model_result.output.model_dump_json()
                                ),
                            ),
                        ),
                    )
                return model_result.output

            short_term_messages.append(
                ChatMessage(
                    role="assistant",
                    content="Action requested: " + _serialize_tool_calls(model_result.tool_calls),
                )
            )

            for index, tool_call in enumerate(model_result.tool_calls, start=1):
                self._ensure_tool_is_loaded(
                    tool_name=tool_call.tool_name,
                    visible_tool_names=visible_tool_names,
                )
                tool_result = await self._kernel.step_tool(
                    run_id=run_id,
                    tenant=tenant,
                    tool_name=tool_call.tool_name,
                    arguments=model_from_tool_arguments_json(tool_call.arguments_json),
                    step_key=f"turn_{iteration}_tool_{index}_{tool_call.tool_name}",
                )
                short_term_messages.append(
                    ChatMessage(
                        role="tool",
                        content=f"{tool_call.tool_name}: {tool_result.result_json}",
                    )
                )
                if tool_call.tool_name == self.LOAD_SKILL_NAME:
                    maybe_skill = extract_loaded_skill_name(tool_result.result_json)
                    if maybe_skill is not None:
                        loaded_skills.add(maybe_skill)

        raise RuntimeError(
            f"Agent exceeded max iterations ({max_iterations}) without reaching an answer."
        )

    async def _run_reflection(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        iteration: int,
        messages: tuple[ChatMessage, ...],
    ) -> None:
        experience_store = self._context_builder.experience_store
        task_category = self._context_builder.task_category
        if experience_store is None or not task_category:
            return

        reflection_result = await execute_model_step(
            kernel=self._kernel,
            run_id=run_id,
            tenant=tenant,
            model=self._reflection_model,
            messages=self._reflection_messages(
                tenant_id=tenant.tenant_id,
                task_category=task_category,
                transcript_messages=messages,
            ),
            output_schema=ReflectionResult,
            step_key=f"turn_{iteration}_reflection",
            visible_tool_names=set(),
        )
        if reflection_result.tool_calls:
            raise RuntimeError("Reflection step returned tool calls; expected none.")

        extracted_rules = self._normalized_rules(
            rules=reflection_result.output.extracted_rules,
            tenant_id=tenant.tenant_id,
            task_category=task_category,
        )
        if extracted_rules:
            await experience_store.save_rules(extracted_rules)

    async def _ensure_run_exists(self, *, run_id: str, tenant: TenantContext) -> None:
        try:
            await self._kernel.load_run(run_id=run_id)
        except ValueError:
            await self._kernel.start_run(tenant=tenant, run_id=run_id)

    async def _compact_if_needed(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        iteration: int,
        short_term_messages: tuple[ChatMessage, ...],
    ) -> tuple[ChatMessage, ...]:
        strategy = self._compaction
        if strategy is None:
            return short_term_messages
        if not strategy.should_compact(messages=short_term_messages, model=model):
            return short_term_messages
        if len(short_term_messages) <= strategy.keep_recent_messages:
            return short_term_messages

        if strategy.keep_recent_messages == 0:
            window = short_term_messages
            recent_messages: tuple[ChatMessage, ...] = ()
        else:
            window = short_term_messages[: -strategy.keep_recent_messages]
            recent_messages = short_term_messages[-strategy.keep_recent_messages :]
        if not window:
            return short_term_messages

        summary_input = (
            ChatMessage(
                role="system",
                content=(
                    "Summarize the following history into a compact set of durable facts "
                    "and decisions. Preserve names, entities, constraints, and outcomes."
                ),
            ),
            *window,
        )
        compact_result = await execute_model_step(
            kernel=self._kernel,
            run_id=run_id,
            tenant=tenant,
            model=strategy.summarize_with_model,
            messages=summary_input,
            output_schema=CompactionSummary,
            step_key=f"turn_{iteration}_compact",
            visible_tool_names=set(),
        )
        compacted_history = [
            ChatMessage(
                role="system",
                content=f"Past Events Summary: {compact_result.output.summary}",
            )
        ]
        compacted_history.extend(recent_messages)
        return tuple(compacted_history)

    def _ensure_tool_is_loaded(
        self,
        *,
        tool_name: str,
        visible_tool_names: set[str] | None,
    ) -> None:
        if not self._progressive_skills or visible_tool_names is None:
            return
        if tool_name in visible_tool_names:
            return
        if tool_name == self.LOAD_SKILL_NAME:
            return
        raise RuntimeError(
            f"Tool {tool_name!r} is not currently loaded. "
            "Call load_skill(skill_name=...) first."
        )

    def _reflection_messages(
        self,
        *,
        tenant_id: str,
        task_category: str,
        transcript_messages: tuple[ChatMessage, ...],
    ) -> tuple[ChatMessage, ...]:
        transcript = "\n".join(
            f"{message.role}: {message.content}" for message in transcript_messages
        )
        return (
            ChatMessage(
                role="system",
                content=(
                    "Extract reusable learnings from this completed run. "
                    "Return only durable rules that improve future runs for the same task. "
                    "If no reusable learning exists, return an empty extracted_rules list."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"tenant_id={tenant_id}\n"
                    f"task_category={task_category}\n\n"
                    "Return rules using these exact tenant_id and task_category values.\n\n"
                    "Conversation transcript:\n"
                    f"{transcript}"
                ),
            ),
        )

    def _normalized_rules(
        self,
        *,
        rules: list[ExperienceRule],
        tenant_id: str,
        task_category: str,
    ) -> list[ExperienceRule]:
        return [
            rule.model_copy(
                update={
                    "tenant_id": tenant_id,
                    "task_category": task_category,
                }
            )
            for rule in rules
        ]


def _serialize_tool_calls(tool_calls: tuple[ToolCall, ...]) -> str:
    return json.dumps(
        [
            {
                "tool_name": tool_call.tool_name,
                "arguments_json": tool_call.arguments_json,
            }
            for tool_call in tool_calls
        ]
    )


__all__ = ["AutonomousAgent"]
