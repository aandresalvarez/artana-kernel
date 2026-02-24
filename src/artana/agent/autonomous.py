from __future__ import annotations

import hashlib
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
from artana.canonicalization import canonical_json_dumps
from artana.events import ChatMessage, ToolCallMessage, ToolFunctionCall
from artana.kernel import ArtanaKernel, ContextVersion, ReplayPolicy
from artana.middleware import BudgetExceededError
from artana.models import TenantContext
from artana.ports.model import ToolCall

OutputT = TypeVar("OutputT", bound=BaseModel)


class MaxIterationsExceeded(RuntimeError):
    pass


class AgentRunFailed(RuntimeError):
    def __init__(self, *, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


class _CompactionArtifact(BaseModel):
    window_hash: str
    summary_hash: str
    summary: str
    strategy_version: str
    summarize_with_model: str
    window_message_count: int


class AutonomousAgent:
    LOAD_SKILL_NAME = "load_skill"
    CORE_MEMORY_APPEND = "core_memory_append"
    CORE_MEMORY_REPLACE = "core_memory_replace"
    CORE_MEMORY_SEARCH = "core_memory_search"
    QUERY_EVENT_HISTORY = "query_event_history"
    RECORD_INTENT_PLAN = "record_intent_plan"
    COMPACTION_ARTIFACT_SUMMARY_TYPE = "artifact::agent_compaction"

    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        context_builder: ContextBuilder | None = None,
        compaction: CompactionStrategy | None = None,
        memory_store: MemoryStore | None = None,
        auto_reflect: bool = False,
        reflection_model: str = "gpt-4o-mini",
        replay_policy: ReplayPolicy = "strict",
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
        self._replay_policy = replay_policy
        self._progressive_skills = context_builder.progressive_skills
        self._runtime_tools = RuntimeToolManager(
            kernel=kernel,
            memory_store=memory_store,
            progressive_skills=self._progressive_skills,
            load_skill_name=self.LOAD_SKILL_NAME,
            core_memory_append=self.CORE_MEMORY_APPEND,
            core_memory_replace=self.CORE_MEMORY_REPLACE,
            core_memory_search=self.CORE_MEMORY_SEARCH,
            query_event_history=self.QUERY_EVENT_HISTORY,
            record_intent_plan=self.RECORD_INTENT_PLAN,
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

        active_run_id = run_id
        short_term_messages: list[ChatMessage] = [ChatMessage(role="user", content=prompt)]
        loaded_skills: set[str] = set()

        for iteration in range(1, max_iterations + 1):
            try:
                short_term_messages = list(
                    await self._compact_if_needed(
                        run_id=active_run_id,
                        tenant=tenant,
                        model=model,
                        iteration=iteration,
                        short_term_messages=tuple(short_term_messages),
                    )
                )
            except BudgetExceededError as exc:
                raise AgentRunFailed(
                    status="failed_budget_exceeded",
                    message=(
                        "AutonomousAgent stopped because the tenant budget was exhausted "
                        f"during compaction: {exc}"
                    ),
                ) from exc
            try:
                visible_tool_names = self._runtime_tools.visible_tool_names(
                    loaded_skills=loaded_skills,
                    tenant_capabilities=tenant.capabilities,
                )
                memory_text = await self._memory_store.load(run_id=active_run_id)
                available_skill_summaries = self._runtime_tools.available_skill_summaries(
                    tenant_capabilities=tenant.capabilities
                )

                context_messages = await self._context_builder.build_messages(
                    run_id=active_run_id,
                    tenant=tenant,
                    short_term_messages=tuple(short_term_messages),
                    system_prompt=system_prompt,
                    active_skills=frozenset(loaded_skills),
                    available_skill_summaries=available_skill_summaries,
                    memory_text=memory_text,
                )
                model_result = await execute_model_step(
                    kernel=self._kernel,
                    run_id=active_run_id,
                    tenant=tenant,
                    model=model,
                    messages=context_messages,
                    output_schema=output_schema,
                    step_key=f"turn_{iteration}_model",
                    visible_tool_names=visible_tool_names,
                    replay_policy=self._replay_policy,
                    context_version=self._context_version(system_prompt=system_prompt),
                )
                active_run_id = model_result.run_id
                await self._emit_run_summary(
                    run_id=active_run_id,
                    tenant=tenant,
                    summary_type="agent_model_step",
                    step_key=f"turn_{iteration}_model",
                    payload={
                        "iteration": iteration,
                        "replayed": model_result.replayed,
                        "replayed_with_drift": model_result.replayed_with_drift,
                        "tool_calls": [
                            tool_call.tool_name for tool_call in model_result.tool_calls
                        ],
                        "forked_from_run_id": model_result.forked_from_run_id,
                    },
                )
            except BudgetExceededError as exc:
                raise AgentRunFailed(
                    status="failed_budget_exceeded",
                    message=(
                        "AutonomousAgent stopped because the tenant budget was exhausted: "
                        f"{exc}"
                    ),
                ) from exc
            if not model_result.tool_calls:
                if self._auto_reflect:
                    try:
                        await self._run_reflection(
                            run_id=active_run_id,
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
                    except BudgetExceededError:
                        # Reflection is optional best-effort behavior.
                        pass
                return model_result.output

            assistant_tool_calls = _to_assistant_tool_calls(model_result.tool_calls)
            short_term_messages.append(
                ChatMessage(role="assistant", content="", tool_calls=assistant_tool_calls)
            )

            for index, tool_call in enumerate(model_result.tool_calls, start=1):
                self._ensure_tool_is_loaded(
                    tool_name=tool_call.tool_name,
                    visible_tool_names=visible_tool_names,
                )
                tool_call_id = _require_tool_call_id(tool_call)
                tool_result = await self._kernel.step_tool(
                    run_id=active_run_id,
                    tenant=tenant,
                    tool_name=tool_call.tool_name,
                    arguments=model_from_tool_arguments_json(tool_call.arguments_json),
                    step_key=f"turn_{iteration}_tool_{index}_{tool_call.tool_name}",
                )
                await self._emit_run_summary(
                    run_id=active_run_id,
                    tenant=tenant,
                    summary_type="agent_tool_step",
                    step_key=f"turn_{iteration}_tool_{index}_{tool_call.tool_name}",
                    payload={
                        "iteration": iteration,
                        "tool_name": tool_call.tool_name,
                        "replayed": tool_result.replayed,
                        "result_json": tool_result.result_json,
                    },
                )
                short_term_messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=tool_call_id,
                        name=tool_call.tool_name,
                        content=tool_result.result_json,
                    )
                )
                if tool_call.tool_name == self.LOAD_SKILL_NAME:
                    maybe_skill = extract_loaded_skill_name(tool_result.result_json)
                    if maybe_skill is not None:
                        loaded_skills.add(maybe_skill)

        raise MaxIterationsExceeded(
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
            replay_policy=self._replay_policy,
            context_version=self._context_version(
                system_prompt=(
                    "Extract reusable learnings from this completed run and return "
                    "durable rules."
                )
            ),
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

        window_hash = self._compaction_window_hash(
            window=window,
            strategy=strategy,
        )
        cached_artifact = await self._load_compaction_artifact(run_id=run_id)
        if (
            cached_artifact is not None
            and cached_artifact.window_hash == window_hash
            and cached_artifact.strategy_version == strategy.version
            and cached_artifact.summarize_with_model == strategy.summarize_with_model
        ):
            compacted_history = [
                ChatMessage(
                    role="system",
                    content=f"Past Events Summary: {cached_artifact.summary}",
                )
            ]
            compacted_history.extend(recent_messages)
            return tuple(compacted_history)

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
            replay_policy=self._replay_policy,
            context_version=self._context_version(
                system_prompt=summary_input[0].content,
            ),
        )
        summary_text = compact_result.output.summary
        await self._emit_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type=self.COMPACTION_ARTIFACT_SUMMARY_TYPE,
            step_key=f"turn_{iteration}_compact_artifact",
            payload={
                "window_hash": window_hash,
                "summary_hash": hashlib.sha256(summary_text.encode("utf-8")).hexdigest(),
                "summary": summary_text,
                "strategy_version": strategy.version,
                "summarize_with_model": strategy.summarize_with_model,
                "window_message_count": len(window),
            },
        )
        compacted_history = [
            ChatMessage(
                role="system",
                content=f"Past Events Summary: {summary_text}",
            )
        ]
        compacted_history.extend(recent_messages)
        return tuple(compacted_history)

    async def _load_compaction_artifact(self, *, run_id: str) -> _CompactionArtifact | None:
        summary = await self._kernel.get_latest_run_summary(
            run_id=run_id,
            summary_type=self.COMPACTION_ARTIFACT_SUMMARY_TYPE,
        )
        if summary is None:
            return None
        try:
            payload: object = json.loads(summary.summary_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return _CompactionArtifact.model_validate(payload)
        except Exception:
            return None

    def _compaction_window_hash(
        self,
        *,
        window: tuple[ChatMessage, ...],
        strategy: CompactionStrategy,
    ) -> str:
        payload = {
            "strategy_version": strategy.version,
            "summarize_with_model": strategy.summarize_with_model,
            "window": [message.model_dump(mode="json") for message in window],
        }
        return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()

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

    def _context_version(self, *, system_prompt: str) -> ContextVersion:
        compaction = self._compaction
        compaction_version = compaction.version if compaction is not None else None
        return ContextVersion(
            system_prompt_hash=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            context_builder_version=self._context_builder.version,
            compaction_version=compaction_version,
        )

    async def _emit_run_summary(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        summary_type: str,
        step_key: str,
        payload: dict[str, object],
    ) -> None:
        await self._kernel.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type=summary_type,
            summary_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            step_key=step_key,
        )


def _serialize_tool_calls(tool_calls: tuple[ToolCall, ...]) -> str:
    return json.dumps(
        [
            {
                "tool_name": tool_call.tool_name,
                "arguments_json": tool_call.arguments_json,
                "tool_call_id": tool_call.tool_call_id,
            }
            for tool_call in tool_calls
        ]
    )


def _require_tool_call_id(tool_call: ToolCall) -> str:
    if tool_call.tool_call_id is None or tool_call.tool_call_id == "":
        raise ValueError(
            "Model tool call is missing tool_call_id. "
            "Tool protocol requires a stable tool_call_id for each call."
        )
    return tool_call.tool_call_id


def _to_assistant_tool_calls(tool_calls: tuple[ToolCall, ...]) -> list[ToolCallMessage]:
    return [
        ToolCallMessage(
            id=_require_tool_call_id(tool_call),
            function=ToolFunctionCall(
                name=tool_call.tool_name,
                arguments=tool_call.arguments_json,
            ),
        )
        for tool_call in tool_calls
    ]


__all__ = ["AgentRunFailed", "AutonomousAgent", "MaxIterationsExceeded"]
