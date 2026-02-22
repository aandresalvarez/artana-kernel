from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from artana.ports.model import ToolDefinition

ToolExecutionOutcome = Literal[
    "success",
    "transient_error",
    "permanent_error",
    "unknown_outcome",
]


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    run_id: str
    tenant_id: str
    idempotency_key: str
    request_event_id: str | None
    tool_version: str
    schema_version: str


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    outcome: ToolExecutionOutcome
    result_json: str
    received_idempotency_key: str | None = None
    effect_id: str | None = None
    request_id: str | None = None
    error_message: str | None = None


class ToolTransientError(RuntimeError):
    pass


class ToolPermanentError(RuntimeError):
    pass


class ToolUnknownOutcomeError(RuntimeError):
    pass


ToolReturnValue = str | ToolExecutionResult
ToolCallable = Callable[..., Awaitable[ToolReturnValue]]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    name: str
    requires_capability: str | None
    function: ToolCallable
    description: str
    arguments_schema_json: str
    accepts_artana_context: bool


class ToolPort(Protocol):
    def register(
        self, function: ToolCallable, requires_capability: str | None = None
    ) -> None:
        ...

    def list_for_capabilities(self, capabilities: frozenset[str]) -> list[RegisteredTool]:
        ...

    async def call(
        self,
        tool_name: str,
        arguments_json: str,
        *,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        ...

    def to_tool_definitions(self, capabilities: frozenset[str]) -> list[ToolDefinition]:
        ...

    def to_all_tool_definitions(self) -> list[ToolDefinition]:
        ...

    def capability_map(self) -> dict[str, str | None]:
        ...


class LocalToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self, function: ToolCallable, requires_capability: str | None = None
    ) -> None:
        signature = inspect.signature(function)
        required: list[str] = []
        properties: dict[str, dict[str, str]] = {}
        accepts_artana_context = False
        for parameter in signature.parameters.values():
            if parameter.name == "artana_context":
                accepts_artana_context = True
                continue
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            properties[parameter.name] = {"type": "string"}
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        description = inspect.getdoc(function) or ""
        self._tools[function.__name__] = RegisteredTool(
            name=function.__name__,
            requires_capability=requires_capability,
            function=function,
            description=description,
            arguments_schema_json=json.dumps(schema),
            accepts_artana_context=accepts_artana_context,
        )

    def list_for_capabilities(self, capabilities: frozenset[str]) -> list[RegisteredTool]:
        return [
            tool
            for tool in self._tools.values()
            if tool.requires_capability is None or tool.requires_capability in capabilities
        ]

    async def call(
        self,
        tool_name: str,
        arguments_json: str,
        *,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool {tool_name!r} is not registered.")

        parsed_arguments = json.loads(arguments_json)
        if not isinstance(parsed_arguments, dict):
            raise ValueError(
                f"Tool arguments for {tool_name!r} must be a JSON object."
            )

        kwargs: dict[str, object] = {}
        for key, value in parsed_arguments.items():
            if not isinstance(key, str):
                raise ValueError("Tool argument keys must be strings.")
            kwargs[key] = value
        if tool.accepts_artana_context:
            kwargs["artana_context"] = context

        try:
            raw_result = await tool.function(**kwargs)
        except ToolTransientError as exc:
            return ToolExecutionResult(
                outcome="transient_error",
                result_json="",
                received_idempotency_key=context.idempotency_key,
                error_message=str(exc),
            )
        except ToolPermanentError as exc:
            return ToolExecutionResult(
                outcome="permanent_error",
                result_json="",
                received_idempotency_key=context.idempotency_key,
                error_message=str(exc),
            )
        except ToolUnknownOutcomeError:
            raise
        except Exception as exc:
            raise ToolUnknownOutcomeError(str(exc)) from exc

        if isinstance(raw_result, ToolExecutionResult):
            if raw_result.received_idempotency_key is not None:
                return raw_result
            return replace(raw_result, received_idempotency_key=context.idempotency_key)
        if isinstance(raw_result, str):
            return ToolExecutionResult(
                outcome="success",
                result_json=raw_result,
                received_idempotency_key=context.idempotency_key,
            )
        raise ToolPermanentError(
            f"Tool {tool_name!r} returned unsupported type {type(raw_result)!r}."
        )

    def to_tool_definitions(self, capabilities: frozenset[str]) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                arguments_schema_json=tool.arguments_schema_json,
            )
            for tool in self.list_for_capabilities(capabilities)
        ]

    def to_all_tool_definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                arguments_schema_json=tool.arguments_schema_json,
            )
            for tool in self._tools.values()
        ]

    def capability_map(self) -> dict[str, str | None]:
        return {
            tool_name: tool.requires_capability for tool_name, tool in self._tools.items()
        }
