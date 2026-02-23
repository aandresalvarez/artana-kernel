from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum
from types import UnionType
from typing import Any, Literal, Protocol, Union, get_args, get_origin, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
    create_model,
)

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
    tenant_capabilities: frozenset[str] = frozenset()
    tenant_budget_usd_limit: float | None = None


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
    arguments_model: type[BaseModel]
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
        resolved_hints = get_type_hints(function)
        # Pydantic `create_model` typing requires `Any` for dynamic field kwargs.
        model_fields: dict[str, Any] = {}
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
            raw_annotation = resolved_hints.get(parameter.name, parameter.annotation)
            annotation = _strictify_annotation(raw_annotation)
            default: object
            if parameter.default is inspect.Parameter.empty:
                default = ...
            else:
                default = parameter.default
            model_fields[parameter.name] = (annotation, default)

        arguments_model = create_model(
            f"{function.__name__}Arguments",
            __config__=ConfigDict(extra="forbid"),
            **model_fields,
        )
        schema = arguments_model.model_json_schema()
        description = inspect.getdoc(function) or ""
        self._tools[function.__name__] = RegisteredTool(
            name=function.__name__,
            requires_capability=requires_capability,
            function=function,
            description=description,
            arguments_schema_json=json.dumps(schema),
            arguments_model=arguments_model,
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
        if any(not isinstance(key, str) for key in parsed_arguments.keys()):
            raise ValueError("Tool argument keys must be strings.")
        try:
            validated = tool.arguments_model.model_validate(parsed_arguments)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid arguments for tool {tool_name!r}: {exc}"
            ) from exc

        kwargs = validated.model_dump(mode="python")
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


def _strictify_annotation(annotation: object) -> object:
    if annotation is inspect._empty:
        return str
    if annotation is int:
        return StrictInt
    if annotation is float:
        return StrictFloat
    if annotation is bool:
        return StrictBool
    if annotation is Decimal:
        return Decimal
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return annotation

        if origin in (UnionType, Union):
            raw_args = get_args(annotation)
            if len(raw_args) == 2 and type(None) in raw_args:
                non_none = raw_args[0] if raw_args[1] is type(None) else raw_args[1]
                strict_non_none = _strictify_annotation(non_none)
                if isinstance(strict_non_none, type):
                    return strict_non_none | None
                return annotation
        return annotation

    return annotation
