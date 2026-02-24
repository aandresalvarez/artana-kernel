from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    create_model,
)

from artana.json_utils import canonical_json_dumps, sha256_hex
from artana.ports.model import ToolDefinition
from artana.ports.tool_annotations import strictify_annotation

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
ToolRiskLevel = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    name: str
    requires_capability: str | None
    function: ToolCallable
    description: str
    arguments_schema_json: str
    arguments_model: type[BaseModel]
    accepts_artana_context: bool
    tool_version: str
    schema_version: str
    schema_hash: str
    risk_level: ToolRiskLevel
    sandbox_profile: str | None = None


class ToolPort(Protocol):
    def register(
        self,
        function: ToolCallable,
        requires_capability: str | None = None,
        tool_version: str = "1.0.0",
        schema_version: str = "1",
        risk_level: ToolRiskLevel = "medium",
        sandbox_profile: str | None = None,
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
        self,
        function: ToolCallable,
        requires_capability: str | None = None,
        tool_version: str = "1.0.0",
        schema_version: str = "1",
        risk_level: ToolRiskLevel = "medium",
        sandbox_profile: str | None = None,
    ) -> None:
        signature = inspect.signature(function)
        resolved_hints = get_type_hints(function)
        model_fields: dict[str, object] = {}
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
            annotation = strictify_annotation(raw_annotation)
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
        )  # type: ignore[call-overload]  # Pydantic stubs over-constrain dynamic field definitions.
        schema = arguments_model.model_json_schema()
        schema_json = canonical_json_dumps(schema)
        schema_hash = sha256_hex(schema_json)
        description = inspect.getdoc(function) or ""
        self._tools[function.__name__] = RegisteredTool(
            name=function.__name__,
            requires_capability=requires_capability,
            function=function,
            description=description,
            arguments_schema_json=schema_json,
            arguments_model=arguments_model,
            accepts_artana_context=accepts_artana_context,
            tool_version=tool_version,
            schema_version=schema_version,
            schema_hash=schema_hash,
            risk_level=risk_level,
            sandbox_profile=sandbox_profile,
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
                tool_version=tool.tool_version,
                schema_version=tool.schema_version,
                schema_hash=tool.schema_hash,
                risk_level=tool.risk_level,
                sandbox_profile=tool.sandbox_profile,
            )
            for tool in self.list_for_capabilities(capabilities)
        ]

    def to_all_tool_definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                arguments_schema_json=tool.arguments_schema_json,
                tool_version=tool.tool_version,
                schema_version=tool.schema_version,
                schema_hash=tool.schema_hash,
                risk_level=tool.risk_level,
                sandbox_profile=tool.sandbox_profile,
            )
            for tool in self._tools.values()
        ]

    def capability_map(self) -> dict[str, str | None]:
        return {
            tool_name: tool.requires_capability for tool_name, tool in self._tools.items()
        }
