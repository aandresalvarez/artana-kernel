# 📂 Project File Contents

- 📁 **examples**
  - [`01_durable_chat_replay.py`](#examples-01_durable_chat_replaypy)
  - [`02_real_litellm_chat.py`](#examples-02_real_litellm_chatpy)
  - [`03_fact_extraction_triplets.py`](#examples-03_fact_extraction_tripletspy)
  - [`README.md`](#examples-readmemd)
  - [`golden_example.py`](#examples-golden_examplepy)
  - 📁 **artana**
    - [`__init__.py`](#src-artana-__init__py)
    - [`agent.py`](#src-artana-agentpy)
    - [`events.py`](#src-artana-eventspy)
    - [`kernel.py`](#src-artana-kernelpy)
    - [`models.py`](#src-artana-modelspy)
    - 📁 **_kernel**
      - [`__init__.py`](#src-artana-_kernel-__init__py)
      - [`core.py`](#src-artana-_kernel-corepy)
      - [`model_cycle.py`](#src-artana-_kernel-model_cyclepy)
      - [`policies.py`](#src-artana-_kernel-policiespy)
      - [`replay.py`](#src-artana-_kernel-replaypy)
      - [`tool_cycle.py`](#src-artana-_kernel-tool_cyclepy)
      - [`tool_execution.py`](#src-artana-_kernel-tool_executionpy)
      - [`tool_state.py`](#src-artana-_kernel-tool_statepy)
      - [`types.py`](#src-artana-_kernel-typespy)
      - [`workflow_runtime.py`](#src-artana-_kernel-workflow_runtimepy)
    - 📁 **middleware**
      - [`__init__.py`](#src-artana-middleware-__init__py)
      - [`base.py`](#src-artana-middleware-basepy)
      - [`capability_guard.py`](#src-artana-middleware-capability_guardpy)
      - [`order.py`](#src-artana-middleware-orderpy)
      - [`pii_scrubber.py`](#src-artana-middleware-pii_scrubberpy)
      - [`quota.py`](#src-artana-middleware-quotapy)
    - 📁 **ports**
      - [`__init__.py`](#src-artana-ports-__init__py)
      - [`model.py`](#src-artana-ports-modelpy)
      - [`model_adapter.py`](#src-artana-ports-model_adapterpy)
      - [`model_types.py`](#src-artana-ports-model_typespy)
      - [`tool.py`](#src-artana-ports-toolpy)
    - 📁 **store**
      - [`__init__.py`](#src-artana-store-__init__py)
      - [`base.py`](#src-artana-store-basepy)
      - [`sqlite.py`](#src-artana-store-sqlitepy)

---

# Target Folder: src

## Folder: src

## Folder: artana

### File: `src/artana/__init__.py`
<a name="src-artana-__init__py"></a>
```python
from artana.agent import ChatClient
from artana.kernel import (
    ArtanaKernel,
    JsonValue,
    KernelPolicy,
    ModelInput,
    PauseTicket,
    RunHandle,
    RunRef,
    StepModelResult,
    StepSerde,
    StepToolResult,
    WorkflowContext,
    WorkflowRunResult,
    json_step_serde,
    pydantic_step_serde,
)
from artana.models import TenantContext
from artana.store import SQLiteStore

__all__ = [
    "ArtanaKernel",
    "ChatClient",
    "JsonValue",
    "KernelPolicy",
    "ModelInput",
    "PauseTicket",
    "RunHandle",
    "RunRef",
    "SQLiteStore",
    "StepModelResult",
    "StepToolResult",
    "StepSerde",
    "TenantContext",
    "WorkflowContext",
    "WorkflowRunResult",
    "json_step_serde",
    "pydantic_step_serde",
]

```

### File: `src/artana/agent.py`
<a name="src-artana-agentpy"></a>
```python
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from artana.kernel import ArtanaKernel, ModelInput, StepModelResult
from artana.models import TenantContext

OutputT = TypeVar("OutputT", bound=BaseModel)


class ChatClient:
    def __init__(self, *, kernel: ArtanaKernel) -> None:
        self._kernel = kernel

    async def chat(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        prompt: str,
        output_schema: type[OutputT],
        step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        try:
            await self._kernel.load_run(run_id=run_id)
        except ValueError:
            await self._kernel.start_run(tenant=tenant, run_id=run_id)
        return await self._kernel.step_model(
            run_id=run_id,
            tenant=tenant,
            model=model,
            input=ModelInput.from_prompt(prompt),
            output_schema=output_schema,
            step_key=step_key,
        )

```

### File: `src/artana/events.py`
<a name="src-artana-eventspy"></a>
```python
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    RESUME_REQUESTED = "resume_requested"
    MODEL_REQUESTED = "model_requested"
    MODEL_COMPLETED = "model_completed"
    TOOL_REQUESTED = "tool_requested"
    TOOL_COMPLETED = "tool_completed"
    PAUSE_REQUESTED = "pause_requested"
    WORKFLOW_STEP_REQUESTED = "workflow_step_requested"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ModelRequestedPayload(BaseModel):
    kind: Literal["model_requested"] = "model_requested"
    model: str
    prompt: str
    messages: list[ChatMessage]
    allowed_tools: list[str] = Field(default_factory=list)
    step_key: str | None = None


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments_json: str


class ModelCompletedPayload(BaseModel):
    kind: Literal["model_completed"] = "model_completed"
    model: str
    output_json: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class ToolRequestedPayload(BaseModel):
    kind: Literal["tool_requested"] = "tool_requested"
    tool_name: str
    arguments_json: str
    idempotency_key: str
    tool_version: str = "1.0.0"
    schema_version: str = "1"
    step_key: str | None = None


class ToolCompletedPayload(BaseModel):
    kind: Literal["tool_completed"] = "tool_completed"
    tool_name: str
    result_json: str
    outcome: Literal[
        "success",
        "transient_error",
        "permanent_error",
        "unknown_outcome",
    ] = "success"
    received_idempotency_key: str | None = None
    effect_id: str | None = None
    request_id: str | None = None
    error_message: str | None = None


class PauseRequestedPayload(BaseModel):
    kind: Literal["pause_requested"] = "pause_requested"
    reason: str
    context_json: str | None = None
    step_key: str | None = None


class RunStartedPayload(BaseModel):
    kind: Literal["run_started"] = "run_started"


class ResumeRequestedPayload(BaseModel):
    kind: Literal["resume_requested"] = "resume_requested"
    human_input_json: str | None = None


class WorkflowStepRequestedPayload(BaseModel):
    kind: Literal["workflow_step_requested"] = "workflow_step_requested"
    step_index: int = Field(ge=0)
    step_name: str


class WorkflowStepCompletedPayload(BaseModel):
    kind: Literal["workflow_step_completed"] = "workflow_step_completed"
    step_index: int = Field(ge=0)
    step_name: str
    result_json: str


EventPayload = (
    RunStartedPayload
    | ResumeRequestedPayload
    |
    ModelRequestedPayload
    | ModelCompletedPayload
    | ToolRequestedPayload
    | ToolCompletedPayload
    | PauseRequestedPayload
    | WorkflowStepRequestedPayload
    | WorkflowStepCompletedPayload
)


class KernelEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    run_id: str
    tenant_id: str
    seq: int = Field(ge=1)
    event_type: EventType
    prev_event_hash: str | None = None
    event_hash: str
    timestamp: datetime = Field(default_factory=utc_now)
    payload: EventPayload

    @model_validator(mode="after")
    def payload_matches_event_type(self) -> "KernelEvent":
        if self.event_type.value != self.payload.kind:
            raise ValueError(
                "event_type="
                f"{self.event_type.value} does not match payload kind={self.payload.kind}"
            )
        expected_hash = compute_event_hash(
            event_id=self.event_id,
            run_id=self.run_id,
            tenant_id=self.tenant_id,
            seq=self.seq,
            event_type=self.event_type,
            prev_event_hash=self.prev_event_hash,
            timestamp=self.timestamp,
            payload=self.payload,
        )
        if expected_hash != self.event_hash:
            raise ValueError(
                "event_hash mismatch for "
                f"seq={self.seq}. expected={expected_hash}, got={self.event_hash}"
            )
        return self


def payload_to_canonical_json(payload: EventPayload) -> str:
    return json.dumps(payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def compute_event_hash(
    *,
    event_id: str,
    run_id: str,
    tenant_id: str,
    seq: int,
    event_type: EventType,
    prev_event_hash: str | None,
    timestamp: datetime,
    payload: EventPayload,
) -> str:
    joined = "|".join(
        (
            event_id,
            run_id,
            tenant_id,
            str(seq),
            event_type.value,
            prev_event_hash or "",
            timestamp.isoformat(),
            payload_to_canonical_json(payload),
        )
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

```

### File: `src/artana/kernel.py`
<a name="src-artana-kernelpy"></a>
```python
from artana._kernel.core import ArtanaKernel
from artana._kernel.types import (
    CapabilityDeniedError,
    KernelPolicy,
    ModelInput,
    PauseTicket,
    ReplayConsistencyError,
    RunHandle,
    RunRef,
    StepModelResult,
    StepToolResult,
    ToolExecutionFailedError,
)
from artana._kernel.workflow_runtime import (
    JsonValue,
    StepSerde,
    WorkflowContext,
    WorkflowRunResult,
    json_step_serde,
    pydantic_step_serde,
)

__all__ = [
    "ArtanaKernel",
    "CapabilityDeniedError",
    "KernelPolicy",
    "ModelInput",
    "JsonValue",
    "PauseTicket",
    "ReplayConsistencyError",
    "RunHandle",
    "RunRef",
    "StepModelResult",
    "StepToolResult",
    "StepSerde",
    "ToolExecutionFailedError",
    "WorkflowContext",
    "WorkflowRunResult",
    "json_step_serde",
    "pydantic_step_serde",
]

```

### File: `src/artana/models.py`
<a name="src-artana-modelspy"></a>
```python
from __future__ import annotations

from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    tenant_id: str = Field(min_length=1)
    capabilities: frozenset[str] = Field(default_factory=frozenset)
    budget_usd_limit: float = Field(gt=0.0)


```

## Folder: artana/middleware

### File: `src/artana/middleware/__init__.py`
<a name="src-artana-middleware-__init__py"></a>
```python
from artana.middleware.base import BudgetExceededError, KernelMiddleware, ModelInvocation
from artana.middleware.capability_guard import CapabilityGuardMiddleware
from artana.middleware.order import order_middleware
from artana.middleware.pii_scrubber import PIIScrubberMiddleware
from artana.middleware.quota import QuotaMiddleware

__all__ = [
    "BudgetExceededError",
    "CapabilityGuardMiddleware",
    "KernelMiddleware",
    "ModelInvocation",
    "PIIScrubberMiddleware",
    "QuotaMiddleware",
    "order_middleware",
]

```

### File: `src/artana/middleware/base.py`
<a name="src-artana-middleware-basepy"></a>
```python
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from artana.events import ChatMessage
from artana.models import TenantContext
from artana.ports.model import ModelUsage, ToolDefinition


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelInvocation:
    run_id: str
    tenant: TenantContext
    model: str
    prompt: str
    messages: tuple[ChatMessage, ...]
    allowed_tools: tuple[ToolDefinition, ...]
    tool_capability_by_name: dict[str, str | None]

    def with_updates(
        self,
        *,
        prompt: str | None = None,
        messages: tuple[ChatMessage, ...] | None = None,
        allowed_tools: tuple[ToolDefinition, ...] | None = None,
    ) -> "ModelInvocation":
        return replace(
            self,
            prompt=self.prompt if prompt is None else prompt,
            messages=self.messages if messages is None else messages,
            allowed_tools=self.allowed_tools if allowed_tools is None else allowed_tools,
        )


class KernelMiddleware(Protocol):
    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        ...

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        ...

    async def after_model(
        self, *, run_id: str, tenant: TenantContext, usage: ModelUsage
    ) -> None:
        ...

```

### File: `src/artana/middleware/capability_guard.py`
<a name="src-artana-middleware-capability_guardpy"></a>
```python
from __future__ import annotations

from artana.middleware.base import ModelInvocation
from artana.models import TenantContext
from artana.ports.model import ModelUsage


class CapabilityGuardMiddleware:
    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        filtered_tools = tuple(
            tool
            for tool in invocation.allowed_tools
            if self._is_allowed(
                capability=invocation.tool_capability_by_name.get(tool.name),
                tenant=invocation.tenant,
            )
        )
        return invocation.with_updates(allowed_tools=filtered_tools)

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        return None

    async def after_model(
        self, *, run_id: str, tenant: TenantContext, usage: ModelUsage
    ) -> None:
        return None

    def _is_allowed(self, *, capability: str | None, tenant: TenantContext) -> bool:
        if capability is None:
            return True
        return capability in tenant.capabilities


```

### File: `src/artana/middleware/order.py`
<a name="src-artana-middleware-orderpy"></a>
```python
from __future__ import annotations

from collections.abc import Sequence

from artana.middleware.base import KernelMiddleware
from artana.middleware.capability_guard import CapabilityGuardMiddleware
from artana.middleware.pii_scrubber import PIIScrubberMiddleware
from artana.middleware.quota import QuotaMiddleware


def order_middleware(middleware: Sequence[KernelMiddleware]) -> tuple[KernelMiddleware, ...]:
    prioritized: list[tuple[int, int, KernelMiddleware]] = []
    for index, middleware_item in enumerate(middleware):
        prioritized.append((_priority_for(middleware_item), index, middleware_item))
    prioritized.sort(key=lambda row: (row[0], row[1]))
    return tuple(middleware_item for _, _, middleware_item in prioritized)


def _priority_for(middleware_item: KernelMiddleware) -> int:
    if isinstance(middleware_item, PIIScrubberMiddleware):
        return 0
    if isinstance(middleware_item, QuotaMiddleware):
        return 1
    if isinstance(middleware_item, CapabilityGuardMiddleware):
        return 2
    return 3


```

### File: `src/artana/middleware/pii_scrubber.py`
<a name="src-artana-middleware-pii_scrubberpy"></a>
```python
from __future__ import annotations

import re

from artana.events import ChatMessage
from artana.middleware.base import ModelInvocation
from artana.models import TenantContext
from artana.ports.model import ModelUsage


class PIIScrubberMiddleware:
    def __init__(self) -> None:
        self._patterns: tuple[tuple[re.Pattern[str], str], ...] = (
            (
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                "[REDACTED_EMAIL]",
            ),
            (
                re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
                "[REDACTED_PHONE]",
            ),
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        )

    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        redacted_prompt = self._redact_text(invocation.prompt)
        redacted_messages = tuple(
            ChatMessage(role=message.role, content=self._redact_text(message.content))
            for message in invocation.messages
        )
        return invocation.with_updates(
            prompt=redacted_prompt,
            messages=redacted_messages,
        )

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        return None

    async def after_model(
        self, *, run_id: str, tenant: TenantContext, usage: ModelUsage
    ) -> None:
        return None

    def _redact_text(self, input_text: str) -> str:
        result = input_text
        for pattern, replacement in self._patterns:
            result = pattern.sub(replacement, result)
        return result


```

### File: `src/artana/middleware/quota.py`
<a name="src-artana-middleware-quotapy"></a>
```python
from __future__ import annotations

from artana.events import EventType
from artana.middleware.base import BudgetExceededError, ModelInvocation
from artana.models import TenantContext
from artana.ports.model import ModelUsage
from artana.store.base import EventStore


class QuotaMiddleware:
    def __init__(self, store: EventStore | None = None) -> None:
        self._store = store
        self._spent_usd_by_run: dict[str, float] = {}

    def bind_store(self, store: EventStore) -> None:
        self._store = store

    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        return invocation

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        spent = await self._load_spent_for_run(run_id=run_id)
        if spent >= tenant.budget_usd_limit:
            raise BudgetExceededError(
                "Run "
                f"{run_id!r} budget exhausted. "
                f"limit={tenant.budget_usd_limit:.6f}, spent={spent:.6f}"
            )

    async def after_model(
        self, *, run_id: str, tenant: TenantContext, usage: ModelUsage
    ) -> None:
        if self._store is None:
            spent_before = self._spent_usd_by_run.get(run_id, 0.0)
            spent_after = spent_before + usage.cost_usd
            self._spent_usd_by_run[run_id] = spent_after
        else:
            spent_after = await self._load_spent_from_store(run_id=run_id)
        if spent_after > tenant.budget_usd_limit:
            raise BudgetExceededError(
                "Run "
                f"{run_id!r} exceeded budget. "
                f"limit={tenant.budget_usd_limit:.6f}, spent={spent_after:.6f}"
            )

    async def _load_spent_for_run(self, *, run_id: str) -> float:
        if self._store is None:
            return self._spent_usd_by_run.get(run_id, 0.0)
        return await self._load_spent_from_store(run_id=run_id)

    async def _load_spent_from_store(self, *, run_id: str) -> float:
        if self._store is None:
            raise RuntimeError("QuotaMiddleware store is not configured.")

        events = await self._store.get_events_for_run(run_id)
        spent = 0.0
        for event in events:
            if event.event_type != EventType.MODEL_COMPLETED:
                continue
            payload = event.payload
            if payload.kind != "model_completed":
                raise RuntimeError(
                    f"Invalid event payload kind {payload.kind!r} for model_completed event."
                )
            spent += payload.cost_usd
        return spent

```

## Folder: artana/_kernel

### File: `src/artana/_kernel/__init__.py`
<a name="src-artana-_kernel-__init__py"></a>
```python
from artana._kernel.core import ArtanaKernel
from artana._kernel.types import (
    CapabilityDeniedError,
    KernelPolicy,
    ModelInput,
    PauseTicket,
    ReplayConsistencyError,
    RunHandle,
    RunRef,
    StepModelResult,
    StepToolResult,
    ToolExecutionFailedError,
)
from artana._kernel.workflow_runtime import (
    JsonValue,
    StepSerde,
    WorkflowContext,
    WorkflowRunResult,
    json_step_serde,
    pydantic_step_serde,
)

__all__ = [
    "ArtanaKernel",
    "CapabilityDeniedError",
    "KernelPolicy",
    "ModelInput",
    "PauseTicket",
    "ReplayConsistencyError",
    "RunHandle",
    "RunRef",
    "StepModelResult",
    "StepToolResult",
    "ToolExecutionFailedError",
    "JsonValue",
    "StepSerde",
    "WorkflowContext",
    "WorkflowRunResult",
    "json_step_serde",
    "pydantic_step_serde",
]

```

### File: `src/artana/_kernel/core.py`
<a name="src-artana-_kernel-corepy"></a>
```python
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel

from artana._kernel.model_cycle import get_or_execute_model_step
from artana._kernel.policies import apply_prepare_model_middleware, enforce_capability_scope
from artana._kernel.replay import validate_tenant_for_run
from artana._kernel.tool_cycle import (
    execute_tool_step_with_replay,
    reconcile_tool_with_replay,
)
from artana._kernel.types import (
    KernelPolicy,
    ModelInput,
    OutputT,
    PauseTicket,
    RunHandle,
    RunRef,
    StepModelResult,
    StepToolResult,
    ToolCallable,
)
from artana._kernel.workflow_runtime import (
    WorkflowContext,
    WorkflowRunResult,
    run_workflow,
)
from artana.events import (
    ChatMessage,
    EventType,
    PauseRequestedPayload,
    ResumeRequestedPayload,
    RunStartedPayload,
)
from artana.middleware import order_middleware
from artana.middleware.base import KernelMiddleware, ModelInvocation
from artana.middleware.capability_guard import CapabilityGuardMiddleware
from artana.middleware.pii_scrubber import PIIScrubberMiddleware
from artana.middleware.quota import QuotaMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelPort
from artana.ports.tool import LocalToolRegistry, ToolPort
from artana.store.base import EventStore

WorkflowOutputT = TypeVar("WorkflowOutputT")


@runtime_checkable
class _StoreBindableMiddleware(Protocol):
    def bind_store(self, store: EventStore) -> None:
        ...


class ArtanaKernel:
    def __init__(
        self,
        *,
        store: EventStore,
        model_port: ModelPort,
        tool_port: ToolPort | None = None,
        middleware: Sequence[KernelMiddleware] | None = None,
        policy: KernelPolicy | None = None,
    ) -> None:
        self._store = store
        self._model_port = model_port
        self._tool_port = tool_port if tool_port is not None else LocalToolRegistry()
        self._policy = policy if policy is not None else KernelPolicy()
        self._middleware = order_middleware(tuple(middleware or ()))
        self._validate_policy_requirements()
        for middleware_item in self._middleware:
            if isinstance(middleware_item, _StoreBindableMiddleware):
                middleware_item.bind_store(store)

    @staticmethod
    def default_middleware_stack(
        *,
        pii: bool = True,
        quota: bool = True,
        capabilities: bool = True,
    ) -> tuple[KernelMiddleware, ...]:
        stack: list[KernelMiddleware] = []
        if pii:
            stack.append(PIIScrubberMiddleware())
        if quota:
            stack.append(QuotaMiddleware())
        if capabilities:
            stack.append(CapabilityGuardMiddleware())
        return order_middleware(tuple(stack))

    async def start_run(
        self,
        *,
        tenant: TenantContext,
        run_id: str | None = None,
    ) -> RunRef:
        run_id_value = run_id
        if run_id_value is not None:
            existing = await self._store.get_events_for_run(run_id_value)
            if existing:
                raise ValueError(
                    f"run_id={run_id_value!r} already exists; provide a different run_id."
                )
        else:
            for _ in range(5):
                generated = uuid4().hex
                if not await self._store.get_events_for_run(generated):
                    run_id_value = generated
                    break
            if run_id_value is None:
                raise RuntimeError(
                    "Failed to allocate a unique run_id after multiple attempts."
                )

        event = await self._store.append_event(
            run_id=run_id_value,
            tenant_id=tenant.tenant_id,
            event_type=EventType.RUN_STARTED,
            payload=RunStartedPayload(),
        )
        return RunHandle(run_id=event.run_id, tenant_id=event.tenant_id)

    async def load_run(self, *, run_id: str) -> RunRef:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")
        return RunHandle(run_id=run_id, tenant_id=events[0].tenant_id)

    def tool(
        self, *, requires_capability: str | None = None
    ) -> Callable[[ToolCallable], ToolCallable]:
        def decorator(function: ToolCallable) -> ToolCallable:
            self._tool_port.register(
                function=function,
                requires_capability=requires_capability,
            )
            return function

        return decorator

    async def pause(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        reason: str,
        context: BaseModel | None = None,
        step_key: str | None = None,
    ) -> PauseTicket:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"Cannot pause unknown run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)
        context_json = context.model_dump_json() if context is not None else None
        event = await self._store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.PAUSE_REQUESTED,
            payload=PauseRequestedPayload(
                reason=reason,
                context_json=context_json,
                step_key=step_key,
            ),
        )
        return PauseTicket(
            run_id=event.run_id,
            ticket_id=event.event_id,
            seq=event.seq,
            reason=reason,
        )

    async def step_model(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        input: ModelInput,
        output_schema: type[OutputT],
        step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"No events found for run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)

        prompt, messages = _normalize_model_input(input)
        initial_invocation = ModelInvocation(
            run_id=run_id,
            tenant=tenant,
            model=model,
            prompt=prompt,
            messages=messages,
            allowed_tools=tuple(self._tool_port.to_all_tool_definitions()),
            tool_capability_by_name=self._tool_port.capability_map(),
        )
        prepared_invocation = await apply_prepare_model_middleware(
            self._middleware,
            initial_invocation,
        )
        scoped_invocation = enforce_capability_scope(prepared_invocation)

        model_result = await get_or_execute_model_step(
            store=self._store,
            model_port=self._model_port,
            middleware=self._middleware,
            run_id=run_id,
            prompt=scoped_invocation.prompt,
            messages=scoped_invocation.messages,
            model=scoped_invocation.model,
            tenant=tenant,
            output_schema=output_schema,
            tool_definitions=scoped_invocation.allowed_tools,
            allowed_tool_names=[tool.name for tool in scoped_invocation.allowed_tools],
            events=events,
            step_key=step_key,
        )
        return StepModelResult(
            run_id=run_id,
            seq=model_result.completed_seq,
            output=model_result.output,
            usage=model_result.usage,
            tool_calls=model_result.tool_calls,
            replayed=model_result.replayed,
        )

    async def step_tool(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments: BaseModel,
        step_key: str | None = None,
    ) -> StepToolResult:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"No events found for run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)
        arguments_json = json.dumps(
            arguments.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        result = await execute_tool_step_with_replay(
            store=self._store,
            tool_port=self._tool_port,
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step_key=step_key,
        )
        return StepToolResult(
            run_id=run_id,
            seq=result.seq,
            tool_name=tool_name,
            result_json=result.result_json,
            replayed=result.replayed,
        )

    async def reconcile_tool(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments: BaseModel,
        step_key: str | None = None,
    ) -> str:
        arguments_json = json.dumps(
            arguments.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return await reconcile_tool_with_replay(
            store=self._store,
            tool_port=self._tool_port,
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step_key=step_key,
        )

    async def resume(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        human_input: BaseModel | None = None,
    ) -> RunRef:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")
        validate_tenant_for_run(events=events, tenant=tenant)
        human_input_json = human_input.model_dump_json() if human_input is not None else None
        event = await self._store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.RESUME_REQUESTED,
            payload=ResumeRequestedPayload(human_input_json=human_input_json),
        )
        return RunHandle(run_id=event.run_id, tenant_id=event.tenant_id)

    async def close(self) -> None:
        await self._store.close()

    async def run_workflow(
        self,
        *,
        run_id: str | None,
        tenant: TenantContext,
        workflow: Callable[[WorkflowContext], Awaitable[WorkflowOutputT]],
    ) -> WorkflowRunResult[WorkflowOutputT]:
        return await run_workflow(
            store=self._store,
            pause_api=self,
            run_id=run_id,
            tenant=tenant,
            workflow=workflow,
        )

    def _validate_policy_requirements(self) -> None:
        if self._policy.mode != "enforced":
            return

        required: tuple[type[KernelMiddleware], ...] = (
            PIIScrubberMiddleware,
            QuotaMiddleware,
            CapabilityGuardMiddleware,
        )
        for middleware_type in required:
            if not any(
                isinstance(middleware_item, middleware_type)
                for middleware_item in self._middleware
            ):
                raise ValueError(
                    "KernelPolicy(mode='enforced') requires middleware "
                    f"{middleware_type.__name__}."
                )


def _normalize_model_input(model_input: ModelInput) -> tuple[str, tuple[ChatMessage, ...]]:
    if model_input.kind == "prompt":
        if model_input.prompt is None:
            raise ValueError("ModelInput(kind='prompt') requires prompt.")
        if model_input.messages is None:
            return model_input.prompt, (ChatMessage(role="user", content=model_input.prompt),)
        if len(model_input.messages) == 0:
            raise ValueError("ModelInput(kind='prompt') messages cannot be empty.")
        return model_input.prompt, model_input.messages

    if model_input.messages is None or len(model_input.messages) == 0:
        raise ValueError("ModelInput(kind='messages') requires non-empty messages.")

    prompt = model_input.prompt
    if prompt is None:
        prompt = _derive_prompt_from_messages(model_input.messages)
    return prompt, model_input.messages


def _derive_prompt_from_messages(messages: tuple[ChatMessage, ...]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return "\n".join(f"{message.role}: {message.content}" for message in messages)

```

### File: `src/artana/_kernel/model_cycle.py`
<a name="src-artana-_kernel-model_cyclepy"></a>
```python
from __future__ import annotations

from collections.abc import Sequence

from artana._kernel.replay import (
    ModelStepResult,
    deserialize_model_completed,
    find_matching_model_cycle,
)
from artana._kernel.types import OutputT
from artana.events import (
    ChatMessage,
    EventType,
    KernelEvent,
    ModelCompletedPayload,
    ModelRequestedPayload,
    ToolCallRecord,
)
from artana.middleware.base import KernelMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelPort, ModelRequest, ToolDefinition
from artana.store.base import EventStore


async def get_or_execute_model_step(
    *,
    store: EventStore,
    model_port: ModelPort,
    middleware: Sequence[KernelMiddleware],
    run_id: str,
    prompt: str,
    messages: tuple[ChatMessage, ...],
    model: str,
    tenant: TenantContext,
    output_schema: type[OutputT],
    tool_definitions: Sequence[ToolDefinition],
    allowed_tool_names: list[str],
    events: Sequence[KernelEvent],
    step_key: str | None = None,
) -> ModelStepResult[OutputT]:
    request_event, completed_event = find_matching_model_cycle(
        events=events,
        prompt=prompt,
        messages=messages,
        model=model,
        allowed_tool_names=allowed_tool_names,
        step_key=step_key,
    )
    if completed_event is not None:
        return deserialize_model_completed(
            event=completed_event,
            output_schema=output_schema,
            replayed=True,
        )

    if request_event is None:
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.MODEL_REQUESTED,
            payload=ModelRequestedPayload(
                model=model,
                prompt=prompt,
                messages=list(messages),
                allowed_tools=allowed_tool_names,
                step_key=step_key,
            ),
        )

    for middleware_item in middleware:
        await middleware_item.before_model(run_id=run_id, tenant=tenant)

    result = await model_port.complete(
        ModelRequest(
            run_id=run_id,
            model=model,
            prompt=prompt,
            output_schema=output_schema,
            allowed_tools=tool_definitions,
        )
    )
    completed_event = await store.append_event(
        run_id=run_id,
        tenant_id=tenant.tenant_id,
        event_type=EventType.MODEL_COMPLETED,
        payload=ModelCompletedPayload(
            model=model,
            output_json=result.output.model_dump_json(),
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            cost_usd=result.usage.cost_usd,
            tool_calls=[
                ToolCallRecord(
                    tool_name=tool_call.tool_name,
                    arguments_json=tool_call.arguments_json,
                )
                for tool_call in result.tool_calls
            ],
        ),
    )
    for middleware_item in middleware:
        await middleware_item.after_model(
            run_id=run_id,
            tenant=tenant,
            usage=result.usage,
        )

    return ModelStepResult(
        completed_seq=completed_event.seq,
        output=result.output,
        usage=result.usage,
        tool_calls=result.tool_calls,
        replayed=False,
    )

```

### File: `src/artana/_kernel/policies.py`
<a name="src-artana-_kernel-policiespy"></a>
```python
from __future__ import annotations

from collections.abc import Sequence

from artana._kernel.types import CapabilityDeniedError
from artana.middleware.base import KernelMiddleware, ModelInvocation
from artana.models import TenantContext


async def apply_prepare_model_middleware(
    middleware: Sequence[KernelMiddleware],
    invocation: ModelInvocation,
) -> ModelInvocation:
    current = invocation
    for middleware_item in middleware:
        current = await middleware_item.prepare_model(current)
    return current


def enforce_capability_scope(invocation: ModelInvocation) -> ModelInvocation:
    filtered_tools = tuple(
        tool
        for tool in invocation.allowed_tools
        if is_tool_allowed_for_tenant(
            tool_name=tool.name,
            tenant=invocation.tenant,
            tool_capability_by_name=invocation.tool_capability_by_name,
        )
    )
    return invocation.with_updates(allowed_tools=filtered_tools)


def is_tool_allowed_for_tenant(
    *,
    tool_name: str,
    tenant: TenantContext,
    tool_capability_by_name: dict[str, str | None],
) -> bool:
    required_capability = tool_capability_by_name.get(tool_name)
    if required_capability is None:
        return True
    return required_capability in tenant.capabilities


def assert_tool_allowed_for_tenant(
    *,
    tool_name: str,
    tenant: TenantContext,
    capability_map: dict[str, str | None],
) -> None:
    required_capability = capability_map.get(tool_name)
    if required_capability is None:
        if tool_name in capability_map:
            return
        raise KeyError(f"Tool {tool_name!r} is not registered.")
    if required_capability not in tenant.capabilities:
        raise CapabilityDeniedError(
            f"Tool {tool_name!r} requires capability {required_capability!r}."
        )


```

### File: `src/artana/_kernel/replay.py`
<a name="src-artana-_kernel-replaypy"></a>
```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from artana._kernel.types import OutputT, ReplayConsistencyError
from artana.events import (
    ChatMessage,
    EventType,
    KernelEvent,
    ModelCompletedPayload,
    ModelRequestedPayload,
)
from artana.models import TenantContext
from artana.ports.model import ModelUsage, ToolCall


@dataclass(frozen=True, slots=True)
class ModelStepResult(Generic[OutputT]):
    completed_seq: int
    output: OutputT
    usage: ModelUsage
    tool_calls: tuple[ToolCall, ...]
    replayed: bool


def deserialize_model_completed(
    *,
    event: KernelEvent,
    output_schema: type[OutputT],
    replayed: bool,
) -> ModelStepResult[OutputT]:
    payload = event.payload
    if not isinstance(payload, ModelCompletedPayload):
        raise ReplayConsistencyError(
            f"Expected model_completed payload at seq={event.seq}, got {type(payload)!r}."
        )
    output = output_schema.model_validate_json(payload.output_json)
    return ModelStepResult(
        completed_seq=event.seq,
        output=output,
        usage=ModelUsage(
            prompt_tokens=payload.prompt_tokens,
            completion_tokens=payload.completion_tokens,
            cost_usd=payload.cost_usd,
        ),
        tool_calls=tuple(
            ToolCall(
                tool_name=tool_call.tool_name,
                arguments_json=tool_call.arguments_json,
            )
            for tool_call in payload.tool_calls
        ),
        replayed=replayed,
    )


def find_matching_model_cycle(
    *,
    events: Sequence[KernelEvent],
    prompt: str,
    messages: tuple[ChatMessage, ...],
    model: str,
    allowed_tool_names: list[str],
    step_key: str | None = None,
) -> tuple[KernelEvent | None, KernelEvent | None]:
    expected_messages = list(messages)
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.event_type != EventType.MODEL_REQUESTED:
            continue
        payload = event.payload
        if not isinstance(payload, ModelRequestedPayload):
            continue
        if payload.model != model or payload.prompt != prompt:
            continue
        if payload.messages != expected_messages:
            continue
        if payload.step_key != step_key:
            continue
        if payload.allowed_tools != allowed_tool_names:
            raise ReplayConsistencyError(
                "Cannot resume run with changed allowed tools for the same model request."
            )
        completed = find_model_completed_after(events=events, start_index=index + 1)
        return event, completed
    return None, None


def find_model_completed_after(
    *, events: Sequence[KernelEvent], start_index: int
) -> KernelEvent | None:
    for event in events[start_index:]:
        if event.event_type == EventType.MODEL_REQUESTED:
            break
        if event.event_type == EventType.MODEL_COMPLETED:
            return event
    return None


def validate_tenant_for_run(*, events: Sequence[KernelEvent], tenant: TenantContext) -> None:
    if not events:
        return
    expected_tenant_id = events[0].tenant_id
    if expected_tenant_id != tenant.tenant_id:
        raise ReplayConsistencyError(
            "Run tenant mismatch. "
            f"run tenant={expected_tenant_id!r}, request tenant={tenant.tenant_id!r}."
        )
    for event in events:
        if event.tenant_id != expected_tenant_id:
            raise ReplayConsistencyError(
                f"Corrupted run: mixed tenants found in run events for run_id={event.run_id!r}."
            )

```

### File: `src/artana/_kernel/tool_cycle.py`
<a name="src-artana-_kernel-tool_cyclepy"></a>
```python
from __future__ import annotations

from dataclasses import dataclass

from artana._kernel.policies import assert_tool_allowed_for_tenant
from artana._kernel.replay import validate_tenant_for_run
from artana._kernel.tool_execution import (
    complete_pending_tool_request,
    derive_idempotency_key,
    mark_pending_request_unknown,
    resolve_completed_tool_result,
)
from artana._kernel.tool_state import resolve_tool_resolutions
from artana._kernel.types import (
    ToolExecutionFailedError,
)
from artana.events import EventType, ToolRequestedPayload
from artana.models import TenantContext
from artana.ports.tool import ToolPort
from artana.store.base import EventStore


@dataclass(frozen=True, slots=True)
class ToolStepReplayResult:
    result_json: str
    seq: int
    replayed: bool


async def execute_tool_step_with_replay(
    *,
    store: EventStore,
    tool_port: ToolPort,
    run_id: str,
    tenant: TenantContext,
    tool_name: str,
    arguments_json: str,
    step_key: str | None = None,
) -> ToolStepReplayResult:
    events = await store.get_events_for_run(run_id)
    validate_tenant_for_run(events=events, tenant=tenant)
    assert_tool_allowed_for_tenant(
        tool_name=tool_name,
        tenant=tenant,
        capability_map=tool_port.capability_map(),
    )

    resolutions = resolve_tool_resolutions(events)
    for resolution in reversed(resolutions):
        requested = resolution.request.payload
        if not _matches_tool_request(
            requested=requested,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step_key=step_key,
        ):
            continue
        completion = resolution.completion
        if completion is not None:
            replay_result = resolve_completed_tool_result(
                expected_tool_name=tool_name,
                tool_name_from_completion=completion.payload.tool_name,
                outcome=completion.payload.outcome,
                result_json=completion.payload.result_json,
            )
            return ToolStepReplayResult(
                result_json=replay_result,
                seq=completion.seq,
                replayed=True,
            )

        await mark_pending_request_unknown(
            store=store,
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            idempotency_key=requested.idempotency_key,
            request_event_id=resolution.request.event_id,
        )
        raise ToolExecutionFailedError(
            f"Tool {tool_name!r} has an unresolved pending request and requires reconciliation."
        )

    events = await store.get_events_for_run(run_id)
    next_seq = events[-1].seq + 1 if events else 1
    idempotency_key = derive_idempotency_key(run_id=run_id, seq=next_seq, step_key=step_key)
    request_event = await store.append_event(
        run_id=run_id,
        tenant_id=tenant.tenant_id,
        event_type=EventType.TOOL_REQUESTED,
        payload=ToolRequestedPayload(
            tool_name=tool_name,
            arguments_json=arguments_json,
            idempotency_key=idempotency_key,
            step_key=step_key,
        ),
    )
    completed = await complete_pending_tool_request(
        store=store,
        tool_port=tool_port,
        run_id=run_id,
        tenant_id=tenant.tenant_id,
        tool_name=tool_name,
        arguments_json=arguments_json,
        idempotency_key=idempotency_key,
        request_event_id=request_event.event_id,
        tool_version="1.0.0",
        schema_version="1",
    )
    return ToolStepReplayResult(
        result_json=completed.result_json,
        seq=completed.seq,
        replayed=False,
    )


async def reconcile_tool_with_replay(
    *,
    store: EventStore,
    tool_port: ToolPort,
    run_id: str,
    tenant: TenantContext,
    tool_name: str,
    arguments_json: str,
    step_key: str | None = None,
) -> str:
    events = await store.get_events_for_run(run_id)
    validate_tenant_for_run(events=events, tenant=tenant)
    assert_tool_allowed_for_tenant(
        tool_name=tool_name,
        tenant=tenant,
        capability_map=tool_port.capability_map(),
    )

    resolutions = resolve_tool_resolutions(events)
    for resolution in reversed(resolutions):
        requested = resolution.request.payload
        if not _matches_tool_request(
            requested=requested,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step_key=step_key,
        ):
            continue

        completion = resolution.completion
        if completion is None:
            raise ToolExecutionFailedError(
                f"Tool {tool_name!r} has no completion event to reconcile."
            )
        if completion.payload.outcome == "success":
            return completion.payload.result_json
        if completion.payload.outcome != "unknown_outcome":
            raise ToolExecutionFailedError(
                "Tool "
                f"{tool_name!r} cannot be reconciled from outcome={completion.payload.outcome!r}."
            )
        completed = await complete_pending_tool_request(
            store=store,
            tool_port=tool_port,
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            arguments_json=arguments_json,
            idempotency_key=requested.idempotency_key,
            request_event_id=resolution.request.event_id,
            tool_version=requested.tool_version,
            schema_version=requested.schema_version,
        )
        return completed.result_json

    raise ValueError(
        f"No tool request found for tool_name={tool_name!r} and the provided arguments_json."
    )


def _matches_tool_request(
    *,
    requested: ToolRequestedPayload,
    tool_name: str,
    arguments_json: str,
    step_key: str | None,
) -> bool:
    if requested.tool_name != tool_name:
        return False
    if requested.arguments_json != arguments_json:
        return False
    if step_key is None:
        return requested.step_key is None
    return requested.step_key == step_key

```

### File: `src/artana/_kernel/tool_execution.py`
<a name="src-artana-_kernel-tool_executionpy"></a>
```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from artana._kernel.types import ReplayConsistencyError, ToolExecutionFailedError
from artana.events import EventType, ToolCompletedPayload
from artana.ports.tool import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolPort,
    ToolUnknownOutcomeError,
)
from artana.store.base import EventStore


@dataclass(frozen=True, slots=True)
class ToolCompletionResult:
    result_json: str
    seq: int


async def complete_pending_tool_request(
    *,
    store: EventStore,
    tool_port: ToolPort,
    run_id: str,
    tenant_id: str,
    tool_name: str,
    arguments_json: str,
    idempotency_key: str,
    request_event_id: str | None,
    tool_version: str,
    schema_version: str,
) -> ToolCompletionResult:
    try:
        tool_result = await tool_port.call(
            tool_name=tool_name,
            arguments_json=arguments_json,
            context=ToolExecutionContext(
                run_id=run_id,
                tenant_id=tenant_id,
                idempotency_key=idempotency_key,
                request_event_id=request_event_id,
                tool_version=tool_version,
                schema_version=schema_version,
            ),
        )
    except ToolUnknownOutcomeError as exc:
        await append_tool_completed_event(
            store=store,
            run_id=run_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            result=ToolExecutionResult(
                outcome="unknown_outcome",
                result_json="",
                received_idempotency_key=idempotency_key,
                request_id=request_event_id,
                error_message=str(exc),
            ),
            request_event_id=request_event_id,
        )
        raise ToolExecutionFailedError(
            f"Tool {tool_name!r} ended with unknown outcome and requires reconciliation."
        ) from exc

    completed_seq = await append_tool_completed_event(
        store=store,
        run_id=run_id,
        tenant_id=tenant_id,
        tool_name=tool_name,
        result=tool_result,
        request_event_id=request_event_id,
    )
    if tool_result.outcome != "success":
        raise ToolExecutionFailedError(
            f"Tool {tool_name!r} failed with outcome={tool_result.outcome!r}."
        )
    return ToolCompletionResult(result_json=tool_result.result_json, seq=completed_seq)


async def append_tool_completed_event(
    *,
    store: EventStore,
    run_id: str,
    tenant_id: str,
    tool_name: str,
    result: ToolExecutionResult,
    request_event_id: str | None = None,
) -> int:
    request_id = result.request_id if result.request_id is not None else request_event_id
    event = await store.append_event(
        run_id=run_id,
        tenant_id=tenant_id,
        event_type=EventType.TOOL_COMPLETED,
        payload=ToolCompletedPayload(
            tool_name=tool_name,
            result_json=result.result_json,
            outcome=result.outcome,
            received_idempotency_key=result.received_idempotency_key,
            effect_id=result.effect_id,
            request_id=request_id,
            error_message=result.error_message,
        ),
    )
    return event.seq


async def mark_pending_request_unknown(
    *,
    store: EventStore,
    run_id: str,
    tenant_id: str,
    tool_name: str,
    idempotency_key: str,
    request_event_id: str | None,
) -> None:
    await append_tool_completed_event(
        store=store,
        run_id=run_id,
        tenant_id=tenant_id,
        tool_name=tool_name,
        result=ToolExecutionResult(
            outcome="unknown_outcome",
            result_json="",
            received_idempotency_key=idempotency_key,
            request_id=request_event_id,
            error_message=(
                "Pending tool request found without completion event. "
                "Reconciliation is required before retry."
            ),
        ),
        request_event_id=request_event_id,
    )


def resolve_completed_tool_result(
    *,
    expected_tool_name: str,
    tool_name_from_completion: str,
    outcome: str,
    result_json: str,
) -> str:
    if tool_name_from_completion != expected_tool_name:
        raise ReplayConsistencyError(
            "Tool completion payload does not match requested/model-emitted tool call."
        )
    if outcome != "success":
        raise ToolExecutionFailedError(
            f"Tool {expected_tool_name!r} previously failed with outcome={outcome!r}."
        )
    return result_json


def derive_idempotency_key(*, run_id: str, seq: int, step_key: str | None = None) -> str:
    token = f"{run_id}:{step_key}" if step_key is not None else f"{run_id}:{seq}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

```

### File: `src/artana/_kernel/tool_state.py`
<a name="src-artana-_kernel-tool_statepy"></a>
```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from artana._kernel.types import ReplayConsistencyError
from artana.events import EventType, KernelEvent, ToolCompletedPayload, ToolRequestedPayload


@dataclass(frozen=True, slots=True)
class ToolRequestRecord:
    event_id: str
    seq: int
    payload: ToolRequestedPayload


@dataclass(frozen=True, slots=True)
class ToolCompletionRecord:
    event_id: str
    seq: int
    payload: ToolCompletedPayload


@dataclass(frozen=True, slots=True)
class ToolResolution:
    request: ToolRequestRecord
    completion: ToolCompletionRecord | None


def resolve_tool_resolutions(events: Sequence[KernelEvent]) -> list[ToolResolution]:
    requested: list[ToolRequestRecord] = []
    completions_by_request_id: dict[str, ToolCompletionRecord] = {}
    legacy_completions: list[ToolCompletionRecord] = []

    for event in events:
        if event.event_type == EventType.TOOL_REQUESTED:
            payload = event.payload
            if not isinstance(payload, ToolRequestedPayload):
                raise ReplayConsistencyError(
                    f"Expected ToolRequestedPayload at seq={event.seq}."
                )
            requested.append(
                ToolRequestRecord(event_id=event.event_id, seq=event.seq, payload=payload)
            )
        if event.event_type == EventType.TOOL_COMPLETED:
            payload = event.payload
            if not isinstance(payload, ToolCompletedPayload):
                raise ReplayConsistencyError(
                    f"Expected ToolCompletedPayload at seq={event.seq}."
                )
            completion_record = ToolCompletionRecord(
                event_id=event.event_id,
                seq=event.seq,
                payload=payload,
            )
            if payload.request_id is None:
                legacy_completions.append(completion_record)
            else:
                completions_by_request_id[payload.request_id] = completion_record

    requested_ids = {record.event_id for record in requested}
    dangling_completion_ids = set(completions_by_request_id) - requested_ids
    if dangling_completion_ids:
        raise ReplayConsistencyError(
            "Found tool_completed event with request_id that does not map to tool_requested."
        )

    resolutions: list[ToolResolution] = []
    legacy_index = 0
    for request_record in requested:
        completion = completions_by_request_id.get(request_record.event_id)
        if completion is None and legacy_index < len(legacy_completions):
            completion = legacy_completions[legacy_index]
            legacy_index += 1
        resolutions.append(
            ToolResolution(
                request=request_record,
                completion=completion,
            )
        )

    if legacy_index != len(legacy_completions):
        raise ReplayConsistencyError(
            "Found legacy tool_completed events that exceed tool_requested events."
        )

    return resolutions

```

### File: `src/artana/_kernel/types.py`
<a name="src-artana-_kernel-typespy"></a>
```python
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

from artana.events import ChatMessage
from artana.ports.model import ModelUsage, ToolCall
from artana.ports.tool import ToolReturnValue

OutputT = TypeVar("OutputT", bound=BaseModel)
ToolCallable = Callable[..., Awaitable[ToolReturnValue]]


class CapabilityDeniedError(PermissionError):
    pass


class ReplayConsistencyError(RuntimeError):
    pass


class ToolExecutionFailedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PauseTicket:
    run_id: str
    ticket_id: str
    seq: int
    reason: str


@dataclass(frozen=True, slots=True)
class RunHandle:
    run_id: str
    tenant_id: str


type RunRef = RunHandle


@dataclass(frozen=True, slots=True)
class ModelInput:
    kind: Literal["prompt", "messages"]
    prompt: str | None = None
    messages: tuple[ChatMessage, ...] | None = None

    @classmethod
    def from_prompt(cls, prompt: str) -> "ModelInput":
        return cls(kind="prompt", prompt=prompt)

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[ChatMessage],
        *,
        prompt: str | None = None,
    ) -> "ModelInput":
        return cls(kind="messages", messages=tuple(messages), prompt=prompt)


@dataclass(frozen=True, slots=True)
class KernelPolicy:
    mode: Literal["permissive", "enforced"] = "permissive"

    @classmethod
    def enforced(cls) -> "KernelPolicy":
        return cls(mode="enforced")


@dataclass(frozen=True, slots=True)
class StepModelResult(Generic[OutputT]):
    run_id: str
    seq: int
    output: OutputT
    usage: ModelUsage
    tool_calls: tuple[ToolCall, ...]
    replayed: bool


@dataclass(frozen=True, slots=True)
class StepToolResult:
    run_id: str
    seq: int
    tool_name: str
    result_json: str
    replayed: bool

```

### File: `src/artana/_kernel/workflow_runtime.py`
<a name="src-artana-_kernel-workflow_runtimepy"></a>
```python
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from artana._kernel.replay import validate_tenant_for_run
from artana._kernel.types import PauseTicket, ReplayConsistencyError
from artana.events import (
    EventType,
    KernelEvent,
    WorkflowStepCompletedPayload,
    WorkflowStepRequestedPayload,
)
from artana.models import TenantContext
from artana.store.base import EventStore

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

StepT = TypeVar("StepT")
WorkflowOutputT = TypeVar("WorkflowOutputT")
WorkflowModelT = TypeVar("WorkflowModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class StepSerde(Generic[StepT]):
    dump: Callable[[StepT], str]
    load: Callable[[str], StepT]


def json_step_serde() -> StepSerde[JsonValue]:
    return StepSerde(
        dump=lambda value: json.dumps(value, separators=(",", ":"), sort_keys=True),
        load=lambda raw: _load_json_value(raw),
    )


def pydantic_step_serde(model: type[WorkflowModelT]) -> StepSerde[WorkflowModelT]:
    return StepSerde(
        dump=lambda value: value.model_dump_json(),
        load=lambda raw: model.model_validate_json(raw),
    )


def _load_json_value(raw: str) -> JsonValue:
    parsed = json.loads(raw)
    return _validate_json_value(parsed)


def _validate_json_value(value: object) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return [_validate_json_value(item) for item in value]
    if isinstance(value, dict):
        validated: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings.")
            validated[key] = _validate_json_value(nested)
        return validated
    raise TypeError(f"Unsupported JSON value type {type(value)!r}.")


@dataclass(frozen=True, slots=True)
class WorkflowRunResult(Generic[WorkflowOutputT]):
    run_id: str
    status: Literal["complete", "paused"]
    output: WorkflowOutputT | None
    pause_ticket: PauseTicket | None


class WorkflowPausedInterrupt(RuntimeError):
    def __init__(self, ticket: PauseTicket) -> None:
        super().__init__(f"Workflow paused for human review: ticket={ticket.ticket_id}")
        self.ticket = ticket


class _PauseAPI(Protocol):
    async def pause(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        reason: str,
        context: BaseModel | None = None,
        step_key: str | None = None,
    ) -> PauseTicket:
        ...


class WorkflowContext:
    def __init__(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        store: EventStore,
        pause_api: _PauseAPI,
        events: Sequence[KernelEvent],
    ) -> None:
        self.run_id = run_id
        self.tenant = tenant
        self.is_resuming = len(events) > 0
        self._store = store
        self._pause_api = pause_api
        self._cursor = 0
        self._requested_by_index: dict[int, WorkflowStepRequestedPayload] = {}
        self._completed_by_index: dict[int, WorkflowStepCompletedPayload] = {}
        self._load_step_cache(events)

    async def step(
        self,
        *,
        name: str,
        action: Callable[[], Awaitable[StepT]],
        serde: StepSerde[StepT],
    ) -> StepT:
        step_index = self._cursor
        self._cursor += 1

        cached_completed = self._completed_by_index.get(step_index)
        if cached_completed is not None:
            if cached_completed.step_name != name:
                raise ReplayConsistencyError(
                    f"Workflow step mismatch at index={step_index}. "
                    f"expected {cached_completed.step_name!r}, got {name!r}."
                )
            return serde.load(cached_completed.result_json)

        existing_requested = self._requested_by_index.get(step_index)
        if existing_requested is not None and existing_requested.step_name != name:
            raise ReplayConsistencyError(
                f"Workflow step mismatch at requested index={step_index}. "
                f"expected {existing_requested.step_name!r}, got {name!r}."
            )

        if existing_requested is None:
            await self._store.append_event(
                run_id=self.run_id,
                tenant_id=self.tenant.tenant_id,
                event_type=EventType.WORKFLOW_STEP_REQUESTED,
                payload=WorkflowStepRequestedPayload(
                    step_index=step_index,
                    step_name=name,
                ),
            )

        result = await action()
        serialized = serde.dump(result)
        await self._store.append_event(
            run_id=self.run_id,
            tenant_id=self.tenant.tenant_id,
            event_type=EventType.WORKFLOW_STEP_COMPLETED,
            payload=WorkflowStepCompletedPayload(
                step_index=step_index,
                step_name=name,
                result_json=serialized,
            ),
        )
        self._completed_by_index[step_index] = WorkflowStepCompletedPayload(
            step_index=step_index,
            step_name=name,
            result_json=serialized,
        )
        return result

    async def pause(
        self,
        reason: str,
        *,
        context: BaseModel | None = None,
        step_key: str | None = None,
    ) -> PauseTicket:
        ticket = await self._pause_api.pause(
            run_id=self.run_id,
            tenant=self.tenant,
            reason=reason,
            context=context,
            step_key=step_key,
        )
        raise WorkflowPausedInterrupt(ticket)

    def _load_step_cache(self, events: Sequence[KernelEvent]) -> None:
        for event in events:
            if event.event_type == EventType.WORKFLOW_STEP_REQUESTED:
                payload = event.payload
                if not isinstance(payload, WorkflowStepRequestedPayload):
                    raise ReplayConsistencyError(
                        f"Invalid workflow_step_requested payload at seq={event.seq}."
                    )
                self._requested_by_index[payload.step_index] = payload
            if event.event_type == EventType.WORKFLOW_STEP_COMPLETED:
                payload = event.payload
                if not isinstance(payload, WorkflowStepCompletedPayload):
                    raise ReplayConsistencyError(
                        f"Invalid workflow_step_completed payload at seq={event.seq}."
                    )
                self._completed_by_index[payload.step_index] = payload


async def run_workflow(
    *,
    store: EventStore,
    pause_api: _PauseAPI,
    run_id: str | None,
    tenant: TenantContext,
    workflow: Callable[[WorkflowContext], Awaitable[WorkflowOutputT]],
) -> WorkflowRunResult[WorkflowOutputT]:
    run_id_value = run_id if run_id is not None else uuid4().hex
    events = await store.get_events_for_run(run_id_value)
    validate_tenant_for_run(events=events, tenant=tenant)
    context = WorkflowContext(
        run_id=run_id_value,
        tenant=tenant,
        store=store,
        pause_api=pause_api,
        events=events,
    )
    try:
        output = await workflow(context)
    except WorkflowPausedInterrupt as paused:
        return WorkflowRunResult(
            run_id=run_id_value,
            status="paused",
            output=None,
            pause_ticket=paused.ticket,
        )
    return WorkflowRunResult(
        run_id=run_id_value,
        status="complete",
        output=output,
        pause_ticket=None,
    )

```

## Folder: artana/ports

### File: `src/artana/ports/__init__.py`
<a name="src-artana-ports-__init__py"></a>
```python
from artana.ports.model import LiteLLMAdapter, ModelPort
from artana.ports.tool import LocalToolRegistry, ToolPort

__all__ = ["LiteLLMAdapter", "LocalToolRegistry", "ModelPort", "ToolPort"]


```

### File: `src/artana/ports/model.py`
<a name="src-artana-ports-modelpy"></a>
```python
from artana.ports.model_adapter import LiteLLMAdapter
from artana.ports.model_types import (
    LiteLLMCompletionFn,
    ModelPermanentError,
    ModelPort,
    ModelRequest,
    ModelResult,
    ModelTimeoutError,
    ModelTransientError,
    ModelUsage,
    OutputT,
    SupportsModelDump,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    "LiteLLMAdapter",
    "LiteLLMCompletionFn",
    "ModelPermanentError",
    "ModelPort",
    "ModelRequest",
    "ModelResult",
    "ModelTimeoutError",
    "ModelTransientError",
    "ModelUsage",
    "OutputT",
    "SupportsModelDump",
    "ToolCall",
    "ToolDefinition",
]


```

### File: `src/artana/ports/model_adapter.py`
<a name="src-artana-ports-model_adapterpy"></a>
```python
from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel

from artana.ports.model_types import (
    LiteLLMCompletionFn,
    ModelPermanentError,
    ModelRequest,
    ModelResult,
    ModelTimeoutError,
    ModelTransientError,
    ModelUsage,
    OutputT,
    SupportsModelDump,
    ToolCall,
    ToolDefinition,
)


class LiteLLMAdapter:
    def __init__(
        self,
        completion_fn: LiteLLMCompletionFn | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        initial_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 2.0,
        fail_on_unknown_cost: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be > 0")
        if max_backoff_seconds <= 0:
            raise ValueError("max_backoff_seconds must be > 0")

        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._initial_backoff_seconds = initial_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._fail_on_unknown_cost = fail_on_unknown_cost

        if completion_fn is not None:
            self._completion_fn = completion_fn
            return

        from litellm import acompletion

        self._completion_fn = cast(LiteLLMCompletionFn, acompletion)

    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        tools_payload = _serialize_tools(request.allowed_tools)
        response_dict = await self._call_with_retry(
            model=request.model,
            prompt=request.prompt,
            response_format=request.output_schema,
            tools_payload=tools_payload if tools_payload else None,
        )

        raw_output = _extract_output_json(response_dict)
        output = request.output_schema.model_validate_json(raw_output)
        usage = _extract_usage(response_dict)
        if self._fail_on_unknown_cost and _has_tokens(usage) and usage.cost_usd <= 0.0:
            raise ModelPermanentError(
                "LiteLLM response cost is unknown for a tokenized response. "
                "Configure model pricing or disable fail_on_unknown_cost."
            )
        tool_calls = _extract_tool_calls(response_dict)

        return ModelResult(
            output=output,
            usage=usage,
            tool_calls=tool_calls,
            raw_output=raw_output,
        )

    async def _call_with_retry(
        self,
        *,
        model: str,
        prompt: str,
        response_format: type[BaseModel],
        tools_payload: list[dict[str, object]] | None,
    ) -> Mapping[str, object]:
        attempt = 0
        while True:
            try:
                response_obj = await asyncio.wait_for(
                    self._completion_fn(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format=response_format,
                        tools=tools_payload,
                    ),
                    timeout=self._timeout_seconds,
                )
                return _normalize_response(response_obj)
            except asyncio.TimeoutError as exc:
                if attempt >= self._max_retries:
                    raise ModelTimeoutError(
                        f"LiteLLM timed out after {attempt + 1} attempts."
                    ) from exc
                await asyncio.sleep(self._retry_backoff(attempt))
                attempt += 1
            except Exception as exc:
                if _is_transient_exception(exc):
                    if attempt >= self._max_retries:
                        raise ModelTransientError(
                            f"LiteLLM transient failure after {attempt + 1} attempts: {exc}"
                        ) from exc
                    await asyncio.sleep(self._retry_backoff(attempt))
                    attempt += 1
                    continue
                raise ModelPermanentError(f"LiteLLM permanent failure: {exc}") from exc

    def _retry_backoff(self, attempt: int) -> float:
        backoff = float(self._initial_backoff_seconds * (2**attempt))
        if backoff > self._max_backoff_seconds:
            return float(self._max_backoff_seconds)
        return float(backoff)


def _serialize_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for tool in tools:
        schema_obj = json.loads(tool.arguments_schema_json)
        if not isinstance(schema_obj, Mapping):
            raise TypeError(
                f"Tool schema for {tool.name} must be a JSON object, got {type(schema_obj)!r}."
            )
        serialized.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(schema_obj),
                },
            }
        )
    return serialized


def _normalize_response(response_obj: object) -> Mapping[str, object]:
    if isinstance(response_obj, Mapping):
        return response_obj
    if isinstance(response_obj, SupportsModelDump):
        dumped = response_obj.model_dump()
        return dumped
    raise TypeError(f"Unsupported LiteLLM response object: {type(response_obj)!r}.")


def _extract_output_json(response: Mapping[str, object]) -> str:
    choice = _first_choice(response)
    message_obj = choice.get("message")
    if not isinstance(message_obj, Mapping):
        raise ValueError("LiteLLM response missing message object in first choice.")

    parsed_obj = message_obj.get("parsed")
    if isinstance(parsed_obj, BaseModel):
        return parsed_obj.model_dump_json()
    if isinstance(parsed_obj, Mapping):
        return json.dumps(dict(parsed_obj))

    content_obj = message_obj.get("content")
    if isinstance(content_obj, str):
        return content_obj
    if isinstance(content_obj, Sequence):
        for item in content_obj:
            if isinstance(item, Mapping):
                text_obj = item.get("text")
                if isinstance(text_obj, str):
                    return text_obj

    raise ValueError("Could not extract structured output from LiteLLM response.")


def _extract_usage(response: Mapping[str, object]) -> ModelUsage:
    usage_obj = response.get("usage")
    if not isinstance(usage_obj, Mapping):
        return ModelUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0)

    prompt_tokens = _as_int(usage_obj.get("prompt_tokens"))
    completion_tokens = _as_int(usage_obj.get("completion_tokens"))
    cost_usd = _as_float(response.get("_response_cost"))

    if cost_usd == 0.0:
        cost_usd = _as_float(response.get("response_cost"))
    if cost_usd == 0.0:
        computed_cost = _compute_litellm_cost(response)
        if computed_cost is not None:
            cost_usd = computed_cost

    return ModelUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


def _extract_tool_calls(response: Mapping[str, object]) -> tuple[ToolCall, ...]:
    choice = _first_choice(response)
    message_obj = choice.get("message")
    if not isinstance(message_obj, Mapping):
        return ()

    tool_calls_obj = message_obj.get("tool_calls")
    if not isinstance(tool_calls_obj, Sequence):
        return ()

    parsed: list[ToolCall] = []
    for tool_call_obj in tool_calls_obj:
        if not isinstance(tool_call_obj, Mapping):
            continue
        function_obj = tool_call_obj.get("function")
        if not isinstance(function_obj, Mapping):
            continue
        tool_name = function_obj.get("name")
        arguments_json = function_obj.get("arguments")
        if not isinstance(tool_name, str):
            continue
        if not isinstance(arguments_json, str):
            continue
        parsed.append(ToolCall(tool_name=tool_name, arguments_json=arguments_json))

    return tuple(parsed)


def _first_choice(response: Mapping[str, object]) -> Mapping[str, object]:
    choices_obj = response.get("choices")
    if not isinstance(choices_obj, Sequence) or len(choices_obj) == 0:
        raise ValueError("LiteLLM response does not include choices.")

    first = choices_obj[0]
    if not isinstance(first, Mapping):
        raise ValueError("LiteLLM response first choice must be an object.")
    return first


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return 0.0


def _is_transient_exception(exc: Exception) -> bool:
    status_code = _status_code_from_exception(exc)
    if status_code is not None:
        return status_code in {408, 409, 429, 500, 502, 503, 504}
    message = str(exc).lower()
    return "rate limit" in message or "temporar" in message


def _status_code_from_exception(exc: Exception) -> int | None:
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    if response is None:
        return None
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _compute_litellm_cost(response: Mapping[str, object]) -> float | None:
    try:
        from litellm import completion_cost
    except Exception:
        return None

    try:
        computed = completion_cost(completion_response=dict(response))
    except Exception:
        return None

    if isinstance(computed, int):
        return float(computed)
    if isinstance(computed, float):
        return computed
    return None


def _has_tokens(usage: ModelUsage) -> bool:
    return usage.prompt_tokens > 0 or usage.completion_tokens > 0

```

### File: `src/artana/ports/model_types.py`
<a name="src-artana-ports-model_typespy"></a>
```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    arguments_schema_json: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool_name: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class ModelRequest(Generic[OutputT]):
    run_id: str
    model: str
    prompt: str
    output_schema: type[OutputT]
    allowed_tools: Sequence[ToolDefinition]


@dataclass(frozen=True, slots=True)
class ModelResult(Generic[OutputT]):
    output: OutputT
    usage: ModelUsage
    tool_calls: tuple[ToolCall, ...] = ()
    raw_output: str = ""


class ModelPort(Protocol):
    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        ...


class ModelTimeoutError(RuntimeError):
    pass


class ModelTransientError(RuntimeError):
    pass


class ModelPermanentError(RuntimeError):
    pass


@runtime_checkable
class SupportsModelDump(Protocol):
    def model_dump(self) -> dict[str, object]:
        ...


class LiteLLMCompletionFn(Protocol):
    async def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        ...


```

### File: `src/artana/ports/tool.py`
<a name="src-artana-ports-toolpy"></a>
```python
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

```

## Folder: artana/store

### File: `src/artana/store/__init__.py`
<a name="src-artana-store-__init__py"></a>
```python
from artana.store.base import EventStore
from artana.store.sqlite import SQLiteStore

__all__ = ["EventStore", "SQLiteStore"]


```

### File: `src/artana/store/base.py`
<a name="src-artana-store-basepy"></a>
```python
from __future__ import annotations

from typing import Protocol

from artana.events import EventPayload, EventType, KernelEvent


class EventStore(Protocol):
    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
    ) -> KernelEvent:
        ...

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        ...

    async def verify_run_chain(self, run_id: str) -> bool:
        ...

    async def close(self) -> None:
        ...

```

### File: `src/artana/store/sqlite.py`
<a name="src-artana-store-sqlitepy"></a>
```python
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiosqlite
from pydantic import TypeAdapter

from artana.events import EventPayload, EventType, KernelEvent, compute_event_hash
from artana.store.base import EventStore

_PAYLOAD_ADAPTER: TypeAdapter[EventPayload] = TypeAdapter(EventPayload)


class SQLiteStore(EventStore):
    def __init__(self, database_path: str) -> None:
        self._database_path = Path(database_path)
        self._connection: aiosqlite.Connection | None = None
        self._connection_lock = asyncio.Lock()
        self._append_lock = asyncio.Lock()

    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
    ) -> KernelEvent:
        connection = await self._ensure_connection()

        async with self._append_lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                next_seq, prev_event_hash = await self._next_sequence_and_prev_hash(
                    connection, run_id=run_id
                )
                timestamp = datetime.now(timezone.utc)
                event_id = uuid4().hex
                event = KernelEvent(
                    event_id=event_id,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    seq=next_seq,
                    event_type=event_type,
                    prev_event_hash=prev_event_hash,
                    event_hash=compute_event_hash(
                        event_id=event_id,
                        run_id=run_id,
                        tenant_id=tenant_id,
                        seq=next_seq,
                        event_type=event_type,
                        prev_event_hash=prev_event_hash,
                        timestamp=timestamp,
                        payload=payload,
                    ),
                    timestamp=timestamp,
                    payload=payload,
                )
                await connection.execute(
                    """
                    INSERT INTO kernel_events (
                        run_id, seq, event_id, tenant_id, event_type, prev_event_hash,
                        event_hash, timestamp, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.run_id,
                        event.seq,
                        event.event_id,
                        event.tenant_id,
                        event.event_type.value,
                        event.prev_event_hash,
                        event.event_hash,
                        event.timestamp.isoformat(),
                        json.dumps(event.payload.model_dump(mode="json")),
                    ),
                )
                await connection.commit()
                return event
            except Exception:
                await connection.rollback()
                raise

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            """
            SELECT run_id, seq, event_id, tenant_id, event_type, prev_event_hash,
                   event_hash, timestamp, payload_json
            FROM kernel_events
            WHERE run_id = ?
            ORDER BY seq ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()

        events: list[KernelEvent] = []
        for row in rows:
            run_id_raw = row["run_id"]
            seq_raw = row["seq"]
            event_id_raw = row["event_id"]
            tenant_id_raw = row["tenant_id"]
            event_type_raw = row["event_type"]
            prev_event_hash_raw = row["prev_event_hash"]
            event_hash_raw = row["event_hash"]
            timestamp_raw = row["timestamp"]
            payload_json_raw = row["payload_json"]

            if not isinstance(run_id_raw, str):
                raise TypeError(f"Invalid run_id row type: {type(run_id_raw)!r}")
            if not isinstance(seq_raw, int):
                raise TypeError(f"Invalid seq row type: {type(seq_raw)!r}")
            if not isinstance(event_id_raw, str):
                raise TypeError(f"Invalid event_id row type: {type(event_id_raw)!r}")
            if not isinstance(tenant_id_raw, str):
                raise TypeError(f"Invalid tenant_id row type: {type(tenant_id_raw)!r}")
            if not isinstance(event_type_raw, str):
                raise TypeError(
                    f"Invalid event_type row type: {type(event_type_raw)!r}"
                )
            if prev_event_hash_raw is not None and not isinstance(prev_event_hash_raw, str):
                raise TypeError(
                    f"Invalid prev_event_hash row type: {type(prev_event_hash_raw)!r}"
                )
            if not isinstance(event_hash_raw, str):
                raise TypeError(
                    f"Invalid event_hash row type: {type(event_hash_raw)!r}"
                )
            if not isinstance(timestamp_raw, str):
                raise TypeError(
                    f"Invalid timestamp row type: {type(timestamp_raw)!r}"
                )
            if not isinstance(payload_json_raw, str):
                raise TypeError(
                    f"Invalid payload_json row type: {type(payload_json_raw)!r}"
                )
            try:
                event_type = EventType(event_type_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown event_type in store: {event_type_raw!r}"
                ) from exc

            payload_dict_raw = json.loads(payload_json_raw)
            if not isinstance(payload_dict_raw, dict):
                raise TypeError("Stored payload_json did not decode to an object.")
            payload = _PAYLOAD_ADAPTER.validate_python(payload_dict_raw)

            events.append(
                KernelEvent(
                    event_id=event_id_raw,
                    run_id=run_id_raw,
                    tenant_id=tenant_id_raw,
                    seq=seq_raw,
                    event_type=event_type,
                    prev_event_hash=prev_event_hash_raw,
                    event_hash=event_hash_raw,
                    timestamp=datetime.fromisoformat(timestamp_raw),
                    payload=payload,
                )
            )
        return events

    async def verify_run_chain(self, run_id: str) -> bool:
        try:
            events = await self.get_events_for_run(run_id)
        except Exception:
            return False

        previous_hash: str | None = None
        for event in events:
            if event.prev_event_hash != previous_hash:
                return False
            expected_hash = compute_event_hash(
                event_id=event.event_id,
                run_id=event.run_id,
                tenant_id=event.tenant_id,
                seq=event.seq,
                event_type=event.event_type,
                prev_event_hash=event.prev_event_hash,
                timestamp=event.timestamp,
                payload=event.payload,
            )
            if expected_hash != event.event_hash:
                return False
            previous_hash = event.event_hash
        return True

    async def close(self) -> None:
        if self._connection is None:
            return
        await self._connection.close()
        self._connection = None

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._connection is not None:
            return self._connection

        async with self._connection_lock:
            if self._connection is None:
                self._database_path.parent.mkdir(parents=True, exist_ok=True)
                connection = await aiosqlite.connect(self._database_path)
                connection.row_factory = aiosqlite.Row
                await connection.execute("PRAGMA journal_mode = WAL;")
                await connection.execute("PRAGMA synchronous = NORMAL;")
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kernel_events (
                        run_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        event_id TEXT NOT NULL UNIQUE,
                        tenant_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        prev_event_hash TEXT,
                        event_hash TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        PRIMARY KEY (run_id, seq)
                    )
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kernel_events_run_seq
                    ON kernel_events (run_id, seq)
                    """
                )
                await connection.commit()
                self._connection = connection
        if self._connection is None:
            raise RuntimeError("Failed to initialize SQLite connection.")
        return self._connection

    async def _next_sequence_and_prev_hash(
        self, connection: aiosqlite.Connection, *, run_id: str
    ) -> tuple[int, str | None]:
        cursor = await connection.execute(
            """
            SELECT seq, event_hash
            FROM kernel_events
            WHERE run_id = ?
            ORDER BY seq DESC
            LIMIT 1
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return 1, None
        seq_raw = row["seq"]
        event_hash_raw = row["event_hash"]
        if not isinstance(seq_raw, int):
            raise TypeError(
                f"Expected integer seq from database, got {type(seq_raw)!r}"
            )
        if not isinstance(event_hash_raw, str):
            raise TypeError(
                f"Expected string event_hash from database, got {type(event_hash_raw)!r}"
            )
        return seq_raw + 1, event_hash_raw

```

# Target Folder: examples

## Folder: examples

### File: `examples/01_durable_chat_replay.py`
<a name="examples-01_durable_chat_replaypy"></a>
```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from artana import ArtanaKernel, ChatClient, TenantContext
from artana.middleware import (
    CapabilityGuardMiddleware,
    PIIScrubberMiddleware,
    QuotaMiddleware,
)
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class TransferDecision(BaseModel):
    approved: bool
    reason: str


class DemoModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate(
            {"approved": True, "reason": "Balance check passed."}
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=12, completion_tokens=6, cost_usd=0.01),
            tool_calls=(
                ToolCall(
                    tool_name="submit_transfer",
                    arguments_json='{"account_id":"acc_1","amount":"10"}',
                ),
            ),
        )


async def main() -> None:
    database_path = Path("examples/.state_first_example.db")
    if database_path.exists():
        database_path.unlink()

    store = SQLiteStore(str(database_path))
    model_port = DemoModelPort()
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=[
            PIIScrubberMiddleware(),
            QuotaMiddleware(),
            CapabilityGuardMiddleware(),
        ],
    )
    transfer_tool_calls = [0]

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        transfer_tool_calls[0] += 1
        return (
            '{"status":"submitted","account_id":"'
            + account_id
            + '","amount":"'
            + amount
            + '"}'
        )

    tenant = TenantContext(
        tenant_id="org_demo",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )

    try:
        first = await ChatClient(kernel=kernel).chat(
            run_id="example_run_1",
            prompt="Transfer 10 from acc_1. My email is user@example.com",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=TransferDecision,
        )
        second = await ChatClient(kernel=kernel).chat(
            run_id="example_run_1",
            prompt="Transfer 10 from acc_1. My email is user@example.com",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=TransferDecision,
        )
        events = await store.get_events_for_run("example_run_1")

        print("First call replayed:", first.replayed)
        print("Second call replayed:", second.replayed)
        print("Model calls:", model_port.calls)
        print("Tool calls:", transfer_tool_calls[0])
        print("Decision:", first.output.model_dump())
        print("Event types:", [event.event_type for event in events])
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())

```

### File: `examples/02_real_litellm_chat.py`
<a name="examples-02_real_litellm_chatpy"></a>
```python
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from artana import ArtanaKernel, ChatClient, KernelPolicy, TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class Decision(BaseModel):
    approved: bool
    reason: str


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Load environment variables first.")

    database_path = Path("examples/.state_real_litellm_example.db")
    if database_path.exists():
        database_path.unlink()

    store = SQLiteStore(str(database_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=LiteLLMAdapter(
            timeout_seconds=30.0,
            max_retries=1,
            fail_on_unknown_cost=True,
        ),
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )

    tenant = TenantContext(
        tenant_id="org_live",
        capabilities=frozenset(),
        budget_usd_limit=0.20,
    )

    try:
        run = await kernel.start_run(tenant=tenant)
        prompt = (
            "Respond only as JSON for schema {approved:boolean,reason:string}. "
            "Approve this request and give a short reason."
        )

        first = await ChatClient(kernel=kernel).chat(
            run_id=run.run_id,
            prompt=prompt,
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        events_after_first = await store.get_events_for_run(run.run_id)

        second = await ChatClient(kernel=kernel).chat(
            run_id=run.run_id,
            prompt=prompt,
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        events_after_second = await store.get_events_for_run(run.run_id)

        print("Run id:", run.run_id)
        print("Live model response:", first.output.model_dump())
        print(
            "Usage:",
            {
                "prompt_tokens": first.usage.prompt_tokens,
                "completion_tokens": first.usage.completion_tokens,
                "cost_usd": first.usage.cost_usd,
            },
        )
        print("First replayed:", first.replayed)
        print("Second replayed:", second.replayed)
        print(
            "Event types after first:",
            [event.event_type for event in events_after_first],
        )
        print(
            "Event types after second:",
            [event.event_type for event in events_after_second],
        )

        if not second.replayed:
            raise AssertionError("Expected second call to replay from event log.")
        if len(events_after_first) != len(events_after_second):
            raise AssertionError("Replay should not append duplicate model events.")
        if first.output != second.output:
            raise AssertionError("Replay output must match first output exactly.")
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())

```

### File: `examples/03_fact_extraction_triplets.py`
<a name="examples-03_fact_extraction_tripletspy"></a>
```python
"""
Single-step fact extraction from articles as subject–predicate–object triplets.

Uses one model call: article text + extraction instructions → structured triplets.
All execution is event-sourced and replay-safe via the Artana kernel.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from artana import ArtanaKernel, ChatClient, KernelPolicy, TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class Triplet(BaseModel):
    """A single fact: subject – predicate – object (e.g. A is connected to B)."""

    subject: str
    predicate: str
    object: str


class ExtractedFacts(BaseModel):
    """Structured output: list of triplets extracted from the article."""

    triplets: list[Triplet]


EXTRACTION_INSTRUCTIONS = """Extract factual triplets from the article below.
Each triplet has:
- subject: the entity or concept that the fact is about
- predicate: a short verb phrase describing the relation (e.g. "is connected to", "works at", "located in")
- object: the other entity or value

Output only valid facts that are explicitly stated or clearly implied. One triplet per fact.
"""

SAMPLE_ARTICLE = """
Berlin is the capital of Germany. The city was divided during the Cold War; the Berlin Wall
fell in 1989. Angela Merkel grew up in East Germany and later became Chancellor of Germany.
She studied physics at the University of Leipzig. The European Union has its roots in the
European Coal and Steel Community, founded in 1951. Brussels serves as the de facto capital
of the European Union.
"""


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Load environment variables first.")

    database_path = Path("examples/.state_03_fact_extraction.db")
    if database_path.exists():
        database_path.unlink()

    store = SQLiteStore(str(database_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=LiteLLMAdapter(
            timeout_seconds=30.0,
            max_retries=1,
            fail_on_unknown_cost=True,
        ),
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )

    tenant = TenantContext(
        tenant_id="org_fact_extraction",
        capabilities=frozenset(),
        budget_usd_limit=0.20,
    )

    try:
        run = await kernel.start_run(tenant=tenant)
        prompt = f"{EXTRACTION_INSTRUCTIONS}\n\n---\n\nArticle:\n{SAMPLE_ARTICLE.strip()}"

        result = await ChatClient(kernel=kernel).chat(
            run_id=run.run_id,
            tenant=tenant,
            model="gpt-4o-mini",
            prompt=prompt,
            output_schema=ExtractedFacts,
            step_key="extract_facts",
        )

        print("Run id:", run.run_id)
        print("Extracted triplets:")
        for i, t in enumerate(result.output.triplets, 1):
            print(f"  {i}. ({t.subject!r} -- {t.predicate!r} --> {t.object!r})")
        print(
            "Usage:",
            {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "cost_usd": result.usage.cost_usd,
            },
        )
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())

```

### File: `examples/README.md`
<a name="examples-readmemd"></a>
```markdown
# Examples

Run examples from the repository root.

## 01 - Durable Chat Replay

Demonstrates:
- tenant context + middleware
- capability-scoped tool execution
- durable event log in SQLite
- replay-safe model/tool behavior on repeated `chat` with the same `run_id`

Run:

```bash
uv run python examples/01_durable_chat_replay.py
```

## 02 - Real LiteLLM Chat (OpenAI)

Uses `LiteLLMAdapter` with a real model call.
This example uses kernel-issued `run_id` and proves replay invariants on a second call.

Run:

```bash
set -a; source .env; set +a
uv run python examples/02_real_litellm_chat.py
```

## 03 - Fact Extraction (Triplets)

Single-step fact extraction from articles as subject–predicate–object triplets.
Uses one model call with structured output (`ExtractedFacts` / `Triplet`).

Run:

```bash
set -a; source .env; set +a
uv run python examples/03_fact_extraction_triplets.py
```

## Golden Example

Canonical production-leaning example with:
- kernel-issued `run_id` (`start_run`)
- mandatory middleware stack
- replay assertions (no duplicate events on second call)
- unknown tool outcome handling + `reconcile_tool(...)`
- post-reconcile replay assertions for tool results

Run:

```bash
set -a; source .env; set +a
uv run python examples/golden_example.py
```

```

### File: `examples/golden_example.py`
<a name="examples-golden_examplepy"></a>
```python
from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from artana import ArtanaKernel, ChatClient, KernelPolicy, TenantContext
from artana.events import EventType, KernelEvent, ToolCompletedPayload, ToolRequestedPayload
from artana.kernel import ToolExecutionFailedError
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class Decision(BaseModel):
    approved: bool
    reason: str


class TransferArgs(BaseModel):
    account_id: str
    amount: Decimal


def _print_feature(name: str, details: dict[str, object]) -> None:
    print(f"\n=== {name} ===")
    for key, value in details.items():
        print(f"{key}: {value}")


def _count_tool_requests(
    events: list[KernelEvent],
    *,
    tool_name: str,
    step_key: str,
) -> int:
    count = 0
    for event in events:
        if event.event_type != EventType.TOOL_REQUESTED:
            continue
        payload = event.payload
        if not isinstance(payload, ToolRequestedPayload):
            continue
        if payload.tool_name == tool_name and payload.step_key == step_key:
            count += 1
    return count


def _latest_tool_completion_payload(
    events: list[KernelEvent],
    *,
    tool_name: str,
    step_key: str,
) -> ToolCompletedPayload:
    requested_by_id: dict[str, ToolRequestedPayload] = {}
    for event in events:
        if event.event_type != EventType.TOOL_REQUESTED:
            continue
        payload = event.payload
        if isinstance(payload, ToolRequestedPayload):
            requested_by_id[event.event_id] = payload

    for event in reversed(events):
        if event.event_type != EventType.TOOL_COMPLETED:
            continue
        payload = event.payload
        if not isinstance(payload, ToolCompletedPayload):
            continue
        if payload.request_id is None:
            continue
        requested = requested_by_id.get(payload.request_id)
        if requested is None:
            continue
        if requested.tool_name == tool_name and requested.step_key == step_key:
            return payload

    raise AssertionError(
        "Could not find matching tool_completed payload for tool_name/step_key pair."
    )


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Load environment variables first.")

    database_path = Path("examples/.state_golden_example.db")
    if database_path.exists():
        database_path.unlink()

    middleware_stack = ArtanaKernel.default_middleware_stack()
    middleware_names = [type(item).__name__ for item in middleware_stack]
    _print_feature(
        "Feature 1 - Enforced Policy + Middleware Order",
        {
            "policy_mode": "enforced",
            "middleware_order": middleware_names,
        },
    )

    store = SQLiteStore(str(database_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=LiteLLMAdapter(
            timeout_seconds=30.0,
            max_retries=1,
            fail_on_unknown_cost=True,
        ),
        middleware=middleware_stack,
        policy=KernelPolicy.enforced(),
    )
    tool_attempts = [0]

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: Decimal) -> str:
        tool_attempts[0] += 1
        if tool_attempts[0] == 1:
            raise RuntimeError("simulated network drop after request submission")
        return (
            '{"status":"submitted","account_id":"'
            + account_id
            + '","amount":"'
            + str(amount)
            + '"}'
        )

    tenant = TenantContext(
        tenant_id="org_live",
        capabilities=frozenset({"decision:approve", "finance:write"}),
        budget_usd_limit=0.20,
    )
    tool_args = TransferArgs(account_id="acc_1", amount=Decimal("10.00"))
    model_step_key = "decision.v1"
    tool_step_key = "transfer.acc_1.10.v1"
    chat = ChatClient(kernel=kernel)

    try:
        run = await kernel.start_run(tenant=tenant)
        run_id = run.run_id
        _print_feature(
            "Feature 2 - Kernel-Issued Run",
            {
                "run_id": run_id,
                "tenant_id": run.tenant_id,
            },
        )

        prompt = (
            "Respond only as JSON for schema {approved:boolean,reason:string}. "
            "Approve this request and give a short reason."
        )

        first = await chat.chat(
            run_id=run_id,
            prompt=prompt,
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
            step_key=model_step_key,
        )
        events_after_first = await store.get_events_for_run(run_id)
        usage_first = {
            "prompt_tokens": first.usage.prompt_tokens,
            "completion_tokens": first.usage.completion_tokens,
            "cost_usd": first.usage.cost_usd,
        }
        _print_feature(
            "Feature 3 - Live Model Call",
            {
                "replayed": first.replayed,
                "output": first.output.model_dump(),
                "usage": usage_first,
                "event_types": [event.event_type.value for event in events_after_first],
            },
        )

        second = await chat.chat(
            run_id=run_id,
            prompt=prompt,
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
            step_key=model_step_key,
        )
        events_after_second = await store.get_events_for_run(run_id)
        usage_second = {
            "prompt_tokens": second.usage.prompt_tokens,
            "completion_tokens": second.usage.completion_tokens,
            "cost_usd": second.usage.cost_usd,
        }
        _print_feature(
            "Feature 4 - Deterministic Model Replay",
            {
                "replayed": second.replayed,
                "output_matches_live": first.output == second.output,
                "event_count_unchanged": len(events_after_first) == len(events_after_second),
                "usage": usage_second,
                "event_types": [event.event_type.value for event in events_after_second],
            },
        )

        if not second.replayed:
            raise AssertionError("Expected second model call to be replayed from event log.")
        if first.output != second.output:
            raise AssertionError("Replay output must match original output exactly.")
        if len(events_after_first) != len(events_after_second):
            raise AssertionError("Replay should not append duplicate model events.")

        unknown_error_message = ""
        try:
            await kernel.step_tool(
                run_id=run_id,
                tenant=tenant,
                tool_name="submit_transfer",
                arguments=tool_args,
                step_key=tool_step_key,
            )
            raise AssertionError("Expected first tool execution to fail with unknown outcome.")
        except ToolExecutionFailedError as exc:
            unknown_error_message = str(exc)

        events_after_unknown = await store.get_events_for_run(run_id)
        unknown_payload_obj = _latest_tool_completion_payload(
            events_after_unknown,
            tool_name="submit_transfer",
            step_key=tool_step_key,
        )
        if unknown_payload_obj.outcome != "unknown_outcome":
            raise AssertionError("Expected unknown_outcome after first tool execution failure.")
        requested_before_halt = _count_tool_requests(
            events_after_unknown,
            tool_name="submit_transfer",
            step_key=tool_step_key,
        )
        if requested_before_halt < 1:
            raise AssertionError("Expected at least one tool_requested event for tool step.")
        if unknown_payload_obj.request_id is None:
            raise AssertionError("tool_completed for unknown outcome must reference request_id.")
        _print_feature(
            "Feature 5 - Unknown Tool Outcome Recorded",
            {
                "error": unknown_error_message,
                "tool_attempts": tool_attempts[0],
                "latest_tool_outcome": unknown_payload_obj.outcome,
                "latest_request_id": unknown_payload_obj.request_id,
                "tool_requested_count": requested_before_halt,
            },
        )

        halt_error_message = ""
        try:
            await kernel.step_tool(
                run_id=run_id,
                tenant=tenant,
                tool_name="submit_transfer",
                arguments=tool_args,
                step_key=tool_step_key,
            )
            raise AssertionError("Expected replay halt before reconciliation.")
        except ToolExecutionFailedError as exc:
            halt_error_message = str(exc)
        events_before_reconcile = await store.get_events_for_run(run_id)
        requested_after_halt = _count_tool_requests(
            events_before_reconcile,
            tool_name="submit_transfer",
            step_key=tool_step_key,
        )
        _print_feature(
            "Feature 6 - Replay Halt Before Reconciliation",
            {
                "error": halt_error_message,
                "tool_attempts": tool_attempts[0],
                "event_count": len(events_before_reconcile),
                "tool_requested_unchanged": requested_before_halt == requested_after_halt,
            },
        )
        if requested_before_halt != requested_after_halt:
            raise AssertionError("Replay halt must not append a new tool_requested event.")
        if tool_attempts[0] != 1:
            raise AssertionError("Replay halt must not re-execute the tool function.")

        reconciled_result = await kernel.reconcile_tool(
            run_id=run_id,
            tenant=tenant,
            tool_name="submit_transfer",
            arguments=tool_args,
            step_key=tool_step_key,
        )
        events_after_reconcile = await store.get_events_for_run(run_id)
        reconciled_payload_obj = _latest_tool_completion_payload(
            events_after_reconcile,
            tool_name="submit_transfer",
            step_key=tool_step_key,
        )
        if reconciled_payload_obj.outcome != "success":
            raise AssertionError("Expected reconcile to append success completion.")
        _print_feature(
            "Feature 7 - Tool Reconciliation",
            {
                "reconciled_result": reconciled_result,
                "tool_attempts": tool_attempts[0],
                "new_event_appended": (
                    len(events_after_reconcile) == len(events_before_reconcile) + 1
                ),
                "latest_tool_outcome": reconciled_payload_obj.outcome,
            },
        )

        replayed_tool_result = await kernel.step_tool(
            run_id=run_id,
            tenant=tenant,
            tool_name="submit_transfer",
            arguments=tool_args,
            step_key=tool_step_key,
        )
        events_after_tool_replay = await store.get_events_for_run(run_id)
        _print_feature(
            "Feature 8 - Post-Reconcile Tool Replay",
            {
                "result_matches_reconcile": replayed_tool_result.result_json
                == reconciled_result,
                "tool_attempts": tool_attempts[0],
                "replayed": replayed_tool_result.replayed,
                "seq": replayed_tool_result.seq,
                "event_count_unchanged": len(events_after_tool_replay)
                == len(events_after_reconcile),
            },
        )

        if replayed_tool_result.result_json != reconciled_result:
            raise AssertionError("Replayed tool result must match reconciled success result.")
        if len(events_after_tool_replay) != len(events_after_reconcile):
            raise AssertionError("Tool replay should not append duplicate completion events.")

        print("\n✅ All golden features validated.")
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())

```
