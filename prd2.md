 Best API proposal (single option)

**Make the kernel step-oriented, not chat-oriented.**
`chat()` becomes a thin convenience wrapper in an optional “agent” layer, not the kernel’s core primitive.

## Public API

### 1) Run lifecycle (first-class)

* `await kernel.start_run(tenant, run_id: str | None = None) -> RunRef`
* `await kernel.load_run(run_id) -> RunRef`
* `await kernel.resume(run_id, tenant, human_input) -> RunRef` *(human is just one boundary type)*

### 2) Deterministic execution primitives (kernel core)

* `await kernel.step_model(run_id, tenant, model, input: ModelInput, output_schema, step_key: str | None = None) -> StepResult`
* `await kernel.step_tool(run_id, tenant, tool_name, arguments: BaseModel, step_key: str | None = None) -> ToolResult`
* `await kernel.pause(run_id, tenant, reason, context: BaseModel | None = None, step_key: str | None = None) -> PauseTicket`

### 3) Model input is neutral (no “chat” baked in)

* `ModelInput(kind="prompt", prompt=...)`
* `ModelInput(kind="messages", messages=[...])`

### 4) Replay & idempotency are explicit

* Every step returns `replayed: bool`, plus `seq`
* Tool steps enforce two-phase events (`TOOL_REQUESTED` then `TOOL_COMPLETED`) and idempotency keys derived from `(run_id, seq)` or `(run_id, step_key)`.

### 5) Chat lives outside the kernel

* `artana.agent.ChatClient.chat(...)` wraps `kernel.step_model(...)`
* Kernel stays an execution runtime; agent SDK provides chat ergonomics.

### 6) Enforced mode (authoritative boundary)

* `ArtanaKernel(..., enforced=True)` fails fast if required middleware/policies aren’t present (quota/capability/pii) and if pricing/cost is unknown (never silently “0.0”).

**Net effect:** Artana’s core primitive becomes “durable, governed step execution” (model/tool/pause), which is defensible long-term and works for non-chat agents, batch jobs, autonomous workflows, and Temporal integration.
