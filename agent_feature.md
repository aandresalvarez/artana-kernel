Here is the detailed Feature Specification for the next phase of development. 

This document outlines the capabilities required to upgrade `AutonomousAgent` from a simple `while` loop into a **state-of-the-art, long-running agent runtime** (matching the capabilities of OpenAI Codex and nanobot), while preserving the strict safety of the Artana Kernel.

***

# Feature Specification: Artana Agent Runtime (v1.1)

## Executive Summary
The `ArtanaKernel` successfully provides a durable, policy-enforced, event-sourced OS layer. However, the current `AutonomousAgent` implementation is a "naive" loop. If left running, it will eventually crash due to LLM context window exhaustion, and injecting dozens of tools will consume excessive tokens. 

To support **long-running, enterprise-grade autonomous agents** (agents that run for days, process thousands of events, and manage complex swarms), we need to build advanced Context, Memory, and Skill management into the `artana.agent` layer. **None of these features will touch the core `_kernel`**, ensuring the execution engine remains purely focused on atomic durability and governance.

---

## Feature 1: Context Compaction (Token Window Management)

**The Problem:** The agent loop currently appends every thought, tool call, and tool result to the `messages` array. In a long-running task, this will trigger a `context_length_exceeded` error from the LLM provider, crashing the agent.

**The Solution:** The `AutonomousAgent` must proactively monitor and compress its own history.

*   **Token Estimation:** Integrate `litellm.token_counter` (or a rough character heuristic) to track the size of the `messages` array.
*   **The Compaction Trigger:** When the context reaches a defined threshold (e.g., 80% of the model's max context, or `max_history_messages = 40`), the agent pauses its main task to run a compaction step.
*   **The Summarization Step:** The agent calls `kernel.step_model()` using a fast/cheap model (e.g., `gpt-4o-mini`) with a prompt to summarize the oldest N messages.
*   **Message Replacement:** The agent splices the `messages` array, replacing the old block of messages with a single `ChatMessage(role="system", content="Past Events Summary: [...]")`.

*Architecture Note:* Because the summary is generated via `kernel.step_model()`, the compaction event is perfectly recorded in the SQLite event ledger. Replays remain 100% deterministic.

---

## Feature 2: Progressive Skill Disclosure (Codex/Nanobot Style)

**The Problem:** If an enterprise Tenant has 50 tools registered (e.g., Jira, GitHub, Slack, Postgres), injecting 50 large JSON schemas into the system prompt wastes thousands of tokens per turn and confuses the LLM's attention.

**The Solution:** "Skills as files" / Progressive Loading.

*   **Skill Metadata:** The agent initially provides the LLM with only a lightweight text summary of available tools (e.g., *"Available Skills: `db_query` (Query the database), `jira_create` (Create a ticket)."*).
*   **The `load_skill` Meta-Tool:** We provide the agent with a built-in tool called `load_skill(skill_name: str)`. 
*   **Just-in-Time Schema Injection:** When the LLM realizes it needs to query the database, it calls `load_skill("db_query")`. The tool returns the full instructions, exact JSON schema, and usage examples for that specific tool, which are then added to the context window.
*   **Execution:** On the *next* turn, the LLM now has the context required to actually call the `db_query` tool. 

---

## Feature 3: Long-Term Memory Consolidation

**The Problem:** As Context Compaction (Feature 1) runs, granular details are lost. If an agent learns a crucial user preference or discovers a fact on Turn 5, it might be summarized away by Turn 50.

**The Solution:** An explicit `MemoryStore` that exists outside the prompt window.

*   **Core Memory Tools:** Provide the agent with default tools: `core_memory_append`, `core_memory_replace`, and `core_memory_search`.
*   **Asynchronous Consolidation:** Allow the agent to explicitly write facts to a "Scratchpad" or "Memory File" (stored in SQLite or a simple JSON blob mapped to the `run_id`).
*   **State Injection:** Every time the `ContextBuilder` constructs the prompt for a new turn, it retrieves the current state of the Long-Term Memory and injects it at the top of the system prompt.

---

## Feature 4: Sub-Agent Delegation (The Swarm Protocol)

**The Problem:** A single agent trying to write code, search the web, and review legal documents will suffer from persona dilution and hallucinate.

**The Solution:** First-class support for spawning Sub-Agents as Tools.

*   **The `SubAgentFactory`:** A utility within the `artana.agent` module that easily wraps a new `AutonomousAgent` inside a Tool.
*   **Context/Budget Inheritance:** The factory automatically passes the `TenantContext` down to the sub-agent, ensuring the child agent shares the exact same `$USD` budget cap and is subjected to the same PII scrubbing rules.
*   **Run ID Lineage:** Sub-agents automatically generate linked `run_id`s (e.g., `parent_run_id::sub_agent::idempotency_key`), allowing an auditor to query the SQLite database and perfectly reconstruct the tree of spawned agents.

---

## Feature 5: The `ContextBuilder` Pipeline

**The Problem:** Currently, the `messages` array is initialized with a static `system_prompt`. This is too rigid for complex agents that need to load bootstrap files, identity instructions, and memory on the fly.

**The Solution:** A dedicated `ContextBuilder` class that executes before every `step_model` call.

*   **Dynamic Assembly:** The `ContextBuilder` is responsible for compiling the final prompt. It dynamically concatenates:
    1.  **Identity:** (e.g., "You are a senior DevOps engineer.")
    2.  **Long-Term Memory:** (e.g., "User prefers Python. API key format is X.")
    3.  **Progressive Skills List:** (e.g., "Available tools: [ping, restart_server]")
    4.  **Short-Term Conversation History:** (The compacted sliding window).
*   **Extensibility:** Developers can subclass `ContextBuilder` to inject their own dynamic data (like fetching real-time RAG context from a vector database) without polluting the `AutonomousAgent` loop logic.

---

## Proposed Developer API (How it will look)

Here is how a developer will initialize this next-generation agent:

```python
from artana.agent import AutonomousAgent, ContextBuilder, CompactionStrategy
from artana.agent.memory import SQLiteMemoryStore

# 1. Setup external memory
memory_store = SQLiteMemoryStore(db_path="agent_memory.db")

# 2. Configure the Context Builder
context_builder = ContextBuilder(
    identity="You are a senior data analyst.",
    memory_store=memory_store,
    progressive_skills=True # Enables the load_skill meta-tool
)

# 3. Boot the Agent with Compaction
agent = AutonomousAgent(
    kernel=kernel,
    context_builder=context_builder,
    compaction=CompactionStrategy(
        trigger_at_messages=40,
        summarize_with_model="gpt-4o-mini",
        keep_recent_messages=10
    )
)

# 4. Run!
result = await agent.run(
    run_id="analyst_run_001",
    tenant=tenant,
    model="gpt-4o",
    prompt="Analyze the Q3 sales data and write a report.",
    output_schema=ReportSchema
)
```

## Implementation Phases

*   **Phase 1: ContextBuilder & Compaction.** Solve the crash-on-length problem first. Implement token counting and the LLM summarization step.
*   **Phase 2: Long-Term Memory Tools.** Implement the basic read/write memory tools so the agent can survive compaction without amnesia.
*   **Phase 3: Progressive Skills.** Implement the schema-hiding logic and the `load_skill` tool to save tokens on large enterprise tool registries.
*   **Phase 4: Swarm Helpers.** Provide the clean wrapper for `SubAgentFactory` using the existing `ToolExecutionContext`.