Here is the detailed Product Requirements Document (PRD) to build the **Experience Engine (Inter-Run Learning Memory)** into Artana. 

This PRD ports the best ideas from your old `flujo_hybrid_reasoner` (Win-Patterns, Anti-Patterns, Rule extraction) and seamlessly integrates them into the clean, new Artana architecture.

***

# PRD: Artana Experience Engine (Inter-Run Memory)

## 1. Overview & Problem Statement
Currently, Artana agents possess **Intra-Run Memory** (conversation history and explicit memory tools within a single `run_id`). However, they are **Inter-Run Amnesiacs**. 

If an agent performs a "Weekly Financial Extraction" task and fails 3 times because an API requires a specific date format, it will eventually self-correct and succeed. But next Friday, under a new `run_id`, it starts completely blank and makes the exact same 3 mistakes.

**The Objective:** We need to provide a mechanism for agents to extract "Learnings" (Rules, Win-Patterns, Anti-Patterns) at the end of a run, and automatically inject those learnings into future runs of the same task type. This creates an **Adaptive AI Runtime** that gets cheaper, faster, and more reliable over time.

## 2. Architectural Principles
To maintain Artana's strict separation of concerns, the Experience Engine must adhere to these rules:
1.  **Kernel Purity:** The core `ArtanaKernel` will *not* know about the Experience Engine. It remains a pure I/O and governance engine.
2.  **Standalone Utility:** The `ExperienceStore` must be a standalone Python utility so it can be used inside strict Enterprise Workflows (`ctx.step`).
3.  **Opt-In Automation:** The `AutonomousAgent` will accept the store via `ContextBuilder` to provide "magic" zero-config learning for developers who want it.
4.  **Strict Tenant Isolation:** All memories must be strictly partitioned by `tenant_id` so User A's agent never learns from User B's secure data.

---

## 3. Data Model & Schemas

All types will be strictly defined using Pydantic (No `Any`).

```python
from enum import Enum
from pydantic import BaseModel, Field

class RuleType(str, Enum):
    WIN_PATTERN = "win_pattern"     # "Always format dates as YYYY-MM-DD"
    ANTI_PATTERN = "anti_pattern"   # "Do not use the /v1/users endpoint, it times out"
    FACT = "fact"                   # "The CEO of Acme is Jane Doe"

class ExperienceRule(BaseModel):
    rule_id: str
    tenant_id: str
    task_category: str
    rule_type: RuleType
    content: str
    success_count: int = 0
    fail_count: int = 0

class ReflectionResult(BaseModel):
    extracted_rules: list[ExperienceRule]
```

---

## 4. Developer API (The DX)

We must support both Artana paradigms: the Autonomous Agent (Steering Wheel) and the Durable Workflow (Manual Control).

### Paradigm A: The Autonomous Agent (Auto-Learning)
Developers pass the store to the `ContextBuilder` and enable `auto_reflect`. The agent handles the rest.

```python
from artana.agent.experience import SQLiteExperienceStore
from artana.agent import ContextBuilder, AutonomousAgent

# 1. Boot the Store
exp_store = SQLiteExperienceStore("tenant_experience.db")

# 2. Configure Context Builder to fetch rules for "Financial_Reporting"
ctx_builder = ContextBuilder(
    identity="You are a financial agent.",
    experience_store=exp_store,
    task_category="Financial_Reporting" 
)

# 3. Boot Agent with Auto-Reflection
agent = AutonomousAgent(
    kernel=kernel,
    context_builder=ctx_builder,
    auto_reflect=True # Post-run, the agent extracts and saves rules!
)

# Run 1: Agent struggles, figures it out, saves rule.
# Run 2: Agent reads rule in System Prompt, succeeds instantly.
await agent.run(...) 
```

### Paradigm B: The Durable Workflow (Manual Control)
For strict pipelines, developers import the store and query/update it explicitly as deterministic steps.

```python
async def smart_enterprise_workflow(ctx: WorkflowContext):
    
    # Step 1: Deterministic retrieval of past rules
    async def fetch_rules():
        return exp_store.get_rules(ctx.tenant.tenant_id, "Financial_Reporting")
        
    past_rules = await ctx.step(name="fetch_rules", action=fetch_rules, serde=...)
    
    # Step 2: Inject rules into the prompt
    sys_prompt = f"Follow these rules strictly:\n{past_rules}"
    result = await chat.chat(..., system_prompt=sys_prompt, step_key="extract")
    
    # Step 3: Explicit reflection step at the end of the workflow
    async def reflect_and_learn():
        reflection = await chat.chat(
            prompt="What did we learn from this extraction? Return rules.",
            output_schema=ReflectionResult
        )
        exp_store.save_rules(reflection.output.extracted_rules)
        return "Learnings updated."

    await ctx.step(name="learn", action=reflect_and_learn, serde=...)
    return result
```

---

## 5. Execution Plan (Implementation Milestones)

### Milestone 1: The `ExperienceStore` Interface
**File:** `src/artana/agent/experience.py`
*   Create the `ExperienceStore` protocol.
*   Implement `SQLiteExperienceStore`. 
*   **Schema:** `CREATE TABLE experience_rules (rule_id TEXT PRIMARY KEY, tenant_id TEXT, task_category TEXT, rule_type TEXT, content TEXT, success_count INT, fail_count INT)`
*   **Methods:** 
    *   `save_rules(rules: list[ExperienceRule]) -> None`
    *   `get_rules(tenant_id: str, task_category: str, limit: int = 10) -> list[ExperienceRule]`
    *   *(Optional)* `reinforce_rule(rule_id: str, positive: bool)` (To increment success/fail counts).

### Milestone 2: Update `ContextBuilder`
**File:** `src/artana/agent/context.py`
*   Update `ContextBuilder.__init__` to accept `experience_store` and `task_category`.
*   Update `build_messages()`. If the store and category are present, query the store.
*   Format the rules into a string block and append it to the `system` message:
    ```text
    [PAST LEARNINGS FOR THIS TASK]
    WIN_PATTERN: Always format dates as YYYY-MM-DD.
    ANTI_PATTERN: Do not query the Q3 table without a region ID.
    ```

### Milestone 3: Reflection Logic in `AutonomousAgent`
**File:** `src/artana/agent/autonomous.py`
*   Add an `auto_reflect: bool` flag to `__init__`.
*   Create a private `_run_reflection(run_id, messages)` method.
*   At the end of the `run()` loop (right before returning `model_result.output`), if `auto_reflect` is true:
    1.  Take the conversation history.
    2.  Make a fast `step_model` call (e.g., using `gpt-4o-mini`) requesting the `ReflectionResult` schema.
    3.  Save the `extracted_rules` to the `ExperienceStore`.
*   *Safety Note:* Ensure the reflection step has its own `step_key` (e.g., `turn_{iteration}_reflection`) so it is fully deterministic and replayable.

### Milestone 4: Tests & Examples
*   Write `test_experience_store.py` to verify tenant isolation (User A cannot read User B's rules).
*   Create a new example: `examples/07_adaptive_agent_learning.py` demonstrating the agent failing on Run 1, and succeeding immediately on Run 2 because of the injected rules.