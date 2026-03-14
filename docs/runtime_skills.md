# Runtime Skills

Runtime skills let Artana load reusable agent guidance from `SKILL.md` files on disk.

Use them when you want:
- reusable instructions without copying large prompt blocks into Python
- progressive tool exposure, where tools stay hidden until a skill is active
- stable agent behavior across long runs, including after compaction

## Mental Model

Think of a runtime skill as a small playbook:
- metadata in YAML frontmatter
- instructions in Markdown
- optional bundled tools that become visible only after loading the skill

The basic flow is:
1. Put `SKILL.md` files in one or more skill folders.
2. Create a `FilesystemSkillRegistry(...)` pointing at those folders.
3. Pass that registry into `ContextBuilder(...)`.
4. Let the agent discover and `load_skill("skill_name")` when needed.
5. Artana activates the skill instructions and reveals any bundled tools on the next turn.

## Recommended Folder Layout

Artana scans the configured roots recursively and treats every file named `SKILL.md` as a skill.

Example layout:

```text
my_app/
  skills/
    writing/
      concise_style/
        SKILL.md
    research/
      source_grounding/
        SKILL.md
    support/
      refund_policy/
        SKILL.md
```

Important rules:
- folder names are for humans only; Artana identifies the skill by the `name:` inside `SKILL.md`
- skill names must be unique across all configured roots
- nested folders are fine because discovery is recursive

## `SKILL.md` Format

Every skill file must start with YAML frontmatter.

Instruction-only skill:

```md
---
name: concise_style
version: 1.0.0
summary: Keep responses short and grounded.
---

Keep responses terse and grounded.
Prefer concrete answers over filler.
```

Skill with bundled tools:

```md
---
name: demo_reader
version: 1.0.0
summary: Unlock the demo file reader.
tools:
  - read_demo_file
---

Use `read_demo_file` once this skill is active.
Base claims only on the file contents.
```

Supported fields:
- `name` (required): unique skill identifier
- `version` (required): human-maintained version string
- `summary` (required): lightweight description shown before loading
- `tools` (optional): tool names this skill unlocks
- `requires_capabilities` (optional): tenant capabilities required to load the skill
- `tags` (optional): free-form labels for organization

The Markdown body becomes `instructions_markdown`.

## Wiring It Into an Agent

```python
from artana import AutonomousAgent, ContextBuilder, FilesystemSkillRegistry

skill_registry = FilesystemSkillRegistry(["./skills"])

context_builder = ContextBuilder(
    progressive_skills=True,
    skill_registry=skill_registry,
    preload_skill_names=("concise_style",),
)

agent = AutonomousAgent(
    kernel=kernel,
    context_builder=context_builder,
)
```

Behavior:
- `skill_registry`: where Artana discovers filesystem-backed skills
- `preload_skill_names`: skills that should be active on the first turn
- `allowed_skill_names`: optional extra allowlist for registry skills

If you use `StrongModelAgentHarness` or a domain harness built on it, the same `ContextBuilder(...)` configuration flows through automatically because those harnesses run `AutonomousAgent` internally.

## What Happens at Runtime

When `progressive_skills=True`:
- the agent sees a lightweight skill panel in the prompt
- registry-backed skills appear by summary before they are active
- the agent calls `load_skill(skill_name="...")` to activate one

For registry-backed skills:
- successful loads return instructions and bundled tool names
- bundled tools become visible on the next model turn
- active skill instructions are re-injected every turn so compaction does not erase them

Legacy hidden-tool progressive mode still works:
- if no registry skill matches, `load_skill(...)` falls back to the older hidden-tool schema reveal behavior

## Best Practices

Use instruction-only skills when:
- you want reusable behavior or style guidance
- no new tools are needed
- the skill is mostly “how to think” or “how to write”

Use bundled-tool skills when:
- a set of tools only makes sense in a specific mode
- you want to keep the default tool surface small
- the skill should unlock both instructions and capabilities together

Organize skills by domain, not by agent class:
- `skills/research/...`
- `skills/coding/...`
- `skills/support/...`

Keep each skill focused:
- one skill should represent one mode or responsibility
- avoid giant “do everything” skills

Write summaries for discovery:
- summaries should tell the model when the skill is useful
- keep them short and concrete

Prefer stable tool bundles:
- if a tool is optional or experimental, do not bundle it into production skills
- a missing bundled tool makes the whole skill invalid

Use preloads sparingly:
- preload only skills that should always be active for that agent
- otherwise let the model load skills progressively to keep the prompt smaller

## Failure Modes

`load_skill(...)` can fail in a few important ways:
- `unknown_skill`: no registry skill or legacy hidden tool matched the name
- `forbidden_skill`: tenant capability checks or skill allowlist blocked the load
- `invalid_skill`: the skill references a bundled tool that is not registered

Artana uses all-or-nothing activation for registry skills:
- if one bundled tool is missing, the skill does not partially load
- if bundled tools are blocked by capability rules, the skill does not partially load

## Common Mistakes

Do not:
- assume the folder name is the skill ID; the `name:` field is the real identifier
- duplicate skill names across roots
- bundle tools that might not be registered in the target environment
- preload a skill that is not allowed or not present in the registry

Do:
- keep skill names stable
- keep frontmatter small and valid YAML
- test a new skill with a local-first example before using it in a production harness

## Example

See the local-first walkthrough in [examples/15_file_backed_skills.py](../examples/15_file_backed_skills.py).
