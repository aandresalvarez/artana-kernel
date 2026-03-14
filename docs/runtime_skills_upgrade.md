# Runtime Skills Upgrade Notes

This note explains what changed when Artana moved from hidden-tool-only progressive skills to filesystem-backed runtime skills.

Use this if you already use `progressive_skills=True` or if you need to understand whether your app must change.

## Summary

The old model was:
- progressive skills mostly meant hidden tools
- `load_skill("tool_name")` revealed a hidden tool schema

The new model adds:
- filesystem-backed `SKILL.md` skills
- reusable instructions loaded from disk
- optional tool bundles attached to a skill
- preloaded skills
- active skill instructions preserved across long runs and compaction

This is an additive change. Existing hidden-tool progressive mode still works.

## What Changed

### New public API

New public types:
- `SkillDefinition`
- `SkillRegistry`
- `FilesystemSkillRegistry`

New optional `ContextBuilder(...)` arguments:
- `skill_registry`
- `allowed_skill_names`
- `preload_skill_names`

Example:

```python
from artana import AutonomousAgent, ContextBuilder, FilesystemSkillRegistry

skill_registry = FilesystemSkillRegistry(["./skills"])

agent = AutonomousAgent(
    kernel=kernel,
    context_builder=ContextBuilder(
        progressive_skills=True,
        skill_registry=skill_registry,
        preload_skill_names=("concise_style",),
    ),
)
```

### New runtime behavior

When a skill registry is configured:
- Artana discovers `SKILL.md` files from disk
- the prompt shows skill summaries before activation
- `load_skill("skill_name")` activates a skill
- bundled tools become visible only after activation
- active skill instructions are re-injected every turn

### New `load_skill(...)` payload shape for registry skills

Registry-backed skills return a different payload shape from legacy hidden tools.

Successful registry skill load:

```json
{
  "name": "demo_reader",
  "kind": "registry_skill",
  "loaded": true,
  "summary": "Unlock the demo file reader.",
  "instructions_markdown": "Use read_demo_file once this skill is active.",
  "tool_names": ["read_demo_file"]
}
```

Legacy hidden-tool load still returns the old schema-oriented payload.

### New failure mode for broken skills

In addition to `forbidden_skill`, registry-backed skills can now return:
- `invalid_skill`: the skill references a bundled tool that is not registered

Registry skills use all-or-nothing activation:
- if one bundled tool is missing, nothing from that skill is revealed

### New run summaries

Agent runs now emit an additive summary for active skills:
- summary type: `agent_active_skills`

This does not break existing runs; it only adds more trace data.

## What Stayed Compatible

These surfaces did not break:
- kernel APIs
- harness APIs
- normal `AutonomousAgent` usage without a skill registry
- legacy hidden-tool progressive mode
- `load_skill("tool_name")` for old hidden-tool behavior

If you do not configure `skill_registry`, your existing setup should behave as before.

## Who Needs To Change Code

You need to change code only if you want the new filesystem-backed skill feature.

You likely need no code changes if:
- you do not use progressive skills
- you use progressive skills only for hidden tools
- you do not parse `load_skill(...)` payloads directly

You likely need code changes if:
- you want skills from `SKILL.md` files
- you want preload or allowlist behavior
- you parse `load_skill(...)` results programmatically

## Migration Guidance

### If you use old hidden-tool progressive skills only

You can do nothing.

Your current behavior remains valid:
- hidden tools stay hidden until loaded
- `load_skill("tool_name")` still works

### If you want filesystem-backed skills

1. Create a skill folder, for example `./skills`.
2. Add one or more `SKILL.md` files.
3. Create `FilesystemSkillRegistry(["./skills"])`.
4. Pass it into `ContextBuilder(skill_registry=...)`.
5. Optionally add `preload_skill_names` or `allowed_skill_names`.

### If you parse `load_skill(...)` results

Update your code to handle two cases:
- legacy hidden tool payloads
- registry skill payloads with `kind: "registry_skill"`

## Best-Practice Rollout

Recommended rollout:
- start with one instruction-only skill
- then add one bundled-tool skill
- verify expected `forbidden_skill` and `invalid_skill` behavior in a local-first test

Do not:
- assume all skills are bundled-tool skills
- assume `load_skill(...)` always returns a schema payload
- bundle unstable or environment-specific tools into production skills without validation

## Related Docs

- [Runtime skills usage guide](./runtime_skills.md)
- [Chapter 3](./Chapter3.md)
- [Examples README](../examples/README.md)
- [Local-first runtime skills example](../examples/15_file_backed_skills.py)
