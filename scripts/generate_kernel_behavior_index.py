#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
OUTPUT_PATH = REPO_ROOT / "docs" / "kernel_behavior_index.json"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artana.events import EventType  # noqa: E402


def build_behavior_index() -> dict[str, object]:
    return {
        "schema_version": 1,
        "replay_policies": [
            "strict",
            "allow_prompt_drift",
            "fork_on_drift",
        ],
        "event_types": [event_type.value for event_type in EventType],
        "model_requested_invariants": {
            "required_fields": [
                "model",
                "prompt",
                "messages",
                "allowed_tools",
                "allowed_tool_signatures",
                "allowed_tools_hash",
                "step_key",
                "model_cycle_id",
                "context_version",
            ],
            "tool_signature_token_format": (
                "name|tool_version|schema_version|schema_hash"
            ),
            "context_version_fields": [
                "system_prompt_hash",
                "context_builder_version",
                "compaction_version",
            ],
        },
        "model_terminal_invariants": {
            "outcomes": [
                "completed",
                "failed",
                "timeout",
                "cancelled",
                "abandoned",
            ],
            "required_correlation_fields": [
                "model_cycle_id",
                "source_model_requested_event_id",
            ],
        },
        "tool_io_hooks": [
            "prepare_tool_request(run_id, tenant, tool_name, arguments_json)",
            "prepare_tool_result(run_id, tenant, tool_name, result_json)",
        ],
        "run_summary_types": [
            "agent_model_step",
            "agent_tool_step",
            "capability_decision",
        ],
    }


def render_behavior_index() -> str:
    return json.dumps(build_behavior_index(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if docs/kernel_behavior_index.json is out of date",
    )
    args = parser.parse_args()

    rendered = render_behavior_index()
    if args.check:
        existing = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if existing != rendered:
            print(
                "kernel_behavior_index.json is out of date. "
                "Run scripts/generate_kernel_behavior_index.py",
                file=sys.stderr,
            )
            return 1
        return 0

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
