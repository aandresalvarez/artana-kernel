from __future__ import annotations

import json
import os


def require_openai_api_key(*, script_name: str) -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "OPENAI_API_KEY is required for this live example.\n"
        f"Script: examples/{script_name}\n"
        "Quick start:\n"
        "  1) Add OPENAI_API_KEY=... to .env\n"
        "  2) Load env vars: set -a; source .env; set +a\n"
        f"  3) Re-run: uv run python examples/{script_name}"
    )


def resolve_model(*, env_var: str, default: str) -> str:
    return os.getenv(env_var, default)


def print_example_header(*, title: str, models: dict[str, str]) -> None:
    print(f"=== {title} ===")
    print("Models:")
    for name, model in models.items():
        print(f"  - {name}: {model}")


def print_summary(*, payload: dict[str, object]) -> None:
    print("\nSummary:")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def friendly_exit(*, script_name: str, error: Exception) -> SystemExit:
    return SystemExit(
        f"{script_name} failed: {error}\n\n"
        "Troubleshooting:\n"
        "- Confirm OPENAI_API_KEY is set and loaded.\n"
        "- Verify your account can access the selected model(s).\n"
        "- Re-run after loading env vars: set -a; source .env; set +a"
    )
