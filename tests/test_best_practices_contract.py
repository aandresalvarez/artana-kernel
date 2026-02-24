from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

from artana.kernel import ArtanaKernel
from artana.middleware import order_middleware
from artana.middleware.base import ModelInvocation, PreparedToolRequest
from artana.middleware.capability_guard import CapabilityGuardMiddleware
from artana.middleware.pii_scrubber import PIIScrubberMiddleware
from artana.middleware.quota import QuotaMiddleware
from artana.middleware.safety_policy import SafetyPolicyMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelUsage
from artana.safety import SafetyPolicyConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "artana"


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def test_no_typing_any_in_src_artana() -> None:
    violations: list[str] = []

    for file_path in _python_files(SRC_ROOT):
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        typing_aliases: set[str] = set()
        any_aliases: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"typing", "typing_extensions"}:
                        typing_aliases.add(alias.asname or alias.name)
            if isinstance(node, ast.ImportFrom) and node.module in {"typing", "typing_extensions"}:
                for alias in node.names:
                    if alias.name == "Any":
                        imported_name = alias.asname or "Any"
                        any_aliases.add(imported_name)
                        violations.append(
                            f"{_rel(file_path)}:{node.lineno}: import of {node.module}.Any"
                        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in any_aliases:
                violations.append(f"{_rel(file_path)}:{node.lineno}: usage of {node.id}")
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "Any"
                and isinstance(node.value, ast.Name)
                and node.value.id in typing_aliases
            ):
                violations.append(
                    f"{_rel(file_path)}:{node.lineno}: usage of {node.value.id}.Any"
                )

    assert not violations, "typing.Any is not allowed in src/artana:\n" + "\n".join(
        sorted(set(violations))
    )


def test_agent_and_harness_do_not_access_kernel_private_ports() -> None:
    private_access = re.compile(r"(?:\bkernel|\bself\.kernel)\._(?:store|model_port)\b")
    violations: list[str] = []

    for root in (SRC_ROOT / "agent", SRC_ROOT / "harness"):
        for file_path in _python_files(root):
            lines = file_path.read_text(encoding="utf-8").splitlines()
            for lineno, line in enumerate(lines, start=1):
                if private_access.search(line):
                    violations.append(f"{_rel(file_path)}:{lineno}: {line.strip()}")

    assert not violations, "Agent/Harness must not bypass kernel ports:\n" + "\n".join(violations)


class _CustomMiddleware:
    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        return invocation

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        return None

    async def after_model(self, *, run_id: str, tenant: TenantContext, usage: ModelUsage) -> None:
        return None

    async def prepare_tool_request(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments_json: str,
    ) -> str | PreparedToolRequest:
        return arguments_json

    async def prepare_tool_result(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        result_json: str,
    ) -> str:
        return result_json


def test_middleware_order_is_absolute() -> None:
    ordered = order_middleware(
        (
            _CustomMiddleware(),
            SafetyPolicyMiddleware(config=SafetyPolicyConfig()),
            CapabilityGuardMiddleware(),
            PIIScrubberMiddleware(),
            QuotaMiddleware(),
        )
    )
    ordered_names = [type(item).__name__ for item in ordered]
    assert ordered_names == [
        "PIIScrubberMiddleware",
        "QuotaMiddleware",
        "CapabilityGuardMiddleware",
        "SafetyPolicyMiddleware",
        "_CustomMiddleware",
    ]


def test_default_middleware_stack_has_required_baseline() -> None:
    stack = ArtanaKernel.default_middleware_stack()
    names = [type(item).__name__ for item in stack]
    assert names[:3] == [
        "PIIScrubberMiddleware",
        "QuotaMiddleware",
        "CapabilityGuardMiddleware",
    ]


def test_pre_commit_config_contains_strict_hooks() -> None:
    config_path = REPO_ROOT / ".pre-commit-config.yaml"
    assert config_path.exists(), ".pre-commit-config.yaml must exist."
    config_text = config_path.read_text(encoding="utf-8")
    required_entries = (
        "uv run ruff check .",
        "uv run mypy --strict src tests",
        "uv run pytest",
        "uv run python scripts/generate_kernel_behavior_index.py --check",
    )
    for entry in required_entries:
        assert entry in config_text
    assert config_text.count("pass_filenames: false") >= 4


def test_doc_validation_artifacts_are_ignored_and_untracked() -> None:
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".tmp_doc_validation/" in gitignore_text

    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        return
    if shutil.which("git") is None:
        return
    result = subprocess.run(
        ["git", "ls-files", "--", ".tmp_doc_validation"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_side_effect_examples_use_tool_execution_context_idempotency() -> None:
    checks = {
        REPO_ROOT / "examples" / "01_durable_chat_replay.py": "submit_transfer",
        REPO_ROOT / "examples" / "golden_example.py": "submit_transfer",
    }
    for file_path, function_name in checks.items():
        content = file_path.read_text(encoding="utf-8")
        signature_pattern = re.compile(
            rf"async def {function_name}\([^)]*artana_context:\s*ToolExecutionContext",
            re.DOTALL,
        )
        assert signature_pattern.search(
            content
        ), f"Missing ToolExecutionContext in {_rel(file_path)}"
        assert "artana_context.idempotency_key" in content
