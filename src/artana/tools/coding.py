from __future__ import annotations

import json
import subprocess
from pathlib import Path

from artana.ports.tool import LocalToolRegistry, ToolExecutionContext


class CodingHarnessTools:
    def __init__(self, *, sandbox_root: str) -> None:
        self._sandbox_root = Path(sandbox_root).expanduser().resolve()
        self._sandbox_root.mkdir(parents=True, exist_ok=True)
        self._registry = LocalToolRegistry()
        self._register_tools()

    def registry(self) -> LocalToolRegistry:
        return self._registry

    def _register_tools(self) -> None:
        self._registry.register(
            self.create_worktree,
            requires_capability="coding:worktree",
            side_effect=True,
            risk_level="medium",
        )
        self._registry.register(
            self.read_file,
            requires_capability="coding:read",
            risk_level="low",
        )
        self._registry.register(
            self.apply_patch,
            requires_capability="coding:write",
            side_effect=True,
            risk_level="high",
        )
        self._registry.register(
            self.git_diff,
            requires_capability="coding:read",
            risk_level="low",
        )

    async def create_worktree(
        self,
        artana_context: ToolExecutionContext,
        name: str | None = None,
    ) -> str:
        token = name or artana_context.idempotency_key
        safe_name = _normalize_name(token)
        worktree_path = self._sandbox_root / safe_name
        worktree_path.mkdir(parents=True, exist_ok=True)
        return json.dumps(
            {
                "worktree": safe_name,
                "path": str(worktree_path),
                "created": True,
            },
            ensure_ascii=False,
        )

    async def read_file(self, path: str) -> str:
        resolved = self._resolve_path(path)
        if not resolved.exists() or not resolved.is_file():
            return json.dumps(
                {
                    "ok": False,
                    "error": "file_not_found",
                    "path": str(resolved),
                },
                ensure_ascii=False,
            )
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return json.dumps(
                {
                    "ok": False,
                    "error": "invalid_utf8",
                    "path": str(resolved),
                },
                ensure_ascii=False,
            )
        except OSError:
            return json.dumps(
                {
                    "ok": False,
                    "error": "read_failed",
                    "path": str(resolved),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "ok": True,
                "path": str(resolved),
                "content": content,
            },
            ensure_ascii=False,
        )

    async def apply_patch(
        self,
        path: str,
        content: str,
        artana_context: ToolExecutionContext,
    ) -> str:
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return json.dumps(
            {
                "ok": True,
                "path": str(resolved),
                "bytes_written": len(content.encode("utf-8")),
                "idempotency_key": artana_context.idempotency_key,
            },
            ensure_ascii=False,
        )

    async def git_diff(self, path: str | None = None) -> str:
        args = ["git", "-C", str(self._sandbox_root), "diff", "--"]
        if path is not None:
            args.append(path)
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
        )
        return json.dumps(
            {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            ensure_ascii=False,
        )

    def _resolve_path(self, path: str) -> Path:
        candidate = (self._sandbox_root / path).resolve()
        if not _is_relative_to(candidate, self._sandbox_root):
            raise ValueError("Path escapes sandbox root.")
        return candidate


def _normalize_name(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if char.isalnum() or char in {"_", "-"}:
            chars.append(char)
        else:
            chars.append("_")
    normalized = "".join(chars).strip("_")
    return normalized if normalized != "" else "worktree"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = ["CodingHarnessTools"]
