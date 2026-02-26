from __future__ import annotations

import json
from pathlib import Path

from artana.ports.tool import LocalToolRegistry


class ObservabilityTools:
    def __init__(self, *, root: str) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._registry = LocalToolRegistry()
        self._register_tools()

    def registry(self) -> LocalToolRegistry:
        return self._registry

    def _register_tools(self) -> None:
        self._registry.register(
            self.query_logs,
            requires_capability="observability:logs",
            risk_level="low",
        )
        self._registry.register(
            self.query_metrics,
            requires_capability="observability:metrics",
            risk_level="low",
        )

    async def query_logs(self, file: str, limit: int = 200) -> str:
        if limit <= 0:
            raise ValueError("limit must be > 0")
        resolved = self._resolve_path(file)
        if not resolved.exists() or not resolved.is_file():
            return json.dumps(
                {"ok": False, "error": "file_not_found", "path": str(resolved)},
                ensure_ascii=False,
            )
        try:
            lines = resolved.read_text(encoding="utf-8").splitlines()
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
                "lines": lines[-limit:],
            },
            ensure_ascii=False,
        )

    async def query_metrics(self, file: str) -> str:
        resolved = self._resolve_path(file)
        if not resolved.exists() or not resolved.is_file():
            return json.dumps(
                {"ok": False, "error": "file_not_found", "path": str(resolved)},
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
        try:
            parsed: object = json.loads(content)
        except json.JSONDecodeError:
            return json.dumps(
                {
                    "ok": False,
                    "error": "invalid_json",
                    "path": str(resolved),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "ok": True,
                "path": str(resolved),
                "metrics": parsed,
            },
            ensure_ascii=False,
        )

    def _resolve_path(self, path: str) -> Path:
        candidate = (self._root / path).resolve()
        if not _is_relative_to(candidate, self._root):
            raise ValueError("Path escapes observability root.")
        return candidate


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = ["ObservabilityTools"]
