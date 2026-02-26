from __future__ import annotations

import re


class StepKey:
    def __init__(self, *, namespace: str) -> None:
        normalized = self._normalize_token(namespace)
        if normalized == "":
            raise ValueError("StepKey namespace must contain at least one alphanumeric token.")
        self._namespace = normalized
        self._counters: dict[str, int] = {}

    def next(self, label: str) -> str:
        normalized_label = self._normalize_token(label)
        if normalized_label == "":
            raise ValueError("StepKey label must contain at least one alphanumeric token.")
        count = self._counters.get(normalized_label, 0) + 1
        self._counters[normalized_label] = count
        return f"{self._namespace}_{normalized_label}_{count}"

    @property
    def namespace(self) -> str:
        return self._namespace

    def _normalize_token(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


__all__ = ["StepKey"]
