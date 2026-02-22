from __future__ import annotations

from collections.abc import Sequence

from artana.middleware.base import KernelMiddleware
from artana.middleware.capability_guard import CapabilityGuardMiddleware
from artana.middleware.pii_scrubber import PIIScrubberMiddleware
from artana.middleware.quota import QuotaMiddleware


def order_middleware(middleware: Sequence[KernelMiddleware]) -> tuple[KernelMiddleware, ...]:
    prioritized: list[tuple[int, int, KernelMiddleware]] = []
    for index, middleware_item in enumerate(middleware):
        prioritized.append((_priority_for(middleware_item), index, middleware_item))
    prioritized.sort(key=lambda row: (row[0], row[1]))
    return tuple(middleware_item for _, _, middleware_item in prioritized)


def _priority_for(middleware_item: KernelMiddleware) -> int:
    if isinstance(middleware_item, PIIScrubberMiddleware):
        return 0
    if isinstance(middleware_item, QuotaMiddleware):
        return 1
    if isinstance(middleware_item, CapabilityGuardMiddleware):
        return 2
    return 3

