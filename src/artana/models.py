from __future__ import annotations

from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    tenant_id: str = Field(min_length=1)
    capabilities: frozenset[str] = Field(default_factory=frozenset)
    budget_usd_limit: float = Field(gt=0.0)

