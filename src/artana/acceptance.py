from __future__ import annotations

from pydantic import BaseModel, Field


class ToolGate(BaseModel):
    tool: str = Field(min_length=1)
    must_pass: bool = True
    arguments_json: str = "{}"
    pass_json_path: str | None = None
    pass_if_equals_json: str | None = None


class AcceptanceSpec(BaseModel):
    gates: tuple[ToolGate, ...] = Field(default_factory=tuple)


__all__ = ["AcceptanceSpec", "ToolGate"]
