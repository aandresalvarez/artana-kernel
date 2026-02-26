from __future__ import annotations

import pytest

from artana.util import StepKey


def test_step_key_increments_per_label() -> None:
    step = StepKey(namespace="refactor_auth")
    assert step.next("draft") == "refactor_auth_draft_1"
    assert step.next("draft") == "refactor_auth_draft_2"
    assert step.next("verify") == "refactor_auth_verify_1"


def test_step_key_normalizes_tokens() -> None:
    step = StepKey(namespace="Refactor Auth!")
    assert step.next("draft/model") == "refactor_auth_draft_model_1"


def test_step_key_rejects_empty_namespace_or_label() -> None:
    with pytest.raises(ValueError, match="namespace"):
        StepKey(namespace="!!!")

    step = StepKey(namespace="ok")
    with pytest.raises(ValueError, match="label"):
        step.next("!!!")
