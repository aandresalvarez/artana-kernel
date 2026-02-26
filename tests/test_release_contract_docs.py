from __future__ import annotations

import tomllib
from pathlib import Path


def test_current_version_documented_in_changelog_and_compat_matrix() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject_data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    current_version = pyproject_data["project"]["version"]

    changelog_text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    compatibility_matrix_text = (root / "docs" / "compatibility_matrix.md").read_text(
        encoding="utf-8"
    )

    assert current_version in changelog_text
    assert current_version in compatibility_matrix_text
