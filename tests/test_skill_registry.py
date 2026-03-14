from __future__ import annotations

from pathlib import Path

import pytest

from artana import FilesystemSkillRegistry


def _write_skill_file(root: Path, *, slug: str, content: str) -> Path:
    path = root / slug / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_filesystem_skill_registry_parses_frontmatter_and_markdown(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_file(
        skills_root,
        slug="reader",
        content=(
            "---\n"
            "name: reader_skill\n"
            "version: 1.2.3\n"
            "summary: Read demo artifacts\n"
            "tools:\n"
            "  - read_demo_file\n"
            "requires_capabilities:\n"
            "  - demo:read\n"
            "tags:\n"
            "  - demo\n"
            "---\n"
            "Use the reader skill when you need grounded file access.\n"
        ),
    )

    registry = FilesystemSkillRegistry(skills_root)
    skills = registry.list_skills()

    assert len(skills) == 1
    assert skills[0].name == "reader_skill"
    assert skills[0].version == "1.2.3"
    assert skills[0].summary == "Read demo artifacts"
    assert skills[0].instructions_markdown == (
        "Use the reader skill when you need grounded file access."
    )
    assert skills[0].tools == ("read_demo_file",)
    assert skills[0].requires_capabilities == ("demo:read",)
    assert skills[0].tags == ("demo",)
    assert registry.get_skill("reader_skill") == skills[0]


def test_filesystem_skill_registry_rejects_duplicate_skill_names(tmp_path: Path) -> None:
    first_root = tmp_path / "skills_one"
    second_root = tmp_path / "skills_two"
    duplicate_content = (
        "---\n"
        "name: shared_skill\n"
        "version: 1.0.0\n"
        "summary: Shared skill\n"
        "---\n"
        "Shared instructions.\n"
    )
    _write_skill_file(first_root, slug="shared_a", content=duplicate_content)
    _write_skill_file(second_root, slug="shared_b", content=duplicate_content)

    with pytest.raises(ValueError, match="Duplicate skill name 'shared_skill'"):
        FilesystemSkillRegistry((first_root, second_root))


def test_filesystem_skill_registry_rejects_malformed_metadata(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_file(
        skills_root,
        slug="invalid",
        content=(
            "---\n"
            "name: invalid_skill\n"
            "version: 1.0.0\n"
            "summary: Broken metadata\n"
            "tools: read_demo_file\n"
            "---\n"
            "Broken skill.\n"
        ),
    )

    with pytest.raises(ValueError, match="Skill field 'tools'.*must be a list of strings"):
        FilesystemSkillRegistry(skills_root)


def test_filesystem_skill_registry_allows_instruction_only_skills(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_file(
        skills_root,
        slug="style",
        content=(
            "---\n"
            "name: style_skill\n"
            "version: 1.0.0\n"
            "summary: Style guidance only\n"
            "---\n"
            "Always answer concisely.\n"
        ),
    )

    registry = FilesystemSkillRegistry(skills_root)
    skill = registry.get_skill("style_skill")

    assert skill is not None
    assert skill.tools == ()
    assert skill.instructions_markdown == "Always answer concisely."


def test_filesystem_skill_registry_requires_frontmatter(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_file(
        skills_root,
        slug="missing_frontmatter",
        content="No frontmatter here.\n",
    )

    with pytest.raises(ValueError, match="must start with YAML frontmatter"):
        FilesystemSkillRegistry(skills_root)
