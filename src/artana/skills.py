from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    name: str
    version: str
    summary: str
    instructions_markdown: str
    tools: tuple[str, ...] = ()
    requires_capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


class SkillRegistry(ABC):
    @abstractmethod
    def list_skills(self) -> tuple[SkillDefinition, ...]:
        raise NotImplementedError

    @abstractmethod
    def get_skill(self, name: str) -> SkillDefinition | None:
        raise NotImplementedError

    def skill_names(self) -> frozenset[str]:
        return frozenset(skill.name for skill in self.list_skills())


class FilesystemSkillRegistry(SkillRegistry):
    def __init__(self, root_paths: Sequence[str | Path] | str | Path) -> None:
        if isinstance(root_paths, (str, Path)):
            root_paths = (root_paths,)
        roots = tuple(Path(root).expanduser() for root in root_paths)
        if not roots:
            raise ValueError("FilesystemSkillRegistry requires at least one root path.")

        normalized_roots: list[Path] = []
        for root in roots:
            if not root.exists():
                raise ValueError(f"Skill root does not exist: {root}")
            if not root.is_dir():
                raise ValueError(f"Skill root is not a directory: {root}")
            normalized_roots.append(root)

        definitions: dict[str, SkillDefinition] = {}
        for root in normalized_roots:
            for path in sorted(root.rglob("SKILL.md")):
                definition = _parse_skill_file(path)
                existing = definitions.get(definition.name)
                if existing is not None:
                    raise ValueError(
                        f"Duplicate skill name {definition.name!r} discovered in {path}."
                    )
                definitions[definition.name] = definition

        self._root_paths = tuple(normalized_roots)
        self._definitions = definitions

    @property
    def root_paths(self) -> tuple[Path, ...]:
        return self._root_paths

    def list_skills(self) -> tuple[SkillDefinition, ...]:
        return tuple(self._definitions[name] for name in sorted(self._definitions))

    def get_skill(self, name: str) -> SkillDefinition | None:
        return self._definitions.get(name)


def _parse_skill_file(path: Path) -> SkillDefinition:
    try:
        raw_content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read skill file {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Skill file {path} must be valid UTF-8.") from exc

    frontmatter_text, body = _split_frontmatter(path=path, content=raw_content)
    try:
        metadata = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML frontmatter in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Skill metadata in {path} must be a mapping.")

    name = _require_non_empty_string(value=metadata.get("name"), field="name", path=path)
    version = _require_non_empty_string(
        value=metadata.get("version"),
        field="version",
        path=path,
    )
    summary = _require_non_empty_string(
        value=metadata.get("summary"),
        field="summary",
        path=path,
    )
    tools = _normalize_string_sequence(value=metadata.get("tools"), field="tools", path=path)
    requires_capabilities = _normalize_string_sequence(
        value=metadata.get("requires_capabilities"),
        field="requires_capabilities",
        path=path,
    )
    tags = _normalize_string_sequence(value=metadata.get("tags"), field="tags", path=path)

    return SkillDefinition(
        name=name,
        version=version,
        summary=summary,
        instructions_markdown=body.strip(),
        tools=tools,
        requires_capabilities=requires_capabilities,
        tags=tags,
    )


def _split_frontmatter(*, path: Path, content: str) -> tuple[str, str]:
    normalized = content.replace("\r\n", "\n")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Skill file {path} must start with YAML frontmatter.")
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            return frontmatter, body
    raise ValueError(f"Skill file {path} is missing a closing YAML frontmatter delimiter.")


def _require_non_empty_string(*, field: str, path: Path, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Skill field {field!r} in {path} must be a string.")
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"Skill field {field!r} in {path} must not be empty.")
    return normalized


def _normalize_string_sequence(*, field: str, path: Path, value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Skill field {field!r} in {path} must be a list of strings.")
    normalized_items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Skill field {field!r} in {path} must contain only strings.")
        normalized = item.strip()
        if normalized == "":
            raise ValueError(
                f"Skill field {field!r} in {path} must not contain empty strings."
            )
        if normalized not in normalized_items:
            normalized_items.append(normalized)
    return tuple(normalized_items)


__all__ = ["FilesystemSkillRegistry", "SkillDefinition", "SkillRegistry"]
