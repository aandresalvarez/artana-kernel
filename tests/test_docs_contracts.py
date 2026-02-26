from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
EXAMPLES_README = REPO_ROOT / "examples" / "README.md"
CHAPTER_FILES = tuple(DOCS_DIR / f"Chapter{index}.md" for index in range(1, 7))

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FENCED_BLOCK_RE = re.compile(
    r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)\n```",
    re.DOTALL,
)
URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _chapter_link(index: int) -> str:
    return f"(./Chapter{index}.md)"


def _markdown_docs_files() -> tuple[Path, ...]:
    docs_markdown = sorted(path for path in DOCS_DIR.rglob("*.md"))
    return tuple((*docs_markdown, EXAMPLES_README))


def _normalized_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    # Ignore optional markdown title payloads after the first token.
    return target.split(maxsplit=1)[0]


def _iter_markdown_links(path: Path) -> tuple[str, ...]:
    text = _read(path)
    return tuple(
        _normalized_link_target(match.group(1)) for match in MARKDOWN_LINK_RE.finditer(text)
    )


def test_chapter_docs_exist() -> None:
    missing = [str(path.relative_to(REPO_ROOT)) for path in CHAPTER_FILES if not path.exists()]
    assert not missing, "Missing chapter docs:\n" + "\n".join(missing)


def test_docs_readme_links_chapters_in_recommended_order() -> None:
    docs_readme = _read(DOCS_DIR / "README.md")
    positions: list[int] = []
    for chapter_index in range(1, 7):
        needle = _chapter_link(chapter_index)
        pos = docs_readme.find(needle)
        assert pos >= 0, f"docs/README.md is missing {needle}"
        positions.append(pos)
    assert positions == sorted(positions), "Chapter links in docs/README.md must be ordered 1 -> 6"


def test_no_model_port_none_in_python_fences_for_chapters() -> None:
    violations: list[str] = []

    for chapter in CHAPTER_FILES:
        for block in FENCED_BLOCK_RE.finditer(_read(chapter)):
            language = block.group("lang").strip().lower()
            if not language.startswith("python"):
                continue
            body = block.group("body")
            if "model_port=None" in body:
                line = _read(chapter)[: block.start()].count("\n") + 1
                violations.append(f"{chapter.relative_to(REPO_ROOT)}:{line}")

    assert not violations, "Replace model_port=None in runnable python blocks:\n" + "\n".join(
        violations
    )


def test_cli_docs_cover_current_command_surface() -> None:
    combined = "\n".join(
        (
            _read(DOCS_DIR / "README.md"),
            _read(DOCS_DIR / "Chapter5.md"),
            _read(DOCS_DIR / "kernel_contracts.md"),
            _read(DOCS_DIR / "deep_traceability.md"),
        )
    )
    required_fragments = (
        "artana run status",
        "artana run summaries",
        "artana run artifacts",
        "--json",
        "artana init",
        "--profile enforced|dev",
    )
    missing = [fragment for fragment in required_fragments if fragment not in combined]
    assert not missing, "Missing CLI docs coverage:\n" + "\n".join(missing)


def test_each_chapter_has_contract_and_handoff_sections() -> None:
    for chapter_index, chapter_path in enumerate(CHAPTER_FILES, start=1):
        content = _read(chapter_path)
        assert "## Chapter Metadata" in content, f"{chapter_path.name} is missing Chapter Metadata"
        assert (
            "Code block contract for this chapter:" in content
        ), f"{chapter_path.name} is missing code block contract"
        assert (
            "## You Should Now Be Able To" in content
        ), f"{chapter_path.name} is missing chapter checkpoint section"

        if chapter_index < 6:
            next_link = _chapter_link(chapter_index + 1)
            assert (
                "## Next Chapter" in content
            ), f"{chapter_path.name} is missing Next Chapter handoff"
            assert next_link in content, (
                f"{chapter_path.name} must hand off to Chapter {chapter_index + 1} via {next_link}"
            )
        else:
            assert (
                "## Where To Go Next" in content
            ), f"{chapter_path.name} is missing final handoff section"


def test_markdown_links_under_docs_and_examples_are_resolvable() -> None:
    violations: list[str] = []

    for markdown_file in _markdown_docs_files():
        for target in _iter_markdown_links(markdown_file):
            if not target or target.startswith("#"):
                continue
            if URI_SCHEME_RE.match(target):
                continue
            if target.startswith("/"):
                rel_path = markdown_file.relative_to(REPO_ROOT)
                violations.append(
                    f"{rel_path}: absolute local link is not allowed ({target})"
                )
                continue

            path_part = target.split("#", maxsplit=1)[0]
            if not path_part:
                continue
            candidate = (markdown_file.parent / path_part).resolve()
            if not candidate.exists():
                rel = markdown_file.relative_to(REPO_ROOT)
                violations.append(f"{rel}: broken link target ({target})")

    assert not violations, "Markdown link validation failed:\n" + "\n".join(sorted(violations))
