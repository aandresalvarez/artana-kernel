from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLES_README = EXAMPLES_DIR / "README.md"

LOCAL_FIRST_EXAMPLES = frozenset(
    {
        "01_durable_chat_replay.py",
        "04_autonomous_agent_research.py",
        "05_hard_triplets_workflow.py",
        "07_adaptive_agent_learning.py",
        "09_harness_engineering_dx.py",
    }
)
LIVE_MODEL_EXAMPLES = frozenset(
    {
        "02_real_litellm_chat.py",
        "03_fact_extraction_triplets.py",
        "06_triplets_swarm.py",
        "08_responses_mode.py",
        "golden_example.py",
    }
)


def _example_files() -> frozenset[str]:
    return frozenset(
        path.name for path in EXAMPLES_DIR.glob("*.py") if not path.name.startswith("_")
    )


def test_runtime_profiles_partition_all_examples() -> None:
    files = _example_files()
    assert LOCAL_FIRST_EXAMPLES.isdisjoint(LIVE_MODEL_EXAMPLES)
    assert LOCAL_FIRST_EXAMPLES | LIVE_MODEL_EXAMPLES == files


def test_examples_readme_mentions_all_scripts() -> None:
    readme = EXAMPLES_README.read_text(encoding="utf-8")
    for script_name in sorted(_example_files()):
        assert (
            f"`{script_name}`" in readme or f"examples/{script_name}" in readme
        ), f"examples/README.md is missing {script_name}"


def test_runtime_profiles_match_script_env_requirements() -> None:
    for script_name in sorted(LOCAL_FIRST_EXAMPLES):
        content = (EXAMPLES_DIR / script_name).read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" not in content, (
            f"{script_name} is marked local-first but gates on API key."
        )

    for script_name in sorted(LIVE_MODEL_EXAMPLES):
        content = (EXAMPLES_DIR / script_name).read_text(encoding="utf-8")
        assert "require_openai_api_key(" in content, (
            f"{script_name} is marked live-model but lacks API key bootstrap guard."
        )
        assert "friendly_exit(" in content, (
            f"{script_name} is marked live-model but lacks user-friendly failure output."
        )
        assert "print_summary(" in content, (
            f"{script_name} is marked live-model but lacks consistent summary output."
        )


def test_examples_syntax_compiles() -> None:
    for script_name in sorted(_example_files()):
        source = (EXAMPLES_DIR / script_name).read_text(encoding="utf-8")
        compile(source, script_name, "exec")
