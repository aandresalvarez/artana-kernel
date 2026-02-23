from __future__ import annotations

from pathlib import Path

import pytest

from artana.agent.experience import ExperienceRule, RuleType, SQLiteExperienceStore


@pytest.mark.asyncio
async def test_experience_store_enforces_tenant_and_task_isolation(tmp_path: Path) -> None:
    store = SQLiteExperienceStore(str(tmp_path / "experience.db"))

    try:
        await store.save_rules(
            [
                ExperienceRule(
                    rule_id="rule_finance_a",
                    tenant_id="tenant_a",
                    task_category="Financial_Reporting",
                    rule_type=RuleType.WIN_PATTERN,
                    content="Always format dates as YYYY-MM-DD.",
                    success_count=3,
                    fail_count=0,
                ),
                ExperienceRule(
                    rule_id="rule_finance_b",
                    tenant_id="tenant_b",
                    task_category="Financial_Reporting",
                    rule_type=RuleType.ANTI_PATTERN,
                    content="Do not call /v1/users during monthly close.",
                    success_count=1,
                    fail_count=0,
                ),
                ExperienceRule(
                    rule_id="rule_research_a",
                    tenant_id="tenant_a",
                    task_category="Research",
                    rule_type=RuleType.FACT,
                    content="Acme CEO is Jane Doe.",
                    success_count=1,
                    fail_count=0,
                ),
            ]
        )

        tenant_a_finance = await store.get_rules(
            tenant_id="tenant_a",
            task_category="Financial_Reporting",
        )
        tenant_b_finance = await store.get_rules(
            tenant_id="tenant_b",
            task_category="Financial_Reporting",
        )
        tenant_a_research = await store.get_rules(
            tenant_id="tenant_a",
            task_category="Research",
        )

        assert [rule.rule_id for rule in tenant_a_finance] == ["rule_finance_a"]
        assert [rule.rule_id for rule in tenant_b_finance] == ["rule_finance_b"]
        assert [rule.rule_id for rule in tenant_a_research] == ["rule_research_a"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_experience_store_reinforce_rule_updates_counters(tmp_path: Path) -> None:
    store = SQLiteExperienceStore(str(tmp_path / "experience.db"))

    try:
        await store.save_rules(
            [
                ExperienceRule(
                    rule_id="rule_reinforce",
                    tenant_id="tenant_a",
                    task_category="Financial_Reporting",
                    rule_type=RuleType.WIN_PATTERN,
                    content="Use strict schema validation.",
                )
            ]
        )
        await store.reinforce_rule("rule_reinforce", positive=True)
        await store.reinforce_rule("rule_reinforce", positive=True)
        await store.reinforce_rule("rule_reinforce", positive=False)

        rules = await store.get_rules(
            tenant_id="tenant_a",
            task_category="Financial_Reporting",
        )
        assert len(rules) == 1
        assert rules[0].success_count == 2
        assert rules[0].fail_count == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_experience_store_applies_limit_and_priority_ordering(tmp_path: Path) -> None:
    store = SQLiteExperienceStore(str(tmp_path / "experience.db"))

    try:
        await store.save_rules(
            [
                ExperienceRule(
                    rule_id="rule_low",
                    tenant_id="tenant_a",
                    task_category="Financial_Reporting",
                    rule_type=RuleType.FACT,
                    content="Legacy account IDs include a prefix.",
                    success_count=1,
                    fail_count=2,
                ),
                ExperienceRule(
                    rule_id="rule_top",
                    tenant_id="tenant_a",
                    task_category="Financial_Reporting",
                    rule_type=RuleType.WIN_PATTERN,
                    content="Normalize dates to ISO before API calls.",
                    success_count=5,
                    fail_count=0,
                ),
                ExperienceRule(
                    rule_id="rule_mid",
                    tenant_id="tenant_a",
                    task_category="Financial_Reporting",
                    rule_type=RuleType.ANTI_PATTERN,
                    content="Avoid querying regionless partitions.",
                    success_count=2,
                    fail_count=0,
                ),
            ]
        )

        top_two = await store.get_rules(
            tenant_id="tenant_a",
            task_category="Financial_Reporting",
            limit=2,
        )
        assert [rule.rule_id for rule in top_two] == ["rule_top", "rule_mid"]
    finally:
        await store.close()
