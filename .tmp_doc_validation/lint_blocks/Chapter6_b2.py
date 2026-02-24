from artana.safety import IntentPlanRecord

await kernel.record_intent_plan(
    run_id="billing_run",
    tenant=tenant,
    intent=IntentPlanRecord(
        intent_id="intent_2026_02",
        goal="Send February invoice",
        why="Monthly billing close",
        success_criteria="Invoice sent exactly once",
        assumed_state="Customer account is active and approved",
        applies_to_tools=("send_invoice",),
    ),
)