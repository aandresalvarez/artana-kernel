from artana.harness import SupervisorHarness

supervisor = SupervisorHarness(kernel)

result = await supervisor.run_child(
    harness=ResearchHarness(kernel),
    run_id="child_run"
)