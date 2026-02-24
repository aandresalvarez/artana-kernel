await harness.set_artifact(key="plan", value={"phase": 1})
plan = await harness.get_artifact(key="plan")
print(plan)