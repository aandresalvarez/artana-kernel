await harness.set_artifact(key="plan", value={"version": 2, "status": "approved"})
artifact = await harness.get_artifact(key="plan")