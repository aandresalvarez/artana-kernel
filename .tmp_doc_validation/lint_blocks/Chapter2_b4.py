await harness.set_artifact(key="schema_version", value={"v": 2})
schema = await harness.get_artifact(key="schema_version")
print("Schema:", schema)