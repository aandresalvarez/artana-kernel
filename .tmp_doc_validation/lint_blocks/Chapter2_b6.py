from artana.store import SQLiteStore

events = await kernel.get_events(run_id="pipeline_run_001")

for event in events:
    print(event.seq, event.event_type)

store = SQLiteStore("chapter2_step5.db")
verified = await store.verify_run_chain("pipeline_run_001")
print("Chain valid:", verified)
await store.close()