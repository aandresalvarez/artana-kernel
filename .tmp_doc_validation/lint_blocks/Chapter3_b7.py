from artana.store import SQLiteStore

events = await kernel.get_events(run_id="migration_run")

for event in events:
    print(event.seq, event.event_type)

store = SQLiteStore("chapter3_step3.db")
valid = await store.verify_run_chain("migration_run")
print("Ledger valid:", valid)
await store.close()