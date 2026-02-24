import sqlite3

connection = sqlite3.connect("chapter4_step1.db")

rows = connection.execute(
    """
    SELECT
        tenant_id,
        SUM(CAST(json_extract(payload_json, '$.cost_usd') AS FLOAT)) AS total_spend,
        COUNT(*) AS model_calls
    FROM kernel_events
    WHERE event_type = 'model_completed'
    GROUP BY tenant_id
    ORDER BY total_spend DESC
    """
).fetchall()

for row in rows:
    print(row)

connection.close()