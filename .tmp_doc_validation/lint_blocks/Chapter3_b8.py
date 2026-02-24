from artana.store import PostgresStore

store = PostgresStore(
    dsn="postgresql://user:pass@db:5432/artana",
    min_pool_size=2,
    max_pool_size=20,
)