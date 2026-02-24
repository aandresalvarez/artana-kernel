from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore

kernel = ArtanaKernel(
    store=SQLiteStore("multi_tenant.db"),
    model_port=DemoModelPort(),
)

tenant_a = TenantContext(
    tenant_id="tenant_a",
    capabilities=frozenset({"analytics"}),
    budget_usd_limit=10.0,
)

tenant_b = TenantContext(
    tenant_id="tenant_b",
    capabilities=frozenset(),
    budget_usd_limit=2.0,
)