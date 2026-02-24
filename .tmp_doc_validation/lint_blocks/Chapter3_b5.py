from artana.kernel import KernelPolicy
from artana.middleware import (
    PIIScrubberMiddleware,
    QuotaMiddleware,
    CapabilityGuardMiddleware,
)

kernel = ArtanaKernel(
    store=SQLiteStore("prod.db"),
    model_port=HybridModel(),
    middleware=[
        PIIScrubberMiddleware(),
        QuotaMiddleware(),
        CapabilityGuardMiddleware(),
    ],
    policy=KernelPolicy.enforced(),
)