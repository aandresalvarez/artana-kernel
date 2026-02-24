from artana.middleware import (
    PIIScrubberMiddleware,
    QuotaMiddleware,
    CapabilityGuardMiddleware,
)
from artana.middleware.base import KernelMiddleware, ModelInvocation


class BlockKeywordMiddleware(KernelMiddleware):

    async def prepare_model(self, invocation: ModelInvocation):
        if "forbidden" in invocation.prompt.lower():
            raise ValueError("Blocked keyword detected.")
        return invocation

    async def before_model(self, **kwargs): return None
    async def after_model(self, **kwargs): return None
    async def prepare_tool_request(self, **kwargs): return kwargs["arguments_json"]
    async def prepare_tool_result(self, **kwargs): return kwargs["result_json"]


kernel = ArtanaKernel(
    store=SQLiteStore("chapter2_step5.db"),
    model_port=DemoModelPort(),
    middleware=[
        PIIScrubberMiddleware(),
        QuotaMiddleware(),
        CapabilityGuardMiddleware(),
        BlockKeywordMiddleware(),
    ],
)