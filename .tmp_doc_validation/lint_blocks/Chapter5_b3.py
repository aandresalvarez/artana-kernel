from artana import ArtanaKernel, KernelPolicy, PostgresStore
from artana.ports.model_adapter import LiteLLMAdapter

kernel = ArtanaKernel(
    store=PostgresStore("postgresql://user:pass@db:5432/artana"),  # shared DB
    model_port=LiteLLMAdapter(...),
    middleware=ArtanaKernel.default_middleware_stack(),
    policy=KernelPolicy.enforced(),
)