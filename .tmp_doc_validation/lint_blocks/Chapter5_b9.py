from artana import ArtanaKernel, KernelPolicy
from artana.middleware import SafetyPolicyMiddleware
from artana.ports.model_adapter import LiteLLMAdapter
from artana.safety import SafetyPolicyConfig
from artana.store import PostgresStore

safety = SafetyPolicyMiddleware(config=SafetyPolicyConfig(tools={...}))

kernel = ArtanaKernel(
    store=PostgresStore("postgresql://..."),
    model_port=LiteLLMAdapter(...),
    middleware=ArtanaKernel.default_middleware_stack(safety=safety),
    policy=KernelPolicy.enforced_v2(),
)