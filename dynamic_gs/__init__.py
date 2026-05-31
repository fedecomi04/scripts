"""Top-level package marker.

ns-train auto-discovers methods via the ``nerfstudio.method_configs``
entry-point in pyproject.toml -- there's no plugin registration to do
here. Lazy attribute access below is kept only as a back-compat shim
for the few internal scripts that import config / model classes by
name (e.g. ``from dynamic_gs import DynamicGSModel``)."""

__all__ = [
    "DynamicGS",
    "DynamicGSLive",
    "DynamicGSDataManager",
    "DynamicGSDataManagerConfig",
    "DynamicGSModel",
    "DynamicGSModelConfig",
    "StaticGS",
]


def __getattr__(name):
    if name in {"DynamicGS", "DynamicGSLive", "StaticGS"}:
        from . import dynamic_gs_config as _cfg
        return getattr(_cfg, name)
    if name in {"DynamicGSDataManager", "DynamicGSDataManagerConfig"}:
        from .dynamic_gs_datamanager import DynamicGSDataManager, DynamicGSDataManagerConfig
        return {"DynamicGSDataManager": DynamicGSDataManager,
                "DynamicGSDataManagerConfig": DynamicGSDataManagerConfig}[name]
    if name in {"DynamicGSModel", "DynamicGSModelConfig"}:
        from .dynamic_gs_model import DynamicGSModel, DynamicGSModelConfig
        return {"DynamicGSModel": DynamicGSModel, "DynamicGSModelConfig": DynamicGSModelConfig}[name]
    raise AttributeError(name)
