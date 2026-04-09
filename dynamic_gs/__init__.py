__all__ = [
    "DynamicGS",
    "DynamicGSDataManager",
    "DynamicGSDataManagerConfig",
    "DynamicGSModel",
    "DynamicGSModelConfig",
    "DynamicGSPipeline",
    "DynamicGSPipelineConfig",
]


def __getattr__(name):
    if name == "DynamicGS":
        from .dynamic_gs_config import DynamicGS

        return DynamicGS
    if name in {"DynamicGSDataManager", "DynamicGSDataManagerConfig"}:
        from .dynamic_gs_datamanager import DynamicGSDataManager, DynamicGSDataManagerConfig

        return {"DynamicGSDataManager": DynamicGSDataManager, "DynamicGSDataManagerConfig": DynamicGSDataManagerConfig}[name]
    if name in {"DynamicGSModel", "DynamicGSModelConfig"}:
        from .dynamic_gs_model import DynamicGSModel, DynamicGSModelConfig

        return {"DynamicGSModel": DynamicGSModel, "DynamicGSModelConfig": DynamicGSModelConfig}[name]
    if name in {"DynamicGSPipeline", "DynamicGSPipelineConfig"}:
        from .dynamic_gs_pipeline import DynamicGSPipeline, DynamicGSPipelineConfig

        return {"DynamicGSPipeline": DynamicGSPipeline, "DynamicGSPipelineConfig": DynamicGSPipelineConfig}[name]
    raise AttributeError(name)
