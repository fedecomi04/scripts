import sys as _sys

# CLI sugar: rewrite top-level `--live` to `--pipeline.live True` before
# tyro parses argv. nerfstudio imports this package during entry-point
# discovery, so the rewrite happens before any CLI parsing. We also
# inject a placeholder `--data` if missing — tyro/nerfstudio still want
# the flag present, but live mode overwrites the value internally.
if "--live" in _sys.argv:
    _new_argv = []
    for _arg in _sys.argv:
        if _arg == "--live":
            _new_argv.extend(["--pipeline.live", "True"])
        else:
            _new_argv.append(_arg)
    _sys.argv[:] = _new_argv
    if "--data" not in _sys.argv:
        _sys.argv.extend(["--data", "/tmp/dynamic_gs_live_placeholder"])
# PROBLEM: this runs as a side-effect at every `import dynamic_gs`, not
# only at ns-train invocation. Test scripts that `import dynamic_gs`
# while passing --live in their own argv would have it rewritten too —
# acceptable for our use, since we only invoke this package via
# ns-train.

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
