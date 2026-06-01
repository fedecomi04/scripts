"""Top-level package marker.

ns-train auto-discovers methods via the ``nerfstudio.method_configs``
entry-point in pyproject.toml -- there's no plugin registration to do
here. Lazy attribute access below is kept only as a back-compat shim
for the few internal scripts that import config / model classes by
name (e.g. ``from dynamic_gs import DynamicGSModel``).

Side effect at import time: ``_suppress_nerfstudio_output_writes`` (below)
monkeypatches the two nerfstudio call-sites that would otherwise write
``config.yml`` / ``dataparser_transforms.json`` / tensorboard event files
into the central ``outputs/`` directory. All artifacts we actually need
(``post_fusion_state.pt``, SAM3/SAM3D outputs, ``timing_report*.txt``,
debug images) are written into the dataset dir by our pipelines; the
nerfstudio-managed ``outputs/`` was pure noise. Keeping the trainer's
in-memory machinery (the ``base_dir`` Path, the no-op viewer log path)
intact so nothing downstream breaks — only the actual disk writes are
suppressed.
"""

from pathlib import Path as _Path


def _suppress_nerfstudio_output_writes() -> None:
    """Disable all writes that target the central ``outputs/`` directory.

    Three patches:
      * ``ExperimentConfig.save_config`` (called by ``scripts/train.py``
        right after argparse parses) -> no-op.
      * ``Trainer.train`` opens by writing ``dataparser_transforms.json``
        next to the run dir -> wrap to skip that write and call super.
      * ``Trainer._make_dirs`` (called from ``__init__``) would mkdir the
        base_dir + checkpoint dir even when nothing will be written ->
        no-op so ns-train never even creates an outputs/ subfolder.
    """
    try:
        from nerfstudio.configs.experiment_config import ExperimentConfig
        ExperimentConfig.save_config = lambda self: None  # type: ignore[method-assign]
    except Exception:
        pass

    try:
        from nerfstudio.engine.trainer import Trainer

        _original_train = Trainer.train

        def _train_no_dataparser_dump(self):  # type: ignore[override]
            # Stock Trainer.train opens with a
            # ``dataparser_outputs.save_dataparser_transform(self.base_dir / ...)``
            # call. We don't want any disk write under outputs/, so we
            # stub the underlying save method for the duration of this
            # call (cheap, localised) and then restore it.
            dm = getattr(self.pipeline, "datamanager", None)
            dpo = getattr(dm, "train_dataparser_outputs", None) if dm is not None else None
            saved_method = None
            if dpo is not None and hasattr(dpo, "save_dataparser_transform"):
                saved_method = dpo.save_dataparser_transform
                dpo.save_dataparser_transform = lambda *_a, **_kw: None
            try:
                return _original_train(self)
            finally:
                if saved_method is not None:
                    dpo.save_dataparser_transform = saved_method

        Trainer.train = _train_no_dataparser_dump  # type: ignore[method-assign]
    except Exception:
        pass

    # vis="tensorboard" is what we pass in our method configs because we
    # need ns-train to NOT open its own viser/legacy viewer (port conflict
    # with our viser-direct on 8081). But we don't want the tb event-file
    # dir either. Force-disable the tensorboard branch inside
    # ``setup_event_writer`` — the bool arg is ignored and tensorboard is
    # never instantiated. Wandb / Comet branches untouched.
    try:
        from nerfstudio.utils import writer as _ns_writer

        _orig_setup_event_writer = _ns_writer.setup_event_writer

        def _setup_event_writer_no_tb(
            is_wandb_enabled, is_tensorboard_enabled, is_comet_enabled,
            log_dir, experiment_name, project_name="nerfstudio-project",
        ):
            # Force tb=False — everything else passes through to upstream.
            return _orig_setup_event_writer(
                is_wandb_enabled, False, is_comet_enabled,
                log_dir, experiment_name, project_name,
            )

        _ns_writer.setup_event_writer = _setup_event_writer_no_tb  # type: ignore[assignment]
    except Exception:
        pass


_suppress_nerfstudio_output_writes()


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
