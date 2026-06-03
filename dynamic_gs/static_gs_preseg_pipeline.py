"""``static-gs-preseg`` pipeline — pre-segment, label seed, train once.

The shipped ``static-gs`` flow is: build TSDF seed → train Splatfacto → run
Phase 0a/0b (SAM3 → Fast-SAM3D → CPD/TEASER registration + insertion). This
method flips the order: **segment & label the seed cloud BEFORE training**.

Concretely on each call:

1. ``icp_pose_refine.refine_poses_and_refuse`` — idempotent; ensures
   ``transforms.json`` carries ``pose_source: "icp_refined_from_urdf_v1"``
   (Design Invariant #3 in CLAUDE.md). Backs up the URDF poses on first run.
2. Spawn ``SamWorkerClient`` and pre-load SAM3 in a background thread.
3. ``preseg_seed.build_labeled_seed`` — SAM2-AMG on frame 0 → SAM3-grouped
   merge → SAM2-video propagate across all frames → occlusion-voted 3D
   label transfer onto the existing seed PLY. Writes a sidecar
   ``<ply>.instance_ids.npy`` (N,) int64 next to the PLY.
4. Tear down the SAM worker.
5. ``super().__init__(...)`` (VanillaPipeline) — builds the datamanager
   (which reads the PLY via Nerfstudio's dataparser) and the model (which
   registers ``object_instance_ids`` as zeros sized to ``num_points``).
6. ``_load_sidecar_into_buffer`` — read the sidecar, validate shape, copy
   into ``model.object_instance_ids[:, 0]``. From this point on every
   Gaussian carries its seed-point's instance id.
7. Nerfstudio's trainer runs Splatfacto for ``static_num_steps`` with
   ``NoRefineStrategy`` (no clone/split/prune; Gaussian count is fixed →
   the seed-point ↔ Gaussian-row alignment is stable) and ``means LR=0``
   (Design Invariant #1: positions stay on the seed).
8. ``AFTER_TRAIN`` callback ``_save_post_fusion_state`` writes
   ``static_scene/static_state.pt`` with the SAME schema as the
   existing ``static-gs`` method — only ``object_instance_ids`` is
   pre-populated; ``object_flags`` / ``sam3d_init_target_flags`` /
   ``inserted_flags`` stay zero (the dynamic pipeline sets ``object_flags``
   at D0).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE

from .dynamic_gs_datamanager import DynamicGSDataManagerConfig
from .persistence.post_fusion_cache import save_post_fusion_state
from .static_gs_model import StaticGSModelConfig
from .static_gs_pipeline import StaticGSPipeline, StaticGSPipelineConfig
from .utils.icp_pose_refine import refine_poses_and_refuse
from .utils.preseg_seed import AmgConfig, build_labeled_seed
from .utils.sam_worker import SamWorkerClient


@dataclass
class StaticGSPresegPipelineConfig(StaticGSPipelineConfig):
    """Config for ``static-gs-preseg``."""

    _target: Type = field(default_factory=lambda: StaticGSPresegPipeline)

    # Segmentation knobs (forwarded to build_labeled_seed). Defaults match
    # experiments/sam3_seed_sam2_mvp/amg_merge_propagate.py.
    text_prompts: tuple[str, ...] = ("objects",)
    sam3_confidence_threshold: float = 0.40
    coverage_threshold: float = 0.80
    amg_points_per_side: int = 32
    amg_min_area_px: int = 400
    amg_max_area_frac: float = 0.60
    amg_max_masks: int = 32
    sam2_hf_model: str = "facebook/sam2.1-hiera-large"
    min_obj_votes: int = 2
    occ_tol_m: float = 0.02

    # Behavior knobs.
    refine_poses_with_icp: bool = True
    """Honor Design Invariant #3: ensure transforms.json contains ICP-refined
    poses before training. Idempotent — skips if already refined."""

    reuse_sidecar_if_present: bool = True
    """If <ply>.instance_ids.npy already sits next to the seed PLY (from a
    previous preseg run on the same data), skip SAM3+SAM2 and reuse it.
    Set to False to force a fresh segmentation."""


class StaticGSPresegPipeline(StaticGSPipeline):
    """Pre-segment + label seed + train once. See module docstring."""

    config: StaticGSPresegPipelineConfig

    def __init__(
        self,
        config: StaticGSPresegPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler=None,
    ):
        # Mirror StaticGSPipeline's bookkeeping. _sam3d_generation_outputs
        # stays None (no Phase 0a in this method); _phase0b_done is repurposed
        # as "saved" idempotency guard.
        self._timing: defaultdict[str, list] = defaultdict(list)
        self._sam3d_generation_outputs = None
        self._phase0b_done: bool = False
        self._timing_report_written: bool = False
        self._sidecar_path: Optional[Path] = None
        self._labeled_instance_count: int = 0
        self._num_instances: int = 0

        # Path setup — only the dataset dir + ply path; the model isn't
        # built yet.
        dataset_dir = Path(config.datamanager.data).resolve()
        static_dir = dataset_dir / config.datamanager.static_subdir
        ply_path = static_dir / "depth_camera_init_points.ply"
        sidecar_path = ply_path.with_suffix(".instance_ids.npy")
        preseg_out_dir = static_dir / "preseg_artifacts"
        preseg_out_dir.mkdir(parents=True, exist_ok=True)

        self._dataset_dir = dataset_dir
        self._static_dir = static_dir
        self._ply_path = ply_path
        self._sidecar_path = sidecar_path
        self._preseg_out_dir = preseg_out_dir

        # 1. ICP-refine transforms.json (idempotent).
        if config.refine_poses_with_icp:
            try:
                icp_result = refine_poses_and_refuse(dataset_dir)
                if icp_result.get("skipped"):
                    CONSOLE.log(
                        f"[static-gs-preseg] ICP pose refine skipped: "
                        f"{icp_result.get('reason')}"
                    )
                else:
                    CONSOLE.log(
                        f"[static-gs-preseg] ICP pose refine done — "
                        f"{icp_result.get('frames')} frames, median "
                        f"dt={icp_result.get('dt_mm_median'):.2f}mm "
                        f"dR={icp_result.get('dR_deg_median'):.3f}deg"
                    )
            except Exception as exc:
                CONSOLE.log(
                    f"[static-gs-preseg] ICP pose refine raised; continuing with "
                    f"existing transforms.json: {exc}"
                )

        # 2. Label seed (cache hit short-circuits SAM3 + SAM2).
        if sidecar_path.exists() and config.reuse_sidecar_if_present:
            CONSOLE.log(
                f"[static-gs-preseg] reusing cached sidecar {sidecar_path.name}"
            )
        else:
            self._run_segmentation_and_label_seed(config)

        # 3. Build datamanager + model. VanillaPipeline.__init__ loads the
        # PLY via the dataparser (xyz + rgb only — sidecar ids are loaded
        # separately by us once the model exists). We bypass
        # StaticGSPipeline.__init__ because it runs Phase 0a, which this
        # method doesn't have.
        VanillaPipeline.__init__(
            self,
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )

        # 4. atexit timing report (mirrors StaticGSPipeline).
        import atexit as _atexit
        _atexit.register(self._write_timing_report)

        # 5. Load sidecar into model.object_instance_ids.
        self._load_sidecar_into_buffer()

    # ------------------------------------------------------------------
    # Seed segmentation + sidecar write
    # ------------------------------------------------------------------

    def _run_segmentation_and_label_seed(
        self, config: StaticGSPresegPipelineConfig
    ) -> None:
        if not self._ply_path.exists():
            raise FileNotFoundError(
                f"[static-gs-preseg] expected seed PLY at {self._ply_path}; "
                "run a capture first (TSDF integration must have run)."
            )

        import time as _t

        # Spawn worker + pre-load SAM3 in a background thread; main thread
        # can start setting up SAM2 inside build_labeled_seed in parallel.
        t_spawn0 = _t.perf_counter()
        sam_worker = SamWorkerClient()
        self._timing["P0a.worker_spawn"].append(_t.perf_counter() - t_spawn0)
        CONSOLE.log(
            f"[static-gs-preseg] SAM worker spawned "
            f"({sam_worker.spawn_seconds:.2f}s)"
        )

        load_done = {"err": None, "seconds": 0.0}

        def _bg_load_sam3() -> None:
            try:
                t0 = _t.perf_counter()
                sam_worker.load_sam3(
                    confidence_threshold=config.sam3_confidence_threshold,
                )
                load_done["seconds"] = _t.perf_counter() - t0
            except Exception as exc:
                load_done["err"] = exc

        bg = threading.Thread(target=_bg_load_sam3, name="preseg_sam3_load", daemon=True)
        bg.start()

        # Wait — build_labeled_seed must have SAM3 ready when it calls
        # sam3_infer_raw. We block here for correctness; SAM2 loading
        # happens inside build_labeled_seed, naturally parallel with the
        # tail of this load.
        bg.join()
        if load_done["err"] is not None:
            sam_worker.close()
            raise load_done["err"]
        self._timing["P0a.sam3_load"].append(load_done["seconds"])
        CONSOLE.log(
            f"[static-gs-preseg] SAM3 loaded ({load_done['seconds']:.2f}s)"
        )

        amg_cfg = AmgConfig(
            points_per_side=config.amg_points_per_side,
            min_area_px=config.amg_min_area_px,
            max_area_frac=config.amg_max_area_frac,
            max_masks=config.amg_max_masks,
            black_out_gripper=True,
            sam2_hf_model=config.sam2_hf_model,
        )

        try:
            t_seg0 = _t.perf_counter()
            result = build_labeled_seed(
                dataset_dir=self._dataset_dir,
                ply_path=self._ply_path,
                out_dir=self._preseg_out_dir,
                sam_worker=sam_worker,
                text_prompts=list(config.text_prompts),
                sam3_confidence_threshold=config.sam3_confidence_threshold,
                coverage_threshold=config.coverage_threshold,
                amg_cfg=amg_cfg,
                min_obj_votes=config.min_obj_votes,
                occ_tol_m=config.occ_tol_m,
            )
            self._timing["P0a.build_labeled_seed"].append(
                _t.perf_counter() - t_seg0
            )
            self._sidecar_path = result.instance_ids_path
            self._labeled_instance_count = result.num_labeled_points
            self._num_instances = result.num_instances
            CONSOLE.log(
                f"[static-gs-preseg] sidecar written → {result.instance_ids_path.name} "
                f"(K={result.num_instances}, labeled_points="
                f"{result.num_labeled_points} / total points unknown)"
            )
        finally:
            try:
                sam_worker.unload_sam3()
            except Exception:
                pass
            try:
                sam_worker.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Sidecar → model.object_instance_ids
    # ------------------------------------------------------------------

    def _load_sidecar_into_buffer(self) -> None:
        if self._sidecar_path is None or not self._sidecar_path.exists():
            CONSOLE.log(
                f"[static-gs-preseg] no sidecar at {self._sidecar_path}; "
                "object_instance_ids stays at zero"
            )
            return
        try:
            ids = np.load(self._sidecar_path)
        except Exception as exc:
            CONSOLE.log(
                f"[static-gs-preseg] failed to load sidecar {self._sidecar_path}: {exc}"
            )
            return
        ids = np.asarray(ids).reshape(-1)
        num_points = int(self.model.num_points)
        if ids.shape[0] != num_points:
            raise RuntimeError(
                f"[static-gs-preseg] sidecar/PLY desync: "
                f"sidecar={ids.shape[0]} ids, model.num_points={num_points}. "
                "The dataparser may have permuted points; check Risk #2 in the plan."
            )
        ids_t = torch.from_numpy(ids.astype(np.int64)).to(
            device=self.model.object_instance_ids.device
        )
        with torch.no_grad():
            self.model.object_instance_ids[:, 0] = ids_t
        labeled = int((ids_t > 0).sum().item())
        unique = ids_t.unique().tolist()
        CONSOLE.log(
            f"[static-gs-preseg] loaded sidecar into model.object_instance_ids — "
            f"N={num_points}, labeled={labeled}, ids={unique}"
        )
        self._labeled_instance_count = labeled
        self._num_instances = max(0, len([i for i in unique if i > 0]))

    # ------------------------------------------------------------------
    # AFTER_TRAIN callback — save without Phase 0b
    # ------------------------------------------------------------------

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ):
        # Skip StaticGSPipeline.get_training_callbacks (which registers
        # _finalize_static_training → Phase 0b). Inherit only
        # VanillaPipeline's callbacks + our save.
        callbacks = VanillaPipeline.get_training_callbacks(
            self, training_callback_attributes
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                update_every_num_iters=1,
                func=self._save_post_fusion_state,
            )
        )
        return callbacks

    def _save_post_fusion_state(self, step: int) -> None:
        if self._phase0b_done:
            return
        self._phase0b_done = True

        cache_path = (
            Path(self.config.datamanager.data)
            / self.config.post_fusion_cache_subpath
        )
        ok = save_post_fusion_state(self.model, cache_path)
        if ok:
            obj_count = int(self.model.object_flags.sum().item())
            inst_count = int(
                (self.model.object_instance_ids > 0).any(dim=-1).sum().item()
            )
            CONSOLE.log(
                f"[static-gs-preseg] post-fusion cache written → {cache_path} "
                f"(N={int(self.model.num_points)}, object_flags={obj_count}, "
                f"instance_id>0={inst_count})"
            )
        else:
            CONSOLE.log("[static-gs-preseg] post-fusion cache save failed")
