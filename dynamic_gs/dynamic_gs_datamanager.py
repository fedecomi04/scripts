from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.utils.rich_utils import CONSOLE


class DynamicFrameDataset(InputDataset):
    """Full-image dataset with dedicated depth loading for dynamic frames."""

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["depth_image"]

    def __init__(self, dataparser_outputs, scale_factor=1.0, cache_compressed_images=False):
        super().__init__(
            dataparser_outputs,
            scale_factor=scale_factor,
            cache_compressed_images=cache_compressed_images,
        )
        depth_filenames = dataparser_outputs.metadata.get("depth_filenames")
        if depth_filenames is None:
            raise ValueError("dynamic_scene must provide depth_file_path for every frame.")
        self.depth_filenames = depth_filenames
        self.depth_unit_scale_factor = dataparser_outputs.metadata.get("depth_unit_scale_factor", 1.0)

    def get_metadata(self, data):
        image_idx = data["image_idx"]
        filepath = self.depth_filenames[image_idx]
        height = int(self._dataparser_outputs.cameras.height[image_idx])
        width = int(self._dataparser_outputs.cameras.width[image_idx])
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath,
            height=height,
            width=width,
            scale_factor=scale_factor,
        )
        return {"depth_image": depth_image}


class DynamicFrameFullImageDatamanager(FullImageDatamanager[DynamicFrameDataset]):
    """Typed full-image datamanager for dynamic frames with depth."""


@dataclass
class DynamicGSDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: DynamicGSDataManager)

    data: Optional[Path] = None
    static_subdir: str = "static_scene"
    dynamic_subdir: str = "dynamic_scene"

    enable_static_keyframe_filter: bool = True
    """ORB-SLAM-style greedy keyframe filter applied to the static train
    set once at startup. Frame i is accepted only if there is no already-
    accepted keyframe j with both translation gap <= τ_t AND rotation gap
    <= τ_r. Reduces redundant near-duplicate views before static training
    starts."""
    static_keyframe_translation_m: float = 0.02
    """τ_t in meters. Poses are unscaled (auto_scale_poses=False), so this
    is metric."""
    static_keyframe_rotation_deg: float = 20.0
    """τ_r in degrees, geodesic rotation distance on SO(3)."""

    inner: FullImageDatamanagerConfig = field(
        default_factory=lambda: FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(
                load_3D_points=True,
                eval_mode="all",
                depth_unit_scale_factor=1e-3,
                orientation_method="none",
                center_method="none",
                auto_scale_poses=False,
            ),
            cache_images_type="uint8",
            # CPU cache: BOTH wrapped managers (static + dynamic) cache every
            # frame; at 1920x1200 a ~300-frame dynamic episode is ~8-12 GB on
            # GPU (rgb+depth+mask), which OOMed static-gs on the 16 GB card
            # (PyTorch alone at 12.6 GB before the first refine). Per-step H2D
            # of one frame is ~2-4 ms — negligible against the ~30 ms tick.
            cache_images="cpu",
        )
    )


class DynamicGSDataManager(DataManager):
    """Wrap two FullImageDatamanagers and pin one dynamic frame at a time."""

    config: DynamicGSDataManagerConfig

    def __init__(
        self,
        config: DynamicGSDataManagerConfig,
        device="cpu",
        test_mode="val",
        world_size=1,
        local_rank=0,
        **kwargs,
    ):
        del kwargs
        self.config = config
        self.device = device
        self.test_mode = test_mode
        self.world_size = world_size
        self.local_rank = local_rank

        if config.data is None:
            raise ValueError("dynamic-gs requires --data to point at the root folder.")

        root = Path(config.data)
        static_root = root / config.static_subdir
        dynamic_root = root / config.dynamic_subdir
        if not static_root.exists():
            raise FileNotFoundError(f"Missing static scene folder: {static_root}")
        if not dynamic_root.exists():
            raise FileNotFoundError(f"Missing dynamic scene folder: {dynamic_root}")

        self.static_manager = self._build_manager(static_root, use_depth_dataset=False)
        self.static_total_frames = len(self.static_manager.train_dataset)
        if config.enable_static_keyframe_filter:
            self._filter_static_keyframes(
                translation_thresh_m=config.static_keyframe_translation_m,
                rotation_thresh_deg=config.static_keyframe_rotation_deg,
            )
        self.static_accepted_frames = len(self.static_manager.train_dataset)
        self.dynamic_manager = self._build_manager(dynamic_root, use_depth_dataset=True)
        self.current_dynamic_frame_idx = 0

        if test_mode != "inference" and len(self.dynamic_manager.train_dataset) == 0:
            raise ValueError("dynamic_scene must contain at least one training frame.")

        self.phase = "static"
        self.active_manager = self.static_manager
        self.set_phase("static")
        super().__init__()

    def _build_manager(self, data_path, use_depth_dataset):
        cfg = copy.deepcopy(self.config.inner)
        cfg.data = data_path
        cfg.dataparser.data = data_path
        if use_depth_dataset and hasattr(cfg.dataparser, "load_3D_points"):
            cfg.dataparser.load_3D_points = False
        if use_depth_dataset:
            cfg._target = DynamicFrameFullImageDatamanager
        return cfg.setup(
            device=self.device,
            test_mode=self.test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
        )

    def _filter_static_keyframes(
        self,
        *,
        translation_thresh_m: float,
        rotation_thresh_deg: float,
    ) -> None:
        """ORB-SLAM-style greedy keyframe filter on the static train set.

        Accept frame 0 unconditionally. For each subsequent frame i,
        reject iff there exists an already-accepted keyframe j with
        ``||t_i - t_j|| <= τ_t`` AND ``angle(R_i, R_j) <= τ_r``;
        otherwise accept it. The OR semantics ("far enough in T OR R")
        match ORB-SLAM's keyframe insertion test.

        Cost is ``O(K · N)`` doubles where K is the kept count — runs
        once at startup, before any cached image / param tensor is
        materialized, so the filter incurs no runtime overhead during
        training.
        """
        ds = self.static_manager.train_dataset
        n = len(ds)
        if n <= 1 or translation_thresh_m <= 0.0 or rotation_thresh_deg <= 0.0:
            return

        c2w = ds.cameras.camera_to_worlds.detach().cpu().numpy().astype(np.float64)
        if c2w.ndim != 3 or c2w.shape[1] != 3 or c2w.shape[2] != 4:
            CONSOLE.log(
                f"[dynamic-gs] static keyframe filter skipped: unexpected "
                f"camera_to_worlds shape {tuple(c2w.shape)}"
            )
            return
        R = c2w[:, :3, :3]
        t = c2w[:, :3, 3]
        rot_thresh_rad = float(np.deg2rad(rotation_thresh_deg))
        trans_thresh = float(translation_thresh_m)

        kept_idx: List[int] = [0]
        kept_R: List[np.ndarray] = [R[0]]
        kept_t: List[np.ndarray] = [t[0]]
        for i in range(1, n):
            K_t = np.stack(kept_t, axis=0)
            K_R = np.stack(kept_R, axis=0)
            dt = np.linalg.norm(t[i] - K_t, axis=1)
            traces = np.einsum("ab,kab->k", R[i], K_R)
            cos_theta = np.clip(0.5 * (traces - 1.0), -1.0, 1.0)
            dr = np.arccos(cos_theta)
            near = (dt <= trans_thresh) & (dr <= rot_thresh_rad)
            if not near.any():
                kept_idx.append(i)
                kept_R.append(R[i])
                kept_t.append(t[i])

        if len(kept_idx) == n:
            CONSOLE.log(
                f"[dynamic-gs] static keyframe filter: kept {n}/{n} "
                f"(τ_t={trans_thresh:.4f} m, τ_r={rotation_thresh_deg:.1f}°)"
            )
            return

        keep_tensor = torch.tensor(kept_idx, dtype=torch.long)
        outs = self.static_manager.train_dataparser_outputs
        outs.image_filenames = [outs.image_filenames[i] for i in kept_idx]
        if outs.mask_filenames is not None:
            outs.mask_filenames = [outs.mask_filenames[i] for i in kept_idx]
        outs.cameras = outs.cameras[keep_tensor]
        ds.cameras = ds.cameras[keep_tensor]

        if hasattr(self.static_manager, "train_unsampled_epoch_count"):
            delattr(self.static_manager, "train_unsampled_epoch_count")
        self.static_manager.train_unseen_cameras = self.static_manager.sample_train_cameras()

        CONSOLE.log(
            f"[dynamic-gs] static keyframe filter: kept {len(kept_idx)}/{n} "
            f"(τ_t={trans_thresh:.4f} m, τ_r={rotation_thresh_deg:.1f}°)"
        )

    def set_phase(self, phase):
        self.phase = phase
        self.active_manager = self.static_manager if phase == "static" else self.dynamic_manager
        self.train_dataset = self.active_manager.train_dataset
        self.eval_dataset = self.active_manager.eval_dataset
        self.train_sampler = getattr(self.active_manager, "train_sampler", None)
        self.eval_sampler = getattr(self.active_manager, "eval_sampler", None)
        self.train_dataparser_outputs = self.active_manager.train_dataparser_outputs
        self.includes_time = self.active_manager.includes_time

    def set_dynamic_frame_idx(self, frame_idx: int) -> None:
        num_frames = self.get_num_dynamic_frames()
        if not 0 <= frame_idx < num_frames:
            raise IndexError(f"dynamic frame index {frame_idx} is out of range for {num_frames} frames")
        self.current_dynamic_frame_idx = int(frame_idx)

    def get_num_dynamic_frames(self) -> int:
        return len(self.dynamic_manager.train_dataset)

    def get_dynamic_frame_name(self, frame_idx: int) -> str:
        return self.dynamic_manager.train_dataset.image_filenames[frame_idx].stem

    def get_current_dynamic_frame_name(self) -> str:
        return self.get_dynamic_frame_name(self.current_dynamic_frame_idx)

    def get_initialization_debug_dir(self) -> Path:
        return Path(self.config.data) / self.config.dynamic_subdir / "initialization_debug"

    def get_initialization_artifact_dir(self) -> Path:
        return Path(self.config.data) / self.config.dynamic_subdir / "initialization_artifacts"

    def get_dynamic_debug_dir(self) -> Path:
        return self.get_initialization_debug_dir()

    def _get_dynamic_batch(self, frame_idx: int, split: Literal["train", "eval"]):
        if split == "train":
            dataset = self.dynamic_manager.train_dataset
            cached = self.dynamic_manager.cached_train
        else:
            dataset = self.dynamic_manager.eval_dataset
            cached = self.dynamic_manager.cached_eval

        data = cached[frame_idx].copy()
        data["image"] = data["image"].to(self.device)
        if "mask" in data:
            data["mask"] = data["mask"].to(self.device)

        assert len(dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = dataset.cameras[frame_idx : frame_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = frame_idx
        return camera, data

    def get_current_dynamic_train_batch(self):
        return self._get_dynamic_batch(self.current_dynamic_frame_idx, split="train")

    def get_current_dynamic_eval_batch(self):
        return self._get_dynamic_batch(self.current_dynamic_frame_idx, split="eval")

    @property
    def fixed_indices_eval_dataloader(self):
        if self.phase == "dynamic":
            return [self.get_current_dynamic_eval_batch()]
        return self.active_manager.fixed_indices_eval_dataloader

    def setup_train(self):
        return None

    def setup_eval(self):
        return None

    def forward(self):
        raise NotImplementedError

    def next_train(self, step):
        if self.phase == "dynamic":
            self.train_count += 1
            return self.get_current_dynamic_train_batch()
        return self.active_manager.next_train(step)

    def next_eval(self, step):
        if self.phase == "dynamic":
            self.eval_count += 1
            return self.get_current_dynamic_eval_batch()
        return self.active_manager.next_eval(step)

    def next_eval_image(self, step):
        if self.phase == "dynamic":
            return self.get_current_dynamic_eval_batch()
        return self.active_manager.next_eval_image(step)

    def get_train_rays_per_batch(self):
        if self.phase == "dynamic":
            camera = self.dynamic_manager.train_dataset.cameras[self.current_dynamic_frame_idx].reshape(())
            return int(camera.width[0].item() * camera.height[0].item())
        camera = self.train_dataset.cameras[0].reshape(())
        return int(camera.width[0].item() * camera.height[0].item())

    def get_eval_rays_per_batch(self):
        if self.phase == "dynamic":
            camera = self.dynamic_manager.eval_dataset.cameras[self.current_dynamic_frame_idx].reshape(())
            return int(camera.width[0].item() * camera.height[0].item())
        dataset = self.eval_dataset if self.eval_dataset is not None and len(self.eval_dataset) > 0 else self.train_dataset
        camera = dataset.cameras[0].reshape(())
        return int(camera.width[0].item() * camera.height[0].item())

    def get_datapath(self):
        return self.active_manager.get_datapath()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.active_manager.get_param_groups()

    def get_training_callbacks(self, training_callback_attributes):
        return self.active_manager.get_training_callbacks(training_callback_attributes)
