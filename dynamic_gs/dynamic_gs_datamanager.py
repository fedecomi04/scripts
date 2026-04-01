from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type

from torch.nn import Parameter

from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path


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

    def get_current_dynamic_frame_name(self) -> str:
        image_path = self.dynamic_manager.train_dataset.image_filenames[self.current_dynamic_frame_idx]
        return image_path.stem

    def get_dynamic_debug_dir(self) -> Path:
        return Path(self.config.data) / self.config.dynamic_subdir / "render_masks_esam"

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
