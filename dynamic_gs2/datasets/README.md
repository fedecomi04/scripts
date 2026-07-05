# Test datasets

Ready-to-run recorded datasets for the `dynamic_gs2` pipeline. Each contains
**only the essential raw data** — no pregenerated `static_state.pt`, no
`post_dynamic_state.pt`, no timing reports, no run backups — so a run rebuilds
everything from scratch and validates the full pipeline.

## Download

The datasets are hosted on Google Drive (too large for git). Download the ones
you want and unzip them into this directory (`dynamic_gs2/datasets/`):

> **Google Drive:** <PASTE_YOUR_SHARED_DRIVE_LINK_HERE>

Each dataset unzips to `dynamic_gs2/datasets/<name>/` with the layout below.
Nothing here is required to clone or install the pipeline — only to run the
example commands.

## Layout (per dataset)

```
<name>/
├── static_scene/
│   ├── rgb/                          BGR PNG
│   ├── depth/                        uint16 mm TIFF
│   ├── masks/                        uint8 robot-exclusion mask (0=robot, 255=keep)
│   ├── transforms.json               Nerfstudio-formatted, ICP-refined poses
│   └── depth_camera_init_points.ply  TSDF-fused init seed
└── dynamic_scene/
    ├── rgb/ depth/ masks/
    └── transforms.json
```

## Datasets

| Name | Source | Object | Prompt | Trigger frame | Frames (static / dynamic) |
|---|---|---|---|---|---|
| `zed_final` | Real ZED-Mini capture (1920×1080, tiled-floor scene) | banana | `banana` | `--trigger-frame 79` | 135 / 165 |
| `coke_can_sim` | Gazebo sim (800×800) | Coca-Cola can | `coke can` | `--trigger-frame 78` | 87 / 116 |
| `screwdriver_sim` | Gazebo sim, side view, noisy depth | screwdriver | `screwdriver` | default (last static frame) | 21 / 216 |
| `fidget_spinner_sim` | Gazebo sim, noisy depth | fidget spinner | `fidget spinner` | default (last static frame) | 26 / 259 |

Notes on the prompts (from validation):
- `zed_final` — a distinguishing adjective is not needed; `banana` is unambiguous
  against the terracotta floor.
- `coke_can_sim` — use `coke can` (`soda can` also works; bare `can` fails the
  CLIP presence gate).
- `screwdriver_sim` / `fidget_spinner_sim` — the default anchor (last static
  keyframe) works; no `--trigger-frame` needed.

## Running

From the repo root, with the four conda envs set up (see the top-level README):

```bash
# Full pipeline from scratch (static reconstruction + object completion + dynamic replay):
dynamic_gs2/full_recorded.sh dynamic_gs2/datasets/zed_final          banana           --trigger-frame 79
dynamic_gs2/full_recorded.sh dynamic_gs2/datasets/coke_can_sim       "coke can"       --trigger-frame 78
dynamic_gs2/full_recorded.sh dynamic_gs2/datasets/screwdriver_sim    screwdriver
dynamic_gs2/full_recorded.sh dynamic_gs2/datasets/fidget_spinner_sim "fidget spinner"

# After a run has produced static_scene/static_state.pt, inspect it in the viewer
# (no sim needed) — orbit at http://localhost:8081:
dynamic_gs2/view_dynamic.sh dynamic_gs2/datasets/zed_final --ff
```

The trigger frame selects the anchor keyframe that FastSAM+CLIP segments and
Fast-SAM3D completes; it is the 1-based index into the sorted
`static_scene/transforms.json`. Pick a frame where the object is large and
centered in view.
