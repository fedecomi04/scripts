#!/usr/bin/env python3
"""Generate a single presentation image summarizing the current dynamic-gs pipeline.

Input:
    Path to a dataset root that contains `static_scene/` and `dynamic_scene/`.

The script reads the saved debug outputs produced by the pipeline and assembles
an overview figure for presentations. It reflects the currently implemented
behavior:
    - D0 is the first frame from `dynamic_scene/transforms.json`
    - CoTracker is initialized on D0 live RGB/depth/mask
    - the last static frame is not used for CoTracker initialization
    - the first rigid transform is estimated on D1, not on D0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


C0 = 0.28209479177387814
BACKGROUND = (245, 244, 240)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (220, 216, 208)
TEXT = (25, 25, 25)
SUBTEXT = (85, 85, 85)
ACCENT = (34, 99, 161)
WARNING = (144, 49, 49)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path, help="Path containing static_scene/ and dynamic_scene/")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <dataset_root>/pipeline_presentation.png",
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(42, bold=True)
SECTION_FONT = load_font(30, bold=True)
PANEL_TITLE_FONT = load_font(24, bold=True)
BODY_FONT = load_font(20)
SMALL_FONT = load_font(17)


def read_transforms_images(scene_dir: Path) -> list[Path]:
    transforms_path = scene_dir / "transforms.json"
    data = json.loads(transforms_path.read_text())
    images = []
    for frame in data.get("frames", []):
        file_path = frame["file_path"]
        if file_path.startswith("./"):
            file_path = file_path[2:]
        images.append((scene_dir / file_path).resolve())
    if not images:
        raise RuntimeError(f"No frames found in {transforms_path}")
    return images


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_initialization_dirs(dataset_root: Path) -> tuple[Path, Path]:
    dynamic_scene = dataset_root / "dynamic_scene"
    image_dir = find_first_existing(
        [
            dynamic_scene / "initialization_debug",
            dynamic_scene / "render_masks_esam",
        ]
    )
    if image_dir is None:
        raise FileNotFoundError("Could not find initialization debug directory.")
    artifact_dir = find_first_existing(
        [
            dynamic_scene / "initialization_artifacts",
            image_dir,
        ]
    )
    if artifact_dir is None:
        raise FileNotFoundError("Could not find initialization artifact directory.")
    return image_dir, artifact_dir


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit_image(image: Image.Image, size: tuple[int, int], background: tuple[int, int, int] = PANEL_BG) -> Image.Image:
    fitted = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    max_width: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_spacing: int = 6,
) -> int:
    x, y = xy
    paragraphs = text.splitlines() or [text]
    for paragraph in paragraphs:
        words = paragraph.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if not lines:
            lines = [""]

        for line in lines:
            draw.text((x, y), line, font=font, fill=fill)
            bbox = draw.textbbox((x, y), line, font=font)
            y += (bbox[3] - bbox[1]) + line_spacing
        y += line_spacing
    return y


def create_placeholder(size: tuple[int, int], title: str, detail: str) -> Image.Image:
    image = Image.new("RGB", size, (250, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=18, outline=PANEL_BORDER, width=2)
    draw.text((24, 20), title, font=PANEL_TITLE_FONT, fill=WARNING)
    draw_wrapped_text(draw, detail, (24, 62), size[0] - 48, BODY_FONT, SUBTEXT)
    return image


def load_or_placeholder(path: Path | None, size: tuple[int, int], missing_title: str) -> Image.Image:
    if path is None or not path.exists():
        return create_placeholder(size, missing_title, f"Missing file: {path}")
    return fit_image(load_rgb_image(path), size)


def binary_mask_from_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return (np.array(image) > 127).astype(np.uint8)


def extract_prompt_points(mask_debug_path: Path) -> np.ndarray:
    image = np.array(Image.open(mask_debug_path).convert("RGB"))
    red_mask = (
        (image[..., 0] > 180)
        & (image[..., 0] > image[..., 1] + 80)
        & (image[..., 0] > image[..., 2] + 80)
    ).astype(np.uint8)
    if red_mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    points = []
    for idx in range(1, count):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 4:
            continue
        points.append(centroids[idx])
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def overlay_points_on_rgb(rgb_path: Path, points: np.ndarray, point_source_size: tuple[int, int] | None = None) -> Image.Image:
    image = load_rgb_image(rgb_path)
    draw = ImageDraw.Draw(image)
    radius = max(3, int(round(0.008 * max(image.size))))

    if point_source_size is None:
        point_source_size = image.size
    sx = image.width / max(point_source_size[0], 1)
    sy = image.height / max(point_source_size[1], 1)

    for point in points:
        x = float(point[0]) * sx
        y = float(point[1]) * sy
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0), outline=(255, 255, 255))
    return image


def overlay_mask_on_rgb(rgb_path: Path, mask_path: Path, color: tuple[int, int, int] = (255, 0, 0), alpha: float = 0.35) -> Image.Image:
    image = load_rgb_image(rgb_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(image.size, resample=Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", image.size, color + (0,))
    overlay_alpha = Image.fromarray((np.array(mask) > 127).astype(np.uint8) * int(round(alpha * 255)), mode="L")
    overlay.putalpha(overlay_alpha)
    return Image.alpha_composite(image, overlay).convert("RGB")


def decode_colors_from_vertex(vertex) -> np.ndarray | None:
    names = set(vertex.keys())
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        features_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        return np.clip(features_dc * C0 + 0.5, 0.0, 1.0)
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)
        if colors.max(initial=0.0) > 1.0:
            colors = colors / 255.0
        return np.clip(colors, 0.0, 1.0)
    return None


def load_vertex_ply(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {path}")
            header_lines.append(line.decode("ascii").rstrip())
            if line == b"end_header\n":
                break

        fmt = None
        vertex_count = None
        property_names: list[str] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise RuntimeError(f"PLY list properties are not supported: {path}")
                property_names.append(parts[-1])

        if vertex_count is None or fmt is None:
            raise RuntimeError(f"Could not parse PLY header: {path}")

        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, "<f4") for name in property_names])
            data = np.fromfile(handle, dtype=dtype, count=vertex_count)
            return {name: data[name].astype(np.float32) for name in property_names}

        if fmt == "ascii":
            rows = np.loadtxt(handle, dtype=np.float32, max_rows=vertex_count)
            if rows.ndim == 1:
                rows = rows[None, :]
            return {name: rows[:, idx] for idx, name in enumerate(property_names)}

        raise RuntimeError(f"Unsupported PLY format `{fmt}` in {path}")


def render_ply_preview(ply_path: Path, size: tuple[int, int] = (900, 900)) -> Image.Image:
    vertex = load_vertex_ply(ply_path)
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    colors = decode_colors_from_vertex(vertex)
    if len(points) == 0:
        return create_placeholder(size, "SAM3D point cloud", f"No points found in {ply_path.name}")

    if len(points) > 12000:
        keep = np.linspace(0, len(points) - 1, num=12000)
        keep = np.unique(np.round(keep).astype(np.int64))
        points = points[keep]
        if colors is not None:
            colors = colors[keep]

    centered = points - points.mean(axis=0, keepdims=True)
    azim = math.radians(35.0)
    elev = math.radians(20.0)
    rot_z = np.array(
        [
            [math.cos(azim), -math.sin(azim), 0.0],
            [math.sin(azim), math.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(elev), -math.sin(elev)],
            [0.0, math.sin(elev), math.cos(elev)],
        ],
        dtype=np.float32,
    )
    rotated = centered @ rot_z.T @ rot_x.T
    xy = rotated[:, :2]
    depth = rotated[:, 2]
    scale = max(float(np.abs(xy).max()), 1e-4)
    xy = xy / scale

    margin = 0.08
    px = ((xy[:, 0] + 1.0) * 0.5) * (1.0 - 2.0 * margin) + margin
    py = ((1.0 - (xy[:, 1] + 1.0) * 0.5) * (1.0 - 2.0 * margin)) + margin
    px = px * (size[0] - 1)
    py = py * (size[1] - 1)

    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    scatter_colors = colors if colors is not None else np.full((len(points), 3), 0.35, dtype=np.float32)
    order = np.argsort(depth)
    radius = max(1, int(round(0.003 * max(size))))
    for idx in order:
        x = float(px[idx])
        y = float(py[idx])
        color = tuple(int(round(v * 255)) for v in np.clip(scatter_colors[idx], 0.0, 1.0))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color + (230,))
    return image


def shrink_mask_for_sampling(mask_np: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask_np)
    if len(xs) == 0:
        return mask_np
    side = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    margin_px = max(1, int(round(0.025 * side)))
    kernel_size = 2 * margin_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    inner_mask = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1) > 0
    return inner_mask if np.any(inner_mask) else mask_np


def compute_fast_points(rgb_path: Path, mask_path: Path, max_points: int = 256) -> np.ndarray:
    rgb = np.array(load_rgb_image(rgb_path))
    mask = binary_mask_from_image(mask_path).astype(bool)
    if rgb.shape[:2] != mask.shape:
        mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    mask = shrink_mask_for_sampling(mask)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = gray.copy()
    gray[~mask] = 0
    detector = cv2.FastFeatureDetector_create(threshold=28, nonmaxSuppression=True)
    keypoints = detector.detect(gray, None)
    points = []
    for keypoint in sorted(keypoints, key=lambda item: item.response, reverse=True):
        x = int(round(keypoint.pt[0]))
        y = int(round(keypoint.pt[1]))
        if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
            continue
        if not mask[y, x]:
            continue
        points.append([float(x), float(y)])
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    points_np = np.asarray(points, dtype=np.float32)
    if len(points_np) <= max_points:
        return points_np
    keep = np.linspace(0, len(points_np) - 1, num=max_points)
    keep = np.unique(np.round(keep).astype(np.int64))
    return points_np[keep]


def build_panel(title: str, image: Image.Image, caption: str, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=18, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    draw.text((18, 14), title, font=PANEL_TITLE_FONT, fill=TEXT)
    image_top = 56
    caption_height = 72
    image_box = (18, image_top, width - 18, height - caption_height - 12)
    fitted = fit_image(image, (image_box[2] - image_box[0], image_box[3] - image_box[1]))
    panel.paste(fitted, (image_box[0], image_box[1]))
    draw_wrapped_text(draw, caption, (18, height - caption_height + 2), width - 36, SMALL_FONT, SUBTEXT, line_spacing=4)
    return panel


def build_text_panel(title: str, body: str, width: int, height: int, accent_color: tuple[int, int, int] = ACCENT) -> Image.Image:
    panel = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=18, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    draw.text((18, 14), title, font=PANEL_TITLE_FONT, fill=accent_color)
    draw_wrapped_text(draw, body, (18, 60), width - 36, BODY_FONT, TEXT)
    return panel


def paste_row(canvas: Image.Image, panels: list[Image.Image], x: int, y: int, gap: int) -> None:
    cursor_x = x
    for panel in panels:
        canvas.paste(panel, (cursor_x, y))
        cursor_x += panel.width + gap


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "static_scene").exists() or not (dataset_root / "dynamic_scene").exists():
        raise FileNotFoundError("Input must contain static_scene/ and dynamic_scene/.")

    output_path = args.output.expanduser().resolve() if args.output else dataset_root / "pipeline_presentation.png"

    static_images = read_transforms_images(dataset_root / "static_scene")
    dynamic_images = read_transforms_images(dataset_root / "dynamic_scene")
    first_static = static_images[0]
    last_static = static_images[-1]
    d0_image = dynamic_images[0]
    d1_image = dynamic_images[1] if len(dynamic_images) > 1 else dynamic_images[0]
    d0_stem = d0_image.stem
    d1_stem = d1_image.stem

    init_debug_dir, init_artifact_dir = resolve_initialization_dirs(dataset_root)
    dynamic_debug_dir = dataset_root / "dynamic_scene" / "debug"

    def init_file(name: str) -> Path:
        return init_debug_dir / f"{d0_stem}_{name}"

    live_input_path = find_first_existing([init_file("live_input.png"), d0_image])
    render_path = find_first_existing([init_file("render.png")])
    change_mask_path = find_first_existing([init_file("change_mask.png")])
    live_mask_debug_path = find_first_existing([init_file("live_object_mask.png")])
    live_mask_binary_path = find_first_existing([init_file("live_object_mask_binary.png"), live_mask_debug_path])
    render_mask_debug_path = find_first_existing([init_file("render_object_mask.png")])
    render_mask_binary_path = find_first_existing([init_file("render_object_mask_binary.png"), render_mask_debug_path])
    sam3d_preview_path = find_first_existing([init_file("sam3d_preview.png")])
    sam3d_crop_render_path = find_first_existing([init_file("sam3d_crop_render.png")])

    live_prompt_points = extract_prompt_points(live_mask_debug_path) if live_mask_debug_path is not None else np.zeros((0, 2), dtype=np.float32)
    render_prompt_points = extract_prompt_points(render_mask_debug_path) if render_mask_debug_path is not None else np.zeros((0, 2), dtype=np.float32)

    live_points_image = (
        overlay_points_on_rgb(live_input_path, live_prompt_points, point_source_size=load_rgb_image(live_mask_debug_path).size)
        if live_input_path is not None and live_mask_debug_path is not None
        else create_placeholder((900, 700), "Live ESAM points", "Missing live RGB/mask debug.")
    )
    live_mask_overlay = (
        overlay_mask_on_rgb(live_input_path, live_mask_binary_path)
        if live_input_path is not None and live_mask_binary_path is not None
        else create_placeholder((900, 700), "Live ESAM mask", "Missing live binary mask.")
    )
    render_points_image = (
        overlay_points_on_rgb(render_path, render_prompt_points, point_source_size=load_rgb_image(render_mask_debug_path).size)
        if render_path is not None and render_mask_debug_path is not None
        else create_placeholder((900, 700), "Render ESAM points", "Missing render RGB/mask debug.")
    )
    render_mask_overlay = (
        overlay_mask_on_rgb(render_path, render_mask_binary_path)
        if render_path is not None and render_mask_binary_path is not None
        else create_placeholder((900, 700), "Render ESAM mask", "Missing render binary mask.")
    )

    sam3d_ply_path = find_first_existing(
        [
            init_artifact_dir / f"{d0_stem}_d0_true_sam3d_raw_output.ply",
            init_artifact_dir / f"{d0_stem}_sam3d_raw_output.ply",
            init_debug_dir / f"{d0_stem}_sam3d_raw_output.ply",
        ]
    )
    sam3d_pointcloud_image = (
        render_ply_preview(sam3d_ply_path) if sam3d_ply_path is not None else create_placeholder((900, 900), "SAM3D point cloud", "SAM3D .ply output not found.")
    )

    cotracker_points = (
        compute_fast_points(live_input_path, live_mask_binary_path)
        if live_input_path is not None and live_mask_binary_path is not None
        else np.zeros((0, 2), dtype=np.float32)
    )
    cotracker_init_image = (
        overlay_points_on_rgb(live_input_path, cotracker_points)
        if live_input_path is not None
        else create_placeholder((900, 700), "CoTracker init", "Missing D0 live RGB image.")
    )
    d1_cotracker_debug_path = find_first_existing(
        [
            dynamic_debug_dir / "cotracker_debug" / f"{d1_stem}_cotracker.png",
            dynamic_debug_dir / f"{d1_stem}_cotracker.png",
        ]
    )

    canvas = Image.new("RGB", (2400, 3920), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 30), "Dynamic-GS Pipeline Overview", font=TITLE_FONT, fill=TEXT)
    draw.text((50, 82), dataset_root.name, font=BODY_FONT, fill=SUBTEXT)

    section_y = 140
    draw.text((50, section_y), "1. Static Scene Optimization", font=SECTION_FONT, fill=ACCENT)
    static_text = build_text_panel(
        "Static phase",
        "The current pipeline first optimizes the static scene with depth initialization, then keeps Gaussian means fixed during the static optimization stage. "
        "No pruning or densification is applied in that phase.",
        width=760,
        height=430,
    )
    static_image_panel = build_panel(
        "First static frame",
        load_rgb_image(first_static),
        f"Loaded from static_scene/transforms.json: {first_static.name}",
        width=1540,
        height=430,
    )
    paste_row(canvas, [static_text, static_image_panel], 50, section_y + 48, 30)

    section_y = 670
    draw.text((50, section_y), "2. Dynamic Scene Initialization (D0)", font=SECTION_FONT, fill=ACCENT)
    draw.text((50, section_y + 42), f"D0 = first frame in dynamic_scene/transforms.json: {d0_stem}", font=BODY_FONT, fill=SUBTEXT)

    row_y = section_y + 90
    panels = [
        build_panel(
            "2.1 Live image",
            load_or_placeholder(live_input_path, (700, 700), "D0 live image"),
            "First dynamic RGB image used in the bootstrap.",
            740,
            420,
        ),
        build_panel(
            "2.1 Render from D0 pose",
            load_or_placeholder(render_path, (700, 700), "D0 render"),
            "Static-scene render from the same D0 camera pose.",
            740,
            420,
        ),
        build_panel(
            "2.1 Change mask",
            load_or_placeholder(change_mask_path, (700, 700), "Change mask"),
            "Change mask computed from the D0 live image and the render above.",
            740,
            420,
        ),
    ]
    paste_row(canvas, panels, 50, row_y, 30)

    row_y += 460
    panels = [
        build_panel(
            "2.2 ESAM on D0 live: prompt points",
            live_points_image,
            "Prompt points recovered from the saved live-mask debug image and overlaid on the D0 RGB image.",
            1135,
            430,
        ),
        build_panel(
            "2.2 ESAM on D0 live: mask overlay",
            live_mask_overlay,
            "Resulting D0 live object mask overlaid on the D0 RGB image.",
            1135,
            430,
        ),
    ]
    paste_row(canvas, panels, 50, row_y, 30)

    row_y += 470
    panels = [
        build_panel(
            "2.3 ESAM on render: prompt points",
            render_points_image,
            "Current implementation refines the rendered object mask on the render itself; the overlaid points are the saved render-side prompt points.",
            1135,
            430,
        ),
        build_panel(
            "2.3 ESAM on render: mask overlay",
            render_mask_overlay,
            "Final rendered object mask overlaid on the rendered RGB image.",
            1135,
            430,
        ),
    ]
    paste_row(canvas, panels, 50, row_y, 30)

    row_y += 470
    sam3d_input_image = load_or_placeholder(
        find_first_existing([sam3d_preview_path, sam3d_crop_render_path]),
        (900, 900),
        "SAM3D input",
    )
    panels = [
        build_panel(
            "2.4 SAM3D input image",
            sam3d_input_image,
            "Image actually given to SAM3D, with the rendered object mask overlaid.",
            1135,
            430,
        ),
        build_panel(
            "2.4 SAM3D raw output (.ply)",
            sam3d_pointcloud_image,
            "Preview rendered from the saved SAM3D Gaussian .ply output.",
            1135,
            430,
        ),
    ]
    paste_row(canvas, panels, 50, row_y, 30)

    row_y += 470
    cotracker_note = (
        f"Current implementation check:\n"
        f"- Last static frame: {last_static.name}\n"
        f"- First dynamic frame D0: {d0_stem}\n"
        f"- CoTracker initializes on D0 live RGB/depth/mask\n"
        f"- It does NOT initialize from the last static frame\n"
        f"- The first rigid transform is estimated on D1 ({d1_stem})"
    )
    cotracker_tracking_panel = (
        build_panel(
            "2.5 CoTracker D0->D1 debug",
            load_rgb_image(d1_cotracker_debug_path),
            "If available, this is the first actual tracked rigid-motion debug image (D0 reference to D1).",
            740,
            430,
        )
        if d1_cotracker_debug_path is not None
        else build_text_panel(
            "2.5 CoTracker D0->D1 debug",
            "No saved D1 CoTracker debug image was found in dynamic_scene/debug/cotracker_debug/. "
            "The presentation still reflects the implemented CoTracker initialization logic.",
            width=740,
            height=430,
            accent_color=WARNING,
        )
    )
    panels = [
        build_panel(
            "2.5 Last static frame",
            load_rgb_image(last_static),
            "Shown here only to clarify the frame boundary. The current CoTracker code does not initialize from this frame.",
            740,
            430,
        ),
        build_panel(
            "2.5 CoTracker initialization on D0",
            cotracker_init_image,
            "FAST features sampled inside the D0 live object mask, matching the current CoTracker initialization behavior.",
            740,
            430,
        ),
        cotracker_tracking_panel,
    ]
    paste_row(canvas, panels, 50, row_y, 30)

    note_panel = build_text_panel("Implementation note", cotracker_note, width=2300, height=180, accent_color=WARNING)
    canvas.paste(note_panel, (50, 3670))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"[generate_pipeline_presentation] saved: {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
