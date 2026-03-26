#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OPTICAL_TO_OPENGL = np.eye(4, dtype=np.float64)
OPTICAL_TO_OPENGL[:3, :3] = np.diag([1.0, -1.0, -1.0])


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_dataset_root = (
        repo_root
        / "data_teleoperation"
        / "datasets"
        / "dynaarm_gs_depth_mask_01"
        / "2026-03-25_21-53-47"
    )
    default_workspace = (
        repo_root
        / "data_teleoperation"
        / "colmap_pose_compare"
        / "2026-03-25_21-53-47_masked_all60"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run COLMAP on a dataset using per-image masks, then compare COLMAP poses "
            "against both the original simulation poses and the mask-derived poses."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--workspace", type=Path, default=default_workspace)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--masks-dir", type=Path, default=None)
    parser.add_argument("--colmap-bin", type=str, default="colmap")
    parser.add_argument("--camera-model", type=str, default="PINHOLE")
    parser.add_argument("--sift-use-gpu", type=int, default=0)
    parser.add_argument("--matcher", choices=("exhaustive",), default="exhaustive")
    parser.add_argument("--no-run-colmap", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_checked(cmd: List[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def count_colmap_images(images_txt: Path) -> int:
    count = 0
    with open(images_txt, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 10:
                count += 1
                next(handle, None)
    return count


def select_largest_model_dir(colmap_bin: str, model_dirs: List[Path]) -> Path:
    best_dir = model_dirs[0]
    best_count = -1
    for model_dir in model_dirs:
        with tempfile.TemporaryDirectory(prefix="colmap_model_txt_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            subprocess.run(
                [
                    colmap_bin,
                    "model_converter",
                    "--input_path",
                    str(model_dir),
                    "--output_path",
                    str(tmpdir_path),
                    "--output_type",
                    "TXT",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            images_txt = tmpdir_path / "images.txt"
            count = count_colmap_images(images_txt)
            if count > best_count:
                best_count = count
                best_dir = model_dir
    print(f"Selected COLMAP model {best_dir} with {best_count} registered images")
    return best_dir


def extract_intrinsics(meta: Dict) -> Tuple[float, float, float, float]:
    fx = meta.get("fl_x")
    fy = meta.get("fl_y")
    cx = meta.get("cx")
    cy = meta.get("cy")
    if None in (fx, fy, cx, cy):
        raise ValueError("Missing intrinsics fl_x/fl_y/cx/cy in transforms.json")
    return float(fx), float(fy), float(cx), float(cy)


def run_colmap(
    workspace: Path,
    images_dir: Path,
    masks_dir: Path,
    colmap_bin: str,
    camera_model: str,
    camera_params: str,
    sift_use_gpu: int,
) -> Path:
    database_path = workspace / "database.db"
    sparse_root = workspace / "sparse"
    sparse_txt_root = workspace / "sparse_txt"

    if database_path.exists():
        database_path.unlink()
    if sparse_root.exists():
        shutil.rmtree(sparse_root)
    if sparse_txt_root.exists():
        shutil.rmtree(sparse_txt_root)
    sparse_root.mkdir(parents=True, exist_ok=True)

    run_checked(
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.mask_path",
            str(masks_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            camera_model,
            "--ImageReader.camera_params",
            camera_params,
            "--SiftExtraction.use_gpu",
            str(sift_use_gpu),
        ],
        cwd=workspace,
    )
    run_checked(
        [
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            str(sift_use_gpu),
        ],
        cwd=workspace,
    )
    run_checked(
        [
            colmap_bin,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_root),
        ],
        cwd=workspace,
    )

    model_dirs = sorted([path for path in sparse_root.iterdir() if path.is_dir()])
    if not model_dirs:
        raise RuntimeError(f"COLMAP mapper produced no sparse models in {sparse_root}")

    best_model = select_largest_model_dir(colmap_bin=colmap_bin, model_dirs=model_dirs)
    sparse_txt_root.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(best_model),
            "--output_path",
            str(sparse_txt_root),
            "--output_type",
            "TXT",
        ],
        cwd=workspace,
    )
    return sparse_txt_root


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.tolist()
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def parse_colmap_images(images_txt: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Tuple[float, float, int]]]]:
    poses: Dict[str, np.ndarray] = {}
    observations: Dict[str, List[Tuple[float, float, int]]] = {}
    with open(images_txt, "r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 10:
                continue

            qvec = np.array([float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4])], dtype=np.float64)
            tvec = np.array([float(fields[5]), float(fields[6]), float(fields[7])], dtype=np.float64).reshape(3, 1)
            image_name = fields[9]

            rotation = qvec_to_rotmat(qvec)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = rotation
            w2c[:3, 3:] = tvec
            c2w = np.linalg.inv(w2c)
            c2w[:3, 1:3] *= -1.0
            poses[image_name] = c2w

            obs_fields = handle.readline().strip().split()
            image_obs: List[Tuple[float, float, int]] = []
            for idx in range(0, len(obs_fields), 3):
                point3d_id = int(float(obs_fields[idx + 2]))
                if point3d_id == -1:
                    continue
                image_obs.append((float(obs_fields[idx]), float(obs_fields[idx + 1]), point3d_id))
            observations[image_name] = image_obs

    return poses, observations


def parse_colmap_points(points_txt: Path) -> Dict[int, np.ndarray]:
    points: Dict[int, np.ndarray] = {}
    with open(points_txt, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            points[int(fields[0])] = np.array([float(fields[1]), float(fields[2]), float(fields[3])], dtype=np.float64)
    return points


def umeyama_similarity(src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_zero = src_points - src_mean
    dst_zero = dst_points - dst_mean
    covariance = (dst_zero.T @ src_zero) / src_points.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    sign_fix = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        sign_fix[-1, -1] = -1.0
    rotation = U @ sign_fix @ Vt
    scale = np.trace(np.diag(singular_values) @ sign_fix) / np.mean(np.sum(src_zero**2, axis=1))
    translation = dst_mean - scale * (rotation @ src_mean)
    return float(scale), rotation, translation


def align_c2w(c2w: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = rotation @ c2w[:3, :3]
    aligned[:3, 3] = scale * (rotation @ c2w[:3, 3]) + translation
    return aligned


def rotation_angle_deg(relative_rotation: np.ndarray) -> float:
    cos_theta = (np.trace(relative_rotation) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def set_axes_equal(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0:
        radius = 1e-3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_pose_plot(output_path: Path, candidate_positions: np.ndarray, colmap_positions: np.ndarray, title: str, candidate_label: str) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(candidate_positions[:, 0], candidate_positions[:, 1], candidate_positions[:, 2], c="blue", s=18, label=candidate_label)
    ax.scatter(colmap_positions[:, 0], colmap_positions[:, 1], colmap_positions[:, 2], c="red", s=18, label="COLMAP")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    combined = np.concatenate([candidate_positions, colmap_positions], axis=0)
    set_axes_equal(ax, combined)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_candidate(
    candidate_name: str,
    candidate_poses: Dict[str, np.ndarray],
    colmap_poses: Dict[str, np.ndarray],
    colmap_observations: Dict[str, List[Tuple[float, float, int]]],
    colmap_points: Dict[int, np.ndarray],
    intrinsics: Tuple[float, float, float, float],
    output_dir: Path,
) -> Dict:
    names = sorted(set(candidate_poses).intersection(colmap_poses))
    if not names:
        raise RuntimeError(f"No overlapping images for candidate {candidate_name}")

    candidate_centers = np.stack([candidate_poses[name][:3, 3] for name in names], axis=0)
    colmap_centers = np.stack([colmap_poses[name][:3, 3] for name in names], axis=0)
    scale, world_rotation, world_translation = umeyama_similarity(candidate_centers, colmap_centers)

    aligned_poses: Dict[str, np.ndarray] = {}
    translation_errors: List[float] = []
    rotation_errors_deg: List[float] = []
    per_image_reproj: Dict[str, float] = {}
    reprojection_errors_px: List[float] = []
    fx, fy, cx, cy = intrinsics
    rows: List[Dict] = []

    for name in names:
        aligned = align_c2w(candidate_poses[name], scale, world_rotation, world_translation)
        aligned_poses[name] = aligned
        translation_error = float(np.linalg.norm(aligned[:3, 3] - colmap_poses[name][:3, 3]))
        rotation_error_deg = rotation_angle_deg(aligned[:3, :3].T @ colmap_poses[name][:3, :3])
        translation_errors.append(translation_error)
        rotation_errors_deg.append(rotation_error_deg)

        c2w_opencv = aligned @ OPTICAL_TO_OPENGL
        w2c_opencv = np.linalg.inv(c2w_opencv)
        image_errors: List[float] = []
        for x_px, y_px, point3d_id in colmap_observations.get(name, []):
            point_world = colmap_points.get(point3d_id)
            if point_world is None:
                continue
            point_cam = w2c_opencv[:3, :3] @ point_world + w2c_opencv[:3, 3]
            if point_cam[2] <= 1e-6:
                continue
            u_px = fx * (point_cam[0] / point_cam[2]) + cx
            v_px = fy * (point_cam[1] / point_cam[2]) + cy
            image_errors.append(float(np.hypot(u_px - x_px, v_px - y_px)))

        if image_errors:
            reprojection_errors_px.extend(image_errors)
            per_image_reproj[name] = float(np.mean(image_errors))

        rows.append(
            {
                "image_name": name,
                "translation_error": translation_error,
                "rotation_error_deg": rotation_error_deg,
                "mean_reprojection_error_px": per_image_reproj.get(name, ""),
            }
        )

    csv_path = output_dir / f"{candidate_name}_per_frame_errors.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / f"{candidate_name}_aligned_pose_centers.png"
    save_pose_plot(
        output_path=plot_path,
        candidate_positions=np.stack([align_c2w(candidate_poses[name], scale, world_rotation, world_translation)[:3, 3] for name in names], axis=0),
        colmap_positions=colmap_centers,
        title=f"{candidate_name} aligned to COLMAP",
        candidate_label=candidate_name,
    )

    summary = {
        "candidate_name": candidate_name,
        "matched_images": len(names),
        "similarity_scale": scale,
        "similarity_world_rotation_matrix": world_rotation.tolist(),
        "similarity_world_translation_xyz": world_translation.tolist(),
        "translation_mean": float(np.mean(translation_errors)),
        "translation_median": float(np.median(translation_errors)),
        "translation_max": float(np.max(translation_errors)),
        "rotation_mean_deg": float(np.mean(rotation_errors_deg)),
        "rotation_median_deg": float(np.median(rotation_errors_deg)),
        "rotation_max_deg": float(np.max(rotation_errors_deg)),
        "reprojection_mean_px": float(np.mean(reprojection_errors_px)) if reprojection_errors_px else None,
        "reprojection_median_px": float(np.median(reprojection_errors_px)) if reprojection_errors_px else None,
        "reprojection_p95_px": float(np.quantile(reprojection_errors_px, 0.95)) if reprojection_errors_px else None,
        "reprojection_max_px": float(np.max(reprojection_errors_px)) if reprojection_errors_px else None,
        "per_frame_csv": str(csv_path),
        "pose_center_plot": str(plot_path),
    }
    return summary


def build_candidate_poses(transforms: Dict, field_name: str) -> Dict[str, np.ndarray]:
    poses: Dict[str, np.ndarray] = {}
    for frame in transforms.get("frames", []):
        if field_name not in frame:
            continue
        mat = np.array(frame[field_name], dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"Invalid transform shape for {field_name} in {frame.get('file_path')}: {mat.shape}")
        poses[Path(frame["file_path"]).name] = mat
    return poses


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    workspace = args.workspace.resolve()
    images_dir = (args.images_dir.resolve() if args.images_dir else (dataset_root / "rgb").resolve())
    masks_dir = (args.masks_dir.resolve() if args.masks_dir else (dataset_root / "masks").resolve())
    output_dir = workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = load_json(dataset_root / "transforms.json")
    intrinsics = extract_intrinsics(transforms)
    camera_params = ",".join(str(value) for value in intrinsics)
    print(f"Using COLMAP camera params: {camera_params}")
    print(f"Using image dir: {images_dir}")
    print(f"Using mask dir: {masks_dir}")

    sparse_txt_dir = workspace / "sparse_txt"
    if not args.no_run_colmap:
        sparse_txt_dir = run_colmap(
            workspace=workspace,
            images_dir=images_dir,
            masks_dir=masks_dir,
            colmap_bin=args.colmap_bin,
            camera_model=args.camera_model,
            camera_params=camera_params,
            sift_use_gpu=args.sift_use_gpu,
        )

    images_txt = sparse_txt_dir / "images.txt"
    points_txt = sparse_txt_dir / "points3D.txt"
    if not images_txt.exists() or not points_txt.exists():
        raise FileNotFoundError("Expected COLMAP sparse_txt/images.txt and points3D.txt to exist")

    colmap_poses, colmap_observations = parse_colmap_images(images_txt)
    colmap_points = parse_colmap_points(points_txt)

    candidate_specs = [
        ("sim_original", "source_transform_matrix"),
        ("mask_pose_ns_exact_tf", "mask_pose_ns_transform_matrix"),
        ("mask_pose_render", "transform_matrix"),
    ]
    candidate_summaries: List[Dict] = []
    for candidate_name, field_name in candidate_specs:
        poses = build_candidate_poses(transforms, field_name)
        summary = evaluate_candidate(
            candidate_name=candidate_name,
            candidate_poses=poses,
            colmap_poses=colmap_poses,
            colmap_observations=colmap_observations,
            colmap_points=colmap_points,
            intrinsics=intrinsics,
            output_dir=output_dir,
        )
        summary["transform_field"] = field_name
        candidate_summaries.append(summary)

    final_summary = {
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "workspace": str(workspace),
        "colmap_sparse_txt_dir": str(sparse_txt_dir),
        "colmap_registered_images": len(colmap_poses),
        "dataset_frame_count": len(transforms.get("frames", [])),
        "candidates": candidate_summaries,
    }

    summary_path = output_dir / "masked_pose_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    print(json.dumps(final_summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
