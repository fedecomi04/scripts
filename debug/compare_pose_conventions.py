#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial.transform import Slerp


OPTICAL_TO_OPENGL = np.eye(4, dtype=np.float64)
OPTICAL_TO_OPENGL[:3, :3] = np.diag([1.0, -1.0, -1.0])

CURRENT_SAVER_ROTATION = np.eye(4, dtype=np.float64)
CURRENT_SAVER_ROTATION[:3, :3] = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class CandidateMetrics:
    name: str
    matched_images: int
    similarity_scale: float
    translation_mean: float
    translation_median: float
    translation_max: float
    rotation_mean_deg: float
    rotation_median_deg: float
    rotation_max_deg: float
    reprojection_mean_px: float
    reprojection_median_px: float
    reprojection_p95_px: float
    reprojection_max_px: float
    per_image_reprojection_mean_px: float
    per_image_reprojection_median_px: float
    per_image_reprojection_max_px: float
    reprojection_observation_count: int
    reprojection_image_count: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_dataset_root = repo_root / "data_teleoperation" / "datasets" / "dynaarm_gs_depth_mask_01" / "2026-03-24_19-20-41"
    default_colmap_workspace = repo_root / "data_teleoperation" / "colmap_pose_compare" / "2026-03-24_19-20-41_even50"
    default_output_dir = repo_root / "data_teleoperation" / "pose_debug" / "2026-03-24_19-20-41"

    parser = argparse.ArgumentParser(description="Compare saved pose conventions against COLMAP and sweep a constant time offset.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--colmap-workspace", type=Path, default=default_colmap_workspace)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--overlay-time-offset-sec", type=float, default=-0.15)
    parser.add_argument("--time-offset-start-sec", type=float, default=-0.30)
    parser.add_argument("--time-offset-stop-sec", type=float, default=0.10)
    parser.add_argument("--time-offset-step-sec", type=float, default=0.05)
    return parser.parse_args()


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


def load_transforms(dataset_root: Path) -> Dict:
    with open(dataset_root / "transforms.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def rotation_angle_deg(relative_rotation: np.ndarray) -> float:
    cos_theta = (np.trace(relative_rotation) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def align_c2w(c2w: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = rotation @ c2w[:3, :3]
    aligned[:3, 3] = scale * (rotation @ c2w[:3, 3]) + translation
    return aligned


def evaluate_candidate(
    candidate_name: str,
    candidate_poses: Dict[str, np.ndarray],
    colmap_poses: Dict[str, np.ndarray],
    colmap_observations: Dict[str, List[Tuple[float, float, int]]],
    colmap_points: Dict[int, np.ndarray],
    intrinsics: Tuple[float, float, float, float],
) -> Tuple[CandidateMetrics, Dict[str, float], Dict[str, np.ndarray]]:
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

    for name in names:
        aligned = align_c2w(candidate_poses[name], scale, world_rotation, world_translation)
        aligned_poses[name] = aligned
        translation_errors.append(float(np.linalg.norm(aligned[:3, 3] - colmap_poses[name][:3, 3])))
        rotation_errors_deg.append(rotation_angle_deg(aligned[:3, :3].T @ colmap_poses[name][:3, :3]))

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

    metrics = CandidateMetrics(
        name=candidate_name,
        matched_images=len(names),
        similarity_scale=scale,
        translation_mean=float(np.mean(translation_errors)),
        translation_median=float(np.median(translation_errors)),
        translation_max=float(np.max(translation_errors)),
        rotation_mean_deg=float(np.mean(rotation_errors_deg)),
        rotation_median_deg=float(np.median(rotation_errors_deg)),
        rotation_max_deg=float(np.max(rotation_errors_deg)),
        reprojection_mean_px=float(np.mean(reprojection_errors_px)),
        reprojection_median_px=float(np.median(reprojection_errors_px)),
        reprojection_p95_px=float(np.quantile(reprojection_errors_px, 0.95)),
        reprojection_max_px=float(np.max(reprojection_errors_px)),
        per_image_reprojection_mean_px=float(np.mean(list(per_image_reproj.values()))),
        per_image_reprojection_median_px=float(np.median(list(per_image_reproj.values()))),
        per_image_reprojection_max_px=float(np.max(list(per_image_reproj.values()))),
        reprojection_observation_count=len(reprojection_errors_px),
        reprojection_image_count=len(per_image_reproj),
    )
    similarity = {
        "scale": scale,
        "world_rotation_matrix": world_rotation.tolist(),
        "world_translation_xyz": world_translation.tolist(),
    }
    return metrics, similarity, aligned_poses


def build_interpolator(frames: List[Dict]) -> Tuple[np.ndarray, Slerp, np.ndarray]:
    stamps = np.array([float(frame["rgb_timestamp_sec"]) for frame in frames], dtype=np.float64)
    rotations = Rotation.from_matrix(np.stack([np.array(frame["transform_matrix"], dtype=np.float64)[:3, :3] for frame in frames], axis=0))
    translations = np.stack([np.array(frame["transform_matrix"], dtype=np.float64)[:3, 3] for frame in frames], axis=0)
    return stamps, Slerp(stamps, rotations), translations


def sample_current_pose(frames: List[Dict], stamps: np.ndarray, slerp: Slerp, translations: np.ndarray, sample_time: float) -> np.ndarray:
    if sample_time <= stamps[0]:
        return np.array(frames[0]["transform_matrix"], dtype=np.float64)
    if sample_time >= stamps[-1]:
        return np.array(frames[-1]["transform_matrix"], dtype=np.float64)

    idx = int(np.searchsorted(stamps, sample_time))
    idx0 = max(0, idx - 1)
    idx1 = min(len(stamps) - 1, idx)
    if idx0 == idx1 or stamps[idx1] == stamps[idx0]:
        return np.array(frames[idx0]["transform_matrix"], dtype=np.float64)

    alpha = float((sample_time - stamps[idx0]) / (stamps[idx1] - stamps[idx0]))
    interp_rotation = slerp([sample_time]).as_matrix()[0]
    interp_translation = (1.0 - alpha) * translations[idx0] + alpha * translations[idx1]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = interp_rotation
    out[:3, 3] = interp_translation
    return out


def sweep_time_offsets(frames: List[Dict], colmap_poses: Dict[str, np.ndarray], start_sec: float, stop_sec: float, step_sec: float) -> List[Dict]:
    stamps, slerp, translations = build_interpolator(frames)
    frame_name_to_stamp = {Path(frame["file_path"]).name: float(frame["rgb_timestamp_sec"]) for frame in frames}
    matched_names = [Path(frame["file_path"]).name for frame in frames if Path(frame["file_path"]).name in colmap_poses]

    results: List[Dict] = []
    dt_values = np.arange(start_sec, stop_sec + 0.5 * step_sec, step_sec, dtype=np.float64)
    for dt_sec in dt_values:
        candidate_poses = {
            name: sample_current_pose(frames, stamps, slerp, translations, frame_name_to_stamp[name] + float(dt_sec))
            for name in matched_names
        }
        candidate_centers = np.stack([candidate_poses[name][:3, 3] for name in matched_names], axis=0)
        colmap_centers = np.stack([colmap_poses[name][:3, 3] for name in matched_names], axis=0)
        scale, world_rotation, world_translation = umeyama_similarity(candidate_centers, colmap_centers)

        translation_errors: List[float] = []
        rotation_errors_deg: List[float] = []
        for name in matched_names:
            aligned = align_c2w(candidate_poses[name], scale, world_rotation, world_translation)
            translation_errors.append(float(np.linalg.norm(aligned[:3, 3] - colmap_poses[name][:3, 3])))
            rotation_errors_deg.append(rotation_angle_deg(aligned[:3, :3].T @ colmap_poses[name][:3, :3]))

        results.append(
            {
                "time_offset_sec": float(dt_sec),
                "matched_images": len(matched_names),
                "translation_mean": float(np.mean(translation_errors)),
                "translation_median": float(np.median(translation_errors)),
                "rotation_mean_deg": float(np.mean(rotation_errors_deg)),
                "rotation_median_deg": float(np.median(rotation_errors_deg)),
            }
        )

    return results


def make_overlay_image(
    image_path: Path,
    aligned_c2w_gl: np.ndarray,
    observations: Iterable[Tuple[float, float, int]],
    colmap_points: Dict[int, np.ndarray],
    intrinsics: Tuple[float, float, float, float],
    output_path: Path,
    label: str,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image {image_path}")

    fx, fy, cx, cy = intrinsics
    c2w_opencv = aligned_c2w_gl @ OPTICAL_TO_OPENGL
    w2c_opencv = np.linalg.inv(c2w_opencv)
    draw_count = 0

    for obs_x, obs_y, point3d_id in observations:
        point_world = colmap_points.get(point3d_id)
        if point_world is None:
            continue
        point_cam = w2c_opencv[:3, :3] @ point_world + w2c_opencv[:3, 3]
        if point_cam[2] <= 1e-6:
            continue

        proj_x = fx * (point_cam[0] / point_cam[2]) + cx
        proj_y = fy * (point_cam[1] / point_cam[2]) + cy
        if not (0 <= proj_x < image.shape[1] and 0 <= proj_y < image.shape[0]):
            continue

        cv2.circle(image, (int(round(obs_x)), int(round(obs_y))), 2, (0, 255, 0), -1)
        cv2.circle(image, (int(round(proj_x)), int(round(proj_y))), 2, (0, 0, 255), -1)
        cv2.line(image, (int(round(obs_x)), int(round(obs_y))), (int(round(proj_x)), int(round(proj_y))), (255, 255, 0), 1, lineType=cv2.LINE_AA)
        draw_count += 1
        if draw_count >= 250:
            break

    cv2.putText(image, label, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(image, "green=COLMAP obs, red=projected, cyan=error", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    colmap_workspace = args.colmap_workspace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = load_transforms(dataset_root)
    frames = transforms["frames"]
    frames_by_name = {Path(frame["file_path"]).name: frame for frame in frames}

    colmap_poses, colmap_observations = parse_colmap_images(colmap_workspace / "sparse_txt" / "images.txt")
    colmap_points = parse_colmap_points(colmap_workspace / "sparse_txt" / "points3D.txt")
    intrinsics = (float(transforms["fl_x"]), float(transforms["fl_y"]), float(transforms["cx"]), float(transforms["cy"]))

    candidate_builders = {
        "current_saved": lambda frame: np.array(frame["transform_matrix"], dtype=np.float64),
        "optical_to_opengl_only": lambda frame: np.array(frame["ros_transform_matrix"], dtype=np.float64) @ OPTICAL_TO_OPENGL,
        "raw_ros_no_conversion": lambda frame: np.array(frame["ros_transform_matrix"], dtype=np.float64),
        "mask_saver_net_rotation": lambda frame: np.array(frame["ros_transform_matrix"], dtype=np.float64) @ CURRENT_SAVER_ROTATION,
    }

    candidate_metrics: List[Dict] = []
    candidate_alignment: Dict[str, Dict] = {}
    aligned_pose_sets: Dict[str, Dict[str, np.ndarray]] = {}

    for candidate_name, build_pose in candidate_builders.items():
        poses = {image_name: build_pose(frame) for image_name, frame in frames_by_name.items() if image_name in colmap_poses}
        metrics, similarity, aligned_poses = evaluate_candidate(
            candidate_name=candidate_name,
            candidate_poses=poses,
            colmap_poses=colmap_poses,
            colmap_observations=colmap_observations,
            colmap_points=colmap_points,
            intrinsics=intrinsics,
        )
        candidate_metrics.append(metrics.__dict__)
        candidate_alignment[candidate_name] = similarity
        aligned_pose_sets[candidate_name] = aligned_poses

    time_sweep = sweep_time_offsets(
        frames=frames,
        colmap_poses=colmap_poses,
        start_sec=args.time_offset_start_sec,
        stop_sec=args.time_offset_stop_sec,
        step_sec=args.time_offset_step_sec,
    )

    overlay_name = max(colmap_observations, key=lambda name: len(colmap_observations.get(name, [])))
    overlay_image_path = dataset_root / "rgb" / overlay_name
    overlay_outputs: Dict[str, str] = {}

    make_overlay_image(
        image_path=overlay_image_path,
        aligned_c2w_gl=aligned_pose_sets["current_saved"][overlay_name],
        observations=colmap_observations[overlay_name],
        colmap_points=colmap_points,
        intrinsics=intrinsics,
        output_path=output_dir / "overlay_current_saved.png",
        label=f"{overlay_name} current_saved",
    )
    overlay_outputs["current_saved"] = str((output_dir / "overlay_current_saved.png").resolve())

    make_overlay_image(
        image_path=overlay_image_path,
        aligned_c2w_gl=aligned_pose_sets["optical_to_opengl_only"][overlay_name],
        observations=colmap_observations[overlay_name],
        colmap_points=colmap_points,
        intrinsics=intrinsics,
        output_path=output_dir / "overlay_optical_to_opengl_only.png",
        label=f"{overlay_name} optical_to_opengl_only",
    )
    overlay_outputs["optical_to_opengl_only"] = str((output_dir / "overlay_optical_to_opengl_only.png").resolve())

    matched_names = [Path(frame["file_path"]).name for frame in frames if Path(frame["file_path"]).name in colmap_poses]
    stamps, slerp, translations = build_interpolator(frames)
    name_to_stamp = {Path(frame["file_path"]).name: float(frame["rgb_timestamp_sec"]) for frame in frames}
    offset_poses = {name: sample_current_pose(frames, stamps, slerp, translations, name_to_stamp[name] + float(args.overlay_time_offset_sec)) for name in matched_names}
    _, _, offset_aligned = evaluate_candidate(
        candidate_name=f"current_saved_time_offset_{args.overlay_time_offset_sec:+.2f}s",
        candidate_poses=offset_poses,
        colmap_poses=colmap_poses,
        colmap_observations=colmap_observations,
        colmap_points=colmap_points,
        intrinsics=intrinsics,
    )
    make_overlay_image(
        image_path=overlay_image_path,
        aligned_c2w_gl=offset_aligned[overlay_name],
        observations=colmap_observations[overlay_name],
        colmap_points=colmap_points,
        intrinsics=intrinsics,
        output_path=output_dir / f"overlay_current_saved_time_offset_{args.overlay_time_offset_sec:+.2f}s.png",
        label=f"{overlay_name} current_saved dt={args.overlay_time_offset_sec:+.2f}s",
    )
    overlay_outputs[f"current_saved_time_offset_{args.overlay_time_offset_sec:+.2f}s"] = str((output_dir / f"overlay_current_saved_time_offset_{args.overlay_time_offset_sec:+.2f}s.png").resolve())

    summary = {
        "dataset_root": str(dataset_root),
        "colmap_workspace": str(colmap_workspace),
        "overlay_image_name": overlay_name,
        "candidate_metrics": candidate_metrics,
        "candidate_similarity_transforms": candidate_alignment,
        "time_offset_sweep": time_sweep,
        "overlay_outputs": overlay_outputs,
        "constants": {
            "optical_to_opengl_rotation": OPTICAL_TO_OPENGL.tolist(),
            "current_saver_rotation": CURRENT_SAVER_ROTATION.tolist(),
            "extra_rotation_current_vs_optical_to_opengl": (CURRENT_SAVER_ROTATION @ np.linalg.inv(OPTICAL_TO_OPENGL)).tolist(),
        },
    }

    summary_path = output_dir / "pose_convention_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
