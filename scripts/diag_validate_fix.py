#!/usr/bin/env python3
"""Validate the frame-consistency fix two ways:

(A) EFFICACY: back-project the anchor mask through (1) the ANCHOR depth [the
    fixed pairing — same frame] vs (2) the dataset arm_00039 depth [the buggy
    pairing]. The fixed pairing should be a compact, low-plane-fraction object;
    the buggy one a big, table-heavy blob.

(B) HELPER CORRECTNESS: exercise _append_anchor_as_static_keyframe on a temp
    copy of a real transforms.json and verify the new frame sorts LAST, has the
    right schema, uint16-mm depth, BGR rgb, and that nerfstudio's argsort makes
    it cached_train[-1].

Usage: diag_validate_fix.py <dataset_dir>
"""
import sys, json, shutil, tempfile
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image


def backproj_cam(depth_m, mask, fx, fy, cx, cy):
    ys, xs = np.where(mask & np.isfinite(depth_m) & (depth_m > 1e-4))
    z = depth_m[ys, xs]
    if z.size >= 10:
        med = np.median(z); mad = np.median(np.abs(z - med)) + 1e-6
        k = np.abs(z - med) < 5.0 * 1.4826 * mad
        ys, xs, z = ys[k], xs[k], z[k]
    x = (xs - cx) / fx * z; y = (ys - cy) / fy * z
    return np.stack([x, y, z], -1)


def plane_frac(p):
    if len(p) < 50:
        return 0.0
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(p.astype(np.float64))
    _, inl = pc.segment_plane(0.008, 3, 300)
    return len(inl) / len(p)


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene"
    art = ddir / "dynamic_scene" / "initialization_artifacts"
    dbg = ddir / "dynamic_scene" / "initialization_debug"
    tj = json.load(open(ss / "transforms.json"))
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127
    ai = json.load(open(art / "static0_full_intrinsics.json"))
    anchor_depth = np.array(Image.open(art / "static0_full_depth_meters.tiff")).astype(np.float32)
    ds_depth = np.array(Image.open(ss / "depth" / "arm_00039.tiff")).astype(np.float32) * 1e-3

    print("=== (A) EFFICACY: anchor mask back-projected through each frame's depth ===")
    pa = backproj_cam(anchor_depth, mask, ai["fx"], ai["fy"], ai["cx"], ai["cy"])
    pb = backproj_cam(ds_depth, mask, tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"])
    for name, p in [("FIXED  (anchor mask + ANCHOR depth)", pa),
                    ("BUGGY  (anchor mask + arm_00039 depth)", pb)]:
        ext = (p.max(0) - p.min(0)) * 100
        z = np.percentile(p[:, 2], [5, 95])
        print(f"  {name:42s} N={len(p):>6}  extent(cm)={np.round(ext,1)}  "
              f"zspread={ (z[1]-z[0])*100:.1f}cm  plane%={100*plane_frac(p):.0f}")

    print("\n=== (B) HELPER CORRECTNESS ===")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # import the helper without importing the package __init__ (pulls nerfstudio)
    import importlib.util
    ls = Path(__file__).resolve().parents[1] / "dynamic_gs" / "utils" / "live_session.py"
    # the helper only needs cv2/np/json/os/Path + LiveFrame dataclass; load via a
    # lightweight shim: exec just the function source is overkill — instead copy
    # transforms.json into a temp static_scene and replicate the call by importing
    # the module's symbols that are import-safe.
    from types import SimpleNamespace
    tmp = Path(tempfile.mkdtemp())
    sdir = tmp / "static_scene"; sdir.mkdir(parents=True)
    shutil.copy(ss / "transforms.json", sdir / "transforms.json")
    n_before = len(json.load(open(sdir / "transforms.json"))["frames"])

    # Build a fake anchor LiveFrame-like object.
    H, W = anchor_depth.shape
    rng = np.random.default_rng(0)
    anchor = SimpleNamespace(
        rgb_bgr=(rng.integers(0, 255, (H, W, 3))).astype(np.uint8),
        depth_m=np.nan_to_num(anchor_depth, nan=0.0).astype(np.float32),
        mask_keep=(mask.astype(np.uint8) * 255),
        c2w_4x4=np.eye(4, dtype=np.float64),
    )
    intr = SimpleNamespace(fx=ai["fx"], fy=ai["fy"], cx=ai["cx"], cy=ai["cy"],
                           width=ai["width"], height=ai["height"])

    # exec the helper source in an isolated namespace with its deps
    import cv2, os
    src = ls.read_text()
    start = src.index("def _append_anchor_as_static_keyframe")
    end = src.index("def _prompt_user")
    g = {"json": json, "os": os, "cv2": cv2, "np": np, "Path": Path,
         "Optional": __import__("typing").Optional, "LiveFrame": object}
    exec(src[start:end], g)
    stem = g["_append_anchor_as_static_keyframe"](anchor, intr, sdir)

    meta = json.load(open(sdir / "transforms.json"))
    fr = meta["frames"][-1]
    print(f"  appended stem: {stem}  (frames {n_before} -> {len(meta['frames'])})")
    print(f"  last frame file_path: {fr['file_path']}  has depth/mask keys: "
          f"{'depth_file_path' in fr and 'mask_path' in fr}")
    # sorts last?
    fps = [f["file_path"] for f in meta["frames"]]
    sorts_last = (sorted(fps)[-1] == fr["file_path"])
    print(f"  sorts LAST under lexicographic argsort (==dataparser): {sorts_last}")
    # depth written as uint16 mm?
    dchk = np.array(Image.open(sdir / "depth" / f"{stem}.tiff"))
    print(f"  depth dtype on disk: {dchk.dtype} (expect uint16)  max={dchk.max()} (mm scale)")
    # rgb readable + BGR (we wrote random, just confirm shape/3ch)
    rchk = np.array(Image.open(sdir / "rgb" / f"{stem}.png"))
    mchk = np.array(Image.open(sdir / "masks" / f"{stem}.png"))
    print(f"  rgb shape {rchk.shape}  mask uniq {np.unique(mchk)[:5]}")
    tm = np.asarray(fr["transform_matrix"])
    print(f"  transform_matrix shape {tm.shape}  (expect (4,4))")
    ok = (sorts_last and dchk.dtype == np.uint16 and tm.shape == (4, 4)
          and 'depth_file_path' in fr and 'mask_path' in fr
          and len(meta['frames']) == n_before + 1)
    print(f"\n  HELPER TEST: {'PASS' if ok else 'FAIL'}")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
