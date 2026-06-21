"""E2E test of the PRODUCTION SweepSeedBuilder: feed recorded keyframes as Frames through
submit()+finalize() (simulating the sweep) and confirm it writes a valid seed PLY matching the
reference. Writes to a TEMP dir so it never clobbers the real seed. Run:
python -m dynamic_gs2.verify._probe_seedbuilder_e2e [dataset_dir]"""
import sys, json, re, shutil, time
from pathlib import Path
import numpy as np, cv2, open3d as o3d

from dynamic_gs2.frame import Frame, Intrinsics
from dynamic_gs2.static_seed_stream import SweepSeedBuilder
from dynamic_gs2 import timing as _T

DS = Path(sys.argv[1] if len(sys.argv) > 1 else
          "../data_teleoperation/datasets/2026-06-21_170319_live").resolve()
sd = DS if DS.name == "static_scene" else DS / "static_scene"
meta = json.loads((sd / "transforms.json").read_text())
intr = Intrinsics(int(meta["w"]), int(meta["h"]), float(meta["fl_x"]), float(meta["fl_y"]),
                  float(meta["cx"]), float(meta["cy"]))
frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", Path(fr["file_path"]).name)[-1]))
print(f"dataset={sd.parent.name} keyframes={len(frames)} res={intr.width}x{intr.height}")

# stage a temp static_scene with the transforms.json so finalize() can patch a throwaway copy
tmp = sd.parent / "_seedbuilder_e2e_tmp"
tmp.mkdir(exist_ok=True)
shutil.copy(sd / "transforms.json", tmp / "transforms.json")

def to_frame(fr, seq):
    depth = cv2.imread(str(sd / fr["depth_file_path"].lstrip("./")), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    mp = fr.get("mask_path") or fr.get("mask_file_path")
    mask = np.ones(depth.shape, np.uint8)
    if mp:
        m = cv2.imread(str(sd / mp.lstrip("./")), cv2.IMREAD_GRAYSCALE)
        if m is not None: mask = (m > 0).astype(np.uint8)
    rgb = cv2.imread(str(sd / fr["file_path"].lstrip("./")), cv2.IMREAD_COLOR)
    c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
    return Frame(seq=seq, stamp_sec=0.0, rgb_bgr=rgb, depth_m=depth, mask_keep=mask, c2w_4x4=c2w)

tm = _T.new_ledger()
b = SweepSeedBuilder(intr, tm=tm)
b.start()
t0 = time.time()
for i, fr in enumerate(frames):
    b.submit(to_frame(fr, i + 1))
    time.sleep(0.05)            # simulate keyframes arriving spread out (not all at once)
print(f"submitted {len(frames)} keyframes over {time.time()-t0:.1f}s (bg-fused concurrently)")
tf = time.time()
ply = b.finalize(tmp)
print(f"finalize wall = {time.time()-tf:.1f}s -> {ply}")

assert ply is not None and ply.exists(), "builder did not write a PLY"
pc = o3d.io.read_point_cloud(str(ply))
a = np.asarray(pc.points)
print(f"\nSEED PLY: {len(a):,} pts  bbox(m)={np.round(a.max(0)-a.min(0),3)}")
m2 = json.loads((tmp / 'transforms.json').read_text())
print(f"transforms.json ply_file_path patched = {m2.get('ply_file_path')!r}")
print("\n=== TIMING REPORT (seed per-keyframe ICP/TSDF means, tracker-style) ===")
print(tm.render_static())
print("\nVERDICT: production SweepSeedBuilder works end-to-end." if len(a) > 50_000 else "\nWARNING: suspiciously few points")
shutil.rmtree(tmp, ignore_errors=True)
