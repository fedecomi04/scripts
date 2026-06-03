#!/usr/bin/env python3
# Compute the projection-vote labels at MIN_OBJ_VOTES=2 and =3, and paint RED the
# points whose label CHANGES between the two thresholds (everything else keeps the
# real TSDF RGB). Also writes a red-spheres render so the (few) changed points pop.
import os, json, re, sys
import numpy as np, cv2, open3d as o3d

DATASET_DIR = ("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/"
               "datasets/new_env/static_scene")
PRECISE_PLY = ("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/experiments/"
               "icp_fusion_mvp/output/online_seed.ply")
THIS = os.path.dirname(os.path.abspath(__file__))
SEG_IDS_NPZ = os.path.join(THIS, "output", "seg_ids.npz")
OUT_PLY = os.path.join(THIS, "output", "precise_objects_votediff.ply")
RENDER_PNG = os.path.join(THIS, "output", "_votediff_render.png")

DEPTH_SCALE = 1000.0
OCC_TOL_M = 0.02


def abspath(rel):
    return os.path.join(DATASET_DIR, rel.lstrip("./"))


def decide(votes, min_votes):
    """Label = best object iff it has >=min_votes AND beats the background votes."""
    bg = votes[:, 0]
    ov = votes.copy(); ov[:, 0] = 0
    best = ov.argmax(1)
    cnt = ov[np.arange(len(votes)), best]
    return np.where((cnt >= min_votes) & (cnt > bg), best, 0).astype(np.int64)


def main():
    pc = o3d.io.read_point_cloud(PRECISE_PLY)
    world = np.asarray(pc.points, np.float64)
    N = len(world)
    tsdf_rgb = np.asarray(pc.colors, np.float64).copy()
    if tsdf_rgb.shape != world.shape:
        tsdf_rgb = np.full((N, 3), 0.62)
    print(f"[geom] {N:,} points")

    seg_ids = np.load(SEG_IDS_NPZ)["seg_ids"]
    F, H, W = seg_ids.shape
    K = int(seg_ids.max())
    meta = json.load(open(os.path.join(DATASET_DIR, "transforms.json")))
    fx, fy, cx, cy = meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]
    frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", fr["file_path"])[-1]))

    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    homog = np.concatenate([world, np.ones((N, 1))], 1).T
    votes = np.zeros((N, K + 1), np.int32)
    for i, fr in enumerate(frames):
        c2w_cv = np.asarray(fr["transform_matrix"], np.float64) @ flip
        cam = (np.linalg.inv(c2w_cv) @ homog).T[:, :3]
        z = cam[:, 2]; front = z > 1e-6
        u = np.full(N, -1.0); v = np.full(N, -1.0)
        u[front] = fx * cam[front, 0] / z[front] + cx
        v[front] = fy * cam[front, 1] / z[front] + cy
        in_img = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(in_img)[0]
        ui = u[idx].astype(np.int32); vi = v[idx].astype(np.int32); zi = z[idx]
        depth = cv2.imread(abspath(fr["depth_file_path"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
        dmap = depth[vi, ui]
        vis = (dmap > 0) & (np.abs(zi - dmap) < OCC_TOL_M)
        np.add.at(votes, (idx[vis], seg_ids[i][vi[vis], ui[vis]].astype(np.int64)), 1)
        if i % 20 == 0 or i == F - 1:
            print(f"[vote] frame {i + 1}/{F}")

    ids2 = decide(votes, 2)
    ids3 = decide(votes, 4)
    changed = ids2 != ids3
    nch = int(changed.sum())
    print(f"\n[diff] points that change label between votes=2 and votes=4: {nch:,} ({100*nch/N:.4f}%)")
    # what kind of change
    to_bg = int(((ids2 > 0) & (ids3 == 0)).sum())
    print(f"        object(@2) -> background(@3): {to_bg:,}   |   other: {nch - to_bg:,}")

    # colour: TSDF RGB everywhere, RED on the changed points
    out = tsdf_rgb.copy()
    out[changed] = [1.0, 0.0, 0.0]
    pc.colors = o3d.utility.Vector3dVector(np.clip(out, 0, 1))
    o3d.io.write_point_cloud(OUT_PLY, pc)
    print(f"[out] {OUT_PLY}")

    # render: full cloud (TSDF RGB) + enlarged red spheres at the changed points so they pop
    geoms = [o3d.io.read_point_cloud(PRECISE_PLY)]
    if nch:
        ch_pts = world[changed]
        spheres = o3d.geometry.TriangleMesh()
        base = o3d.geometry.TriangleMesh.create_sphere(radius=0.006, resolution=4)
        cap = min(nch, 4000)
        for p in ch_pts[np.random.default_rng(0).choice(nch, cap, replace=False)] if nch > cap else ch_pts:
            s = o3d.geometry.TriangleMesh(base); s.translate(p); spheres += s
        spheres.paint_uniform_color([1, 0, 0]); spheres.compute_vertex_normals()
        geoms.append(spheres)
        if nch > cap:
            print(f"[render] showing {cap:,} of {nch:,} red markers")
    vis = o3d.visualization.Visualizer(); vis.create_window(visible=False, width=1200, height=1000)
    for g in geoms:
        vis.add_geometry(g)
    vc = vis.get_view_control()
    vc.set_lookat([-0.5, 0.19, 0.55]); vc.set_front([0.2, 0.4, -0.9]); vc.set_up([0, 0, 1]); vc.set_zoom(0.14)
    vis.get_render_option().point_size = 2.0
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(RENDER_PNG, do_render=True); vis.destroy_window()
    print(f"[out] {RENDER_PNG}")


if __name__ == "__main__":
    main()
