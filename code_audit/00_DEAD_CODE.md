# Dead-code audit — independent verification

Method: repo-wide `grep -rn "<symbol>" dynamic_gs scripts --include=*.py` + `pyproject.toml`.
Accounted for: nerfstudio `method_configs` entry points, training callbacks, monkeypatch
targets, getattr/string dispatch, `__main__` op-dispatch handlers, and the CLAUDE.md
Design Invariants (none of the invariant-protected symbols — `_ZERO_LR_OPTIMIZERS`,
`means` LR, `camera_optimizer.mode`, the four identity buffers,
`apply_rigid_object_transform_from_reference`, the viser-direct push path, background
color — appear among the candidates as dead).

`pyproject.toml` references none of the candidate symbols.

---

## DEAD — only the symbol's own definition (+ stale docstrings) references it

| Symbol | File:line | ref_count | Evidence |
|---|---|---|---|
| `_force_viser_direct_push` | dynamic_gs/dynamic_gs_pipeline_base.py:1605 | 1 (def only) | Subclasses call `_push_viser_direct_transforms` directly; alias has zero call sites. |
| `_render_from_camera_at_scale` | dynamic_gs/dynamic_gs_pipeline_base.py:1704 | 1 (def only) | CDN uses `_render_from_camera` (full-res); reduced-res CDN reverted. |
| `_oneshot_ff_due` | dynamic_gs/dynamic_gs_pipeline_base.py:966 | 1 (def only) | `get_train_loss_dict` (:987) inlines the identical predicate. |
| `feedforward_anchor_frame` | dynamic_gs/dynamic_gs_pipeline_base.py:207 | 1 (decl only) | Config field, zero reads. |
| `feedforward_video_out` | dynamic_gs/dynamic_gs_pipeline_base.py:209 | 1 (decl only) | Config field, zero reads; CLAUDE.md confirms no video writer implemented. |
| `feedforward_video_fps` | dynamic_gs/dynamic_gs_pipeline_base.py:210 | 1 (decl only) | Config field, zero reads. |
| `prepare_dynamic_update` | dynamic_gs/dynamic_gs_model.py:1915 | 1 def + own return-key + 2 docstrings | No call site; only def, its `"prepare_dynamic_update_substeps"` key (:2104), doc-comments esam.py:251 / static_gs_model.py:27. Superseded by XFeat tracker + CDN. |
| `refresh_dynamic_state_after_insertion` | dynamic_gs/dynamic_gs_model.py:1842 | 1 (def only) | Zero callers. |
| `_propagate_instance_membership` | dynamic_gs/dynamic_gs_model.py:1391 | 1 (def only) | Multi-object propagation never wired (still roadmap). |
| `_get_render_projection_params` | dynamic_gs/dynamic_gs_model.py:1297 | 1 (def only) | Unused projection helper. |
| `apply_rigid_object_transform` (bare) | dynamic_gs/dynamic_gs_model.py:923 | 1 def + docstrings | All other hits resolve to `..._from_reference` (live write-path, NOT dead). Bare variant has zero callers. |
| `_get_esam_model` | dynamic_gs/dynamic_gs_model.py:1909 | def + 1 call in dead `prepare_dynamic_update` (:2017) | Dead by transitivity. |
| `_spawn_depth_republisher` | dynamic_gs/utils/live_ros_publisher.py:687 | 1 (def only) | Never called; raw 32FC1 depth subscribed directly; `_depth_republisher_proc` stays None. |
| `_total_shm_bytes` | dynamic_gs/utils/live_ros_publisher.py:265 | 1 (def only) | `__init__` recomputes inline (`HEADER_BYTES + NUM_SLOTS*slot_bytes`). |
| `LivePublisher.wait_first_frame` | dynamic_gs/utils/live_ros_publisher.py:1046 | def + op-handler self-call (:1480-1481) | `op=="wait_first_frame"` branch exists but reader never sends it (grep `wait_first` in live_shm_reader = empty). Unreachable. |
| `_KeyframeFilter.num_kept` | dynamic_gs/utils/live_ros_publisher.py:349 | 1 (def only) | Counting done via `len(self._record_frames_written)`. |
| `LiveShmSubscriber.get_singleton` | dynamic_gs/utils/live_shm_reader.py:334 | def + reads own `_singleton` | No external `.get_singleton()` caller. |
| `LiveShmSubscriber._singleton` | dynamic_gs/utils/live_shm_reader.py:265 | written :330, read only by dead `get_singleton` | Orphan support state. |
| `LiveShmSubscriber.save_anchor_for_sam3` | dynamic_gs/utils/live_shm_reader.py:520 | def + own _send/raise | live_session.py uses module-level `_save_anchor_for_sam3` (:182) on the peeked LiveFrame. |
| `LiveShmSubscriber.save_anchor_intrinsics_and_depth` | dynamic_gs/utils/live_shm_reader.py:529 | def + own raise | live_session.py uses module-level `_save_anchor_intrinsics_and_depth` (:207). |
| `_wipe_live_root` (live_session) | dynamic_gs/utils/live_session.py:175 | 1 (def only) | Distinct from live_ros_publisher.py:1383 (that one IS called at :1416). Session wipe done via `LiveShmSubscriber(wipe_live_root=True)` (:567). |
| `_compose_keep_region` | dynamic_gs/utils/xfeat_motion.py:1281 | 1 (def only) | Keep-region recipe inlined in `estimate_and_advance`. |
| `_pre_mask_image` | dynamic_gs/utils/xfeat_motion.py:1225 | def + 1 docstring (:499) | D0/anchor paths extract full-image then post-filter; no pre-mask. |
| `prepare_tracking_rgb_gpu` | dynamic_gs/utils/tracker_common.py:100 | def + docstrings | xfeat_motion has its own `_prepare_tracking_rgb_gpu`. Not in __init__ exports. |
| `prepare_tracking_rgb` | dynamic_gs/utils/tracker_common.py:84 | def + docstrings | Only xfeat hit (:1438) is a docstring mention, not a call. |
| `sample_mask_points` | dynamic_gs/utils/tracker_common.py:165 | def + docstring | KLT residue; XFeat never samples mask points. |
| `_shrink_mask_for_sampling` | dynamic_gs/utils/tracker_common.py:153 | def + 1 call in dead `sample_mask_points` (:178) | Dies with sample_mask_points. |
| `_subsample_points` | dynamic_gs/utils/tracker_common.py:145 | def + 2 calls in dead `sample_mask_points` (:209,:212) | Dies with sample_mask_points. |
| `filter_points_in_image` | dynamic_gs/utils/tracker_common.py:215 | def + docstring | Legacy filter; XFeat filters inline. |
| `filter_points_by_mask_array` | dynamic_gs/utils/tracker_common.py:229 | def + docstring | Legacy filter; XFeat filters inline. |
| `filter_gaussians_by_component_mask` | dynamic_gs/utils/anysplat_decode.py:832 | 1 (def only) | Filtering done inside `reproject_anysplat_to_scene`. |
| `_world_to_image_opengl` | dynamic_gs/utils/anysplat_decode.py:502 | def + 1 call in dead `filter_gaussians_by_component_mask` (:856) | Dies as a pair. |
| `maybe_flush_ff_handle` | dynamic_gs/utils/viser_direct.py:518 | def + 1 docstring (:36) | Sibling stubs are called from pipeline_base; this one is not. |
| `build_active_mask_center_only` | dynamic_gs/utils/active_mask.py:680 | def + 2 in __init__ (import+__all__) | Exported-but-unused; zero call sites. |
| `_largest_extent` | dynamic_gs/utils/sam3d_fusion.py:453 | 1 (def only) | Live extent helper is `_bbox_diagonal`. |
| `_compute_iou` (fastsam copy) | dynamic_gs/utils/fastsam_segmentation.py:49 | 1 (def only) | sam_worker uses the sam3_segmentation copy (`_seg._compute_iou`); fastsam dedup uses inline IoU at :411. |
| `_make_o3d_with_normals` | dynamic_gs/utils/rgbd_fusion_init.py:114 | 1 (def only) | Both normal-estimation sites inline `pc.estimate_normals(...)`. |
| `DynamicKeyframeFilter` | dynamic_gs/utils/keyframe_filter.py:33 | class + import/__all__ + 1 docstring | Never instantiated (`DynamicKeyframeFilter(` = 0). Docstring (pipeline_base:25) says DROPPED from recorded path; live uses inlined `_KeyframeFilter`. |
| `DynamicKeyframeFilter.bulk_filter` | dynamic_gs/utils/keyframe_filter.py:86 | def + 1 docstring | `.bulk_filter(` = 0 hits. Dies with class (`accept`/`num_kept`/`reset` reachable only from it). |

---

## UNCERTAIN — reachable only via a framework override / config branch never exercised in any live or recorded path

NOT clean deletes: each sits behind a nerfstudio Model override surface or a config field
whose value is never set to the triggering case. Live-dead, but a config change / eval
path could reach them.

| Symbol | File:line | ref_count | Evidence |
|---|---|---|---|
| `_get_optim_mask` | dynamic_gs/dynamic_gs_model.py:842 | def + 1 call in `get_loss_dict` dynamic branch (:2372) | Dynamic `get_loss_dict` (phase!="static") never hit: trainer's overridden `get_train_loss_dict` returns a zero-loss dummy (pipeline_base:998) and never calls `model.get_loss_dict`. `get_loss_dict` is a nerfstudio override. |
| `_set_optim_mask` | dynamic_gs/dynamic_gs_model.py:833 | def + 2 calls (:1861 dead `refresh_...`, :2098 dead `prepare_dynamic_update`) | Both callers dead; writes `change_mask_image` consumed only by the loss-bypassed dynamic `get_loss_dict`. |
| `_masked_rgb_l1` | dynamic_gs/dynamic_gs_model.py:1903 | def + 1 call in dynamic `get_loss_dict` (:2377) | Same unreachable branch as `_get_optim_mask`. |
| `_depth_diff_score` | dynamic_gs/utils/active_mask.py:373 | def + 1 call (:569) under `mode=='depth'` | `build_change_mask` live callers pass `mode='rgb'` (ChangeMaskConfig.mode default 'rgb'; change_mask_mode default 'rgb' at dynamic_gs_model.py:77). Branch never taken — config-selectable. |
| `_depth_outlier_score` | dynamic_gs/utils/active_mask.py:419 | def + 1 call (:586) under `mode=='depth_outlier'` | Only fires for a never-configured mode. |
| `combine_object_masks` | dynamic_gs/utils/active_mask.py:216 | def + import/__all__ + 1 call in dead `prepare_dynamic_update` (:2052) | Sole real call site is inside dead `prepare_dynamic_update`; dead-by-transitivity but exported public API. |
| `query_esam_mask` | dynamic_gs/utils/esam.py:156 | def + import/__all__ | Exported + imported into dynamic_gs_model but never called; live path uses `query_esam_mask_pair`. |
| `_run_esam_query` | dynamic_gs/utils/esam.py:133 | def + 1 call in dead `query_esam_mask` (:202) | Transitively dead. |
| `_select_esam_mask` | dynamic_gs/utils/esam.py:121 | def + 1 call in dead `_run_esam_query` (:152) | Transitively dead; re-implemented inline in live `query_esam_mask_pair`. |
| `change_mask_depth_threshold` (StaticGSModelConfig) | dynamic_gs/static_gs_model.py:184 | 1 (decl only) | Zero `config.change_mask_*` reads on static model/pipeline; only DynamicGSModelConfig.change_mask_* read, inside dead `prepare_dynamic_update`. Partial dead copy (no `change_mask_mode` field). |
| `change_mask_rgb_threshold` (StaticGSModelConfig) | dynamic_gs/static_gs_model.py:185 | 1 (decl only) | Same. |
| `change_mask_use_rgb` (StaticGSModelConfig) | dynamic_gs/static_gs_model.py:186 | 1 (decl only) | Same. |
| `change_mask_blur_kernel_size` (StaticGSModelConfig) | dynamic_gs/static_gs_model.py:187 | 1 (decl only) | Same. |
| `change_mask_blur_sigma` (StaticGSModelConfig) | dynamic_gs/static_gs_model.py:188 | 1 (decl only) | Same. |
| `change_mask_filter_radius` (StaticGSModelConfig) | dynamic_gs/static_gs_model.py:189 | 1 (decl only) | Same. |
| `change_mask_min_component_size` (StaticGSModelConfig) | dynamic_gs/static_gs_model.py:190 | 1 (decl only) | Same. |
| `OnlineFusion.idx` (property) | dynamic_gs/utils/online_fusion.py:530 | property + `self._impl.idx` read | No functional external consumer. bench_gpu_fusion.py/profile_fusion.py do `fuser.idx += 1` expecting a plain attribute, but the wrapper property has no setter → AttributeError against the post-2026-06-01 dispatcher. Stale, not a clean delete. |
| `timed` (contextmanager) | dynamic_gs/utils/timing_ledger.py:86 | 1 (def only) | All timing sites use raw `record()`; `.timed(`/`_tl.timed` = 0 hits outside module. Advertised as ergonomic API in docstring — possibly intentionally retained. |

---

## LIVE — referenced / part of live public API (or write-only-but-assigned)

| Symbol | File:line | ref_count | Evidence |
|---|---|---|---|
| `self._accepted_dynamic_frames` | dynamic_gs/dynamic_gs_pipeline_recorded.py:120 | 3 writes (recorded:120, live:132, base:458), 0 reads | Functionally dead (write-only) but a base-class attribute written by BOTH subclasses — removal must touch all 3 sites, not a single-symbol delete. |
| `last_anchor_idx_used` | dynamic_gs/utils/xfeat_motion.py:430 | 3 writes (:430,:590,:866), 0 reads | Write-only diagnostic; no consumer. Functionally dead, multiple write sites. |
| `last_used_fallback_anchor` | dynamic_gs/utils/xfeat_motion.py:431 | 3 writes (:431,:591,:809), 0 reads | Write-only diagnostic; no consumer. |
| `last_pool_size` | dynamic_gs/utils/xfeat_motion.py:433 | 3 writes (:433,:562,:1008), 0 reads | Write-only diagnostic; no consumer. |
| `current_track_count` (property) | dynamic_gs/utils/xfeat_motion.py:456 | property + 3 internal reads (:588,:623,:707) | LIVE: read each tick as track_count_before/after, fed into MotionEstimate. NOT dead. |
| `LabeledSeed` | dynamic_gs/utils/preseg_seed.py:49 | class + return-type of live `build_labeled_seed` (:405,:543) | LIVE: return value of `build_labeled_seed` (called static_gs_preseg_pipeline.py:250, read by attribute). Public API. Its `seg_ids_path` field is written (:546) but never read — dead field on a live dataclass, harmless. |
| `filter_depth_torch` `median=`/`bilateral=` kwargs | dynamic_gs/utils/depth_filter.py:87 | prod callers (recorded:196, base:3253) pass NO kwargs; only compare_depth_filters_zed.py:36 uses them | Function is LIVE; per-stage split kwargs carry no production behavior (default full filter). CLAUDE.md: "unused — all callers run the full filter." Keep params (default-valued, used by bench). |
