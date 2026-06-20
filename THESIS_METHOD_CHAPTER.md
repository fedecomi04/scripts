# Method

This chapter describes the proposed system for live, single-camera dynamic
Gaussian Splatting of a robot workspace. The method is organised around two
phases that share a single Gaussian scene representation. The first phase,
**scene initialization** (Section&nbsp;1), reconstructs a static, photo-realistic
Gaussian model of the workspace from an arm-mounted RGB-D camera and augments it
with complete 3D models of the manipulable objects. The second phase, **live
dynamic scene update** (Section&nbsp;2), runs at interactive rates while the robot
manipulates an object: it tracks the rigid 6-DoF motion of the grasped object,
transforms the corresponding subset of Gaussians in lockstep, and feed-forward
completes surfaces that become newly visible as the object is moved.

Throughout, we denote an RGB-D frame at time $t$ by its colour image
$I_t \in \mathbb{R}^{H\times W\times 3}$, metric depth map
$D_t \in \mathbb{R}^{H\times W}$ (stored on disk as 16-bit millimetres, scaled by
$10^{-3}$ to metres), and a binary robot-exclusion mask $M_t \in \{0,1\}^{H\times W}$
($M_t=0$ on the gripper, $1$ elsewhere). The camera intrinsics are
$K = \mathrm{diag}(f_x, f_y, 1)$ with principal point $(c_x, c_y)$, and the
camera-to-world extrinsics are $T_t = \big[\,R_t \mid t_t\,\big] \in SE(3)$,
expressed in the OpenGL convention (camera $+x$ right, $+y$ up, $-z$ forward) as
stored in the dataset. The scene is represented by a set of 3D Gaussians
$\mathcal{G} = \{g_i\}_{i=1}^{N}$, where each Gaussian
$g_i = (\mu_i, q_i, s_i, \alpha_i, \mathbf{c}_i)$ carries a mean
$\mu_i \in \mathbb{R}^3$, an orientation quaternion $q_i \in \mathbb{H}$, an
anisotropic log-scale $s_i \in \mathbb{R}^3$, an opacity logit $\alpha_i$, and
spherical-harmonic colour coefficients $\mathbf{c}_i$. In addition to these
appearance parameters, every Gaussian carries four persistent *identity* buffers
that the two phases write and read: a binary **object flag** $o_i$ marking
Gaussians that belong to a tracked or inserted object, an integer **instance id**
$\ell_i \in \{0,1,\dots,K\}$ ($0$ = static background), a **SAM3D-target flag**,
and an **inserted flag** marking Gaussians created by feed-forward completion.

---

## 1. Scene Initialization

The goal of scene initialization is to produce a warm-started Gaussian scene
$\mathcal{G}_0$ together with its identity buffers, persisted as a single state
snapshot that the dynamic phase loads directly. The phase comprises six stages:
(i) keyframe-filtered RGB-D capture, (ii) incremental geometric fusion into a
seed point cloud, (iii) density-adaptive thinning of that seed, (iv) static
Gaussian optimisation with frozen geometry, (v) text-prompted object discovery
and single-view 3D reconstruction, and (vi) non-rigid registration and
occlusion-aware fusion of each object model into the scene. Stages (i)–(ii) run
*concurrently with capture*; stages (iii)–(vi) run once capture stops.
Algorithm&nbsp;1 summarises the full procedure.

### 1.1 RGB-D Capture and Keyframe Selection

The operator sweeps the arm-mounted camera over the static workspace, ending with
the camera centred on the object to be manipulated. Camera poses are obtained
from the manipulator's forward kinematics and subsequently refined by the fusion
stage (Section&nbsp;1.2). To bound redundancy and fusion cost, incoming frames are
filtered by a greedy ORB-SLAM-style keyframe selector. A new frame with pose
$(R, t)$ is admitted as a keyframe only if it is sufficiently far from *every*
previously retained keyframe $(R_j, t_j)$ in either translation or rotation:

$$
\text{admit}(R,t) \iff \nexists\, j : \; \lVert t - t_j \rVert_2 \le \tau_t \;\wedge\;
d_{SO(3)}(R, R_j) \le \tau_r,
$$

where $d_{SO(3)}(R,R_j) = \arccos\!\big(\tfrac{1}{2}(\mathrm{tr}(R^\top R_j) - 1)\big)$
is the geodesic angle, $\tau_t = 2\,\text{cm}$, and $\tau_r = 20^\circ$. The
disjunction (a frame is *redundant* only if it is close in **both** translation
and rotation) retains viewpoint-diverse frames during slow sweeps while
suppressing near-duplicates.

**Simulated-sensor noise.** When the workspace is captured in simulation the depth
sensor returns a noise-free $z$-buffer, which would create a sim-to-real gap. To
bridge it, the captured depth is corrupted at the publisher by a noise model
calibrated to our real stereo camera (a ZED&nbsp;X): per-pixel zero-mean axial
Gaussian noise with depth-dependent standard deviation
$\sigma_z(z) = \sigma_0 + k\,z^2$ ($\sigma_0 \approx 1.5\,\text{mm}$,
$k \approx 0.5\,\text{mm/m}^2$, an upper-bound fit to the noisiest real capture),
plus $\sim$1% random dropouts and a range gate. The model is enabled by default
and reproduces the grainy, range-dependent character of real stereo depth, so the
depth filtering and range gating described below are exercised under realistic
conditions rather than on idealised geometry.

### 1.2 Incremental Geometric Fusion

The retained keyframes are fused online into a single metric point cloud that
seeds the Gaussian scene. A background worker maintains a running global model
and, for each new keyframe, performs pose refinement followed by volumetric
integration. Because the dataset stores poses in the OpenGL convention while the
fusion back-end (Open3D) operates in the OpenCV convention (camera $+y$ down,
$+z$ forward), every pose is first converted by the fixed change of basis

$$
T^{\text{cv}} = T^{\text{gl}} \cdot \mathrm{diag}(1, -1, -1, 1).
$$

**Pose refinement (ICP).** For each frame the valid depth pixels
($0.05\,\text{m} < D_t < 2.0\,\text{m}$, the working-range cap) are back-projected to a world-frame point
cloud, decimated (stride 4) and voxel-downsampled at $1\,\text{cm}$, and aligned
to the running global model by coarse-to-fine **point-to-plane ICP** with
correspondence thresholds and iteration budgets $\{(5\,\text{cm}, 6),
(2\,\text{cm}, 12)\}$. The refined pose $T \cdot T^{\text{cv}}_t$ is accepted only
when the ICP fitness exceeds $0.30$; otherwise the kinematic pose is trusted as-is.
This refinement removes the systematic millimetre-scale drift between the
kinematic poses and the fused geometry (measured median $\approx 1\,\text{mm}$,
maximum $\approx 4\,\text{mm}$ over a representative sweep), so that downstream
training requires no per-image camera-pose optimisation.

**Volumetric integration (TSDF).** At the refined pose, the full-resolution
depth (and colour) frame is integrated into a truncated signed-distance field on
a GPU voxel-block grid with voxel size $2\,\text{mm}$ and truncation $8\,\text{mm}$.
The global ICP model is re-voxelised and its normals re-estimated every five
frames. Only depth within the $[0.05, 2.0]\,\text{m}$ working range is integrated
(`depth_trunc` $= 2\,\text{m}$): the $2\,\text{m}$ far cap deliberately excludes
distant background, both because the real stereo sensor's error grows
quadratically with range (Section&nbsp;1.1) and to bound the seed Gaussian count.
When capture stops, the TSDF is meshed-to-points and exported as the
**seed point cloud** $\mathcal{P}$ (each surface point carrying a fused RGB
colour). The seed is *not* depth-filtered before integration; TSDF's multi-view
averaging and the millimetre voxel quantisation already suppress the per-frame
jitter, so explicit filtering would not change the fused surface.

### 1.3 Density-Adaptive Seed Thinning

The TSDF seed is uniformly dense and far larger than necessary for a Splatfacto
initialisation, with most points lying on distant background walls. We thin it by
a near/far split keyed to the *final* camera position $x_{\text{cam}}$ (the
operator's last, object-centred viewpoint), which is the region of interest for
manipulation. Points within $r_{\text{near}} = 1\,\text{m}$ of $x_{\text{cam}}$
are kept at full TSDF density; all farther points are voxel-downsampled at
$v_{\text{far}} = 1\,\text{cm}$:

$$
\mathcal{P}' = \{\, p \in \mathcal{P} : \lVert p - x_{\text{cam}}\rVert \le r_{\text{near}} \,\}
\;\cup\;
\mathrm{voxel}_{v_{\text{far}}}\big(\{\, p \in \mathcal{P} : \lVert p - x_{\text{cam}}\rVert > r_{\text{near}} \,\}\big).
$$

This typically reduces the seed by an order of magnitude (e.g. $\sim$10&nbsp;M
$\to\sim$1&nbsp;M points) while preserving full fidelity in the workspace.
$\mathcal{P}'$ is written to disk as the Gaussian initialisation cloud.

### 1.4 Static Gaussian Optimisation

A Splatfacto model is initialised by placing one Gaussian at each seed point of
$\mathcal{P}'$ and optimised against the captured keyframes by the standard
photometric objective

$$
\mathcal{L} = (1 - \lambda)\,\lVert I - \hat{I}\rVert_1 \;+\; \lambda\,\big(1 - \mathrm{SSIM}(I, \hat{I})\big),
\qquad \lambda = 0.2,
$$

where $\hat I$ is the differentiably rasterised render. Three design choices
specialise this stage for our setting and are treated as hard invariants:

1. **Frozen geometry.** The learning rate on the Gaussian means is set to zero,
   so positions remain locked on the geometrically-correct TSDF seed; only colour,
   scale, orientation, and opacity are optimised. Allowing means to drift under the
   photometric loss within the short training budget visibly smears the
   reconstruction.
2. **No densification.** Splatfacto's adaptive clone/split/prune strategy is
   replaced by a no-op strategy, so the Gaussian count and identity are fixed
   throughout training. This keeps the per-Gaussian identity buffers valid and
   the seed-to-Gaussian correspondence intact.
3. **No camera-pose optimisation.** Because poses are already ICP-refined
   (Section&nbsp;1.2), the camera optimiser is disabled; leaving it active drifts the
   already-correct extrinsics.

The renderer composites against the fixed simulator sky colour
$(0.86, 0.92, 1.0)$ so that training never has to compensate for a background
mismatch. Training runs for at most $500$ steps and terminates early once an
exponential moving average of the loss
($\bar{\mathcal{L}} \leftarrow 0.9\,\bar{\mathcal{L}} + 0.1\,\mathcal{L}$) falls
below $0.02$ for $8$ consecutive steps (after a $100$-step warm-up), which avoids
over-spending the budget on trivially-converged scenes while letting harder scenes
train to the full budget.

The photometric loss is additionally masked to the working depth range: pixels
whose sensor depth falls outside $(0.05, 2.0]\,\text{m}$ are excluded from the loss
(the same $2\,\text{m}$ cap as the seed, Section&nbsp;1.2), so colour is never fit
to distant or no-return pixels where the seed places no Gaussians. Two
geometry-hygiene steps run alongside training. Because densification is off,
Splatfacto's own scale pruning never fires, so a periodic callback (every $10$
steps) **uniformly shrinks** any Gaussian whose largest world-axis exceeds
$5\,\text{cm}$ down to a $1\,\text{cm}$ target, preserving its shape while removing
the few oversized splats that smear the sparse far band. At the end of training a
one-shot **opacity purge** deletes Gaussians with $\mathrm{sigmoid}(\alpha) < 0.05$
(typically a $25$–$40\%$ reduction with no visible change), leaving a leaner scene
for the dynamic phase. Both operations subset all six appearance tensors and all
four identity buffers in lockstep so the per-Gaussian identity stays consistent.

### 1.5 Object Discovery and Single-View 3D Reconstruction (Phase 0a)

The static model captures only the *visible* surfaces of each object; the side
facing away from the camera and the contact surface are missing. To obtain
complete object geometry, we reconstruct each prompted object from a single view
and fuse it into the scene. Phase 0a operates on the **last sweep keyframe**
(the object-centred final viewpoint, which yields the cleanest mask and the best
single-view reconstruction input).

**Text-prompted segmentation.** A class-agnostic segmenter (FastSAM-x) proposes
all instance masks in the anchor frame; each mask is scored against the
user-supplied noun prompt by CLIP (ViT-B/32) cosine similarity on the masked
crop. The set of accepted objects is selected automatically, decoupling *how
many* objects to keep from *whether any* match the prompt:

- a mask whose single connected component spans two physical objects is first
  split into its connected components, so each object scores independently;
- the *count* is set by the largest gap in the descending log-probabilities of
  the candidates (a "cliff" between matching and non-matching masks), accepted
  only when the gap ratio exceeds $2.5$;
- a mask is admitted only if its raw CLIP cosine also clears a robust presence
  gate $\mathrm{med}(\cdot) + 3 \cdot 1.4826\,\mathrm{MAD}(\cdot)$ with an absolute
  margin of $0.04$ above the median.

Surviving masks are area-filtered, border-filtered, deduplicated by IoU, and
capped. The deduplication is *containment-aware* — the same object is often
proposed at several extents (e.g. a tight shaft mask and a full shaft-plus-handle
mask), so candidates are processed largest-area first and any later one that is
highly overlapping with, or mostly contained in, an already-kept mask is dropped,
keeping the *fullest* extent of each object. This yields for each object $k$ a
record $(\text{mask}_k, \text{score}_k, \text{bbox}_k, \text{area}_k)$.

**Single-view 3D reconstruction.** Each accepted mask is first enlarged through its
bounding box before reconstruction: the SAM3D input is a *square* crop centred on
the mask, with side $\max(\text{bbox}_w, \text{bbox}_h) + 2\,p$ ($p = 100\,\text{px}$
padding, floored at $800\,\text{px}$), and the metric depth and intrinsics are
cropped to match. The enlargement is deliberate — a crop tight to the silhouette
starves the generator of surrounding scene context and yields stubby or elongated
reconstructions, whereas the padded box supplies enough context for a faithful
shape. On this crop, a single-image 3D object generator (SAM3D) predicts a
*complete* Gaussian point cloud of the object in a canonical object frame, together
with a canonical-to-camera orientation, optionally conditioned on the cropped
metric depth (as a back-projected point map) to anchor the reconstruction to the
real scale. The output is a dense, complete-but-approximate object cloud
$\mathcal{S}_k$ and a canonical rotation $q_k^{\text{can}}$.

### 1.6 Non-Rigid Registration and Occlusion-Aware Fusion (Phase 0b)

Each generated object cloud $\mathcal{S}_k$ is a complete but only *approximately*
metric model, whereas the masked depth back-projection
$\mathcal{T}_k = \Pi^{-1}(\text{mask}_k, D, K, T)$ is a *partial* but metrically
accurate scan of the same object's visible side (here $\Pi^{-1}$ denotes
mask-gated back-projection to world coordinates, with a $5\,\text{MAD}$ depth
outlier scrub). A single rigid transform cannot conform the approximate complete
model to the accurate partial scan, so we register them in two steps.

**Rigid initialisation.** $\mathcal{S}_k$ is first rotated by its canonical
orientation, isotropically scaled by the ratio of bounding-box diagonals
$\text{diag}(\mathcal{T}_k)/\text{diag}(\mathcal{S}_k)$, and translated so that its
robust centroid (5–95th percentile trimmed mean) coincides with that of
$\mathcal{T}_k$.

**Non-rigid deformation.** Using this rigid alignment as initialisation, a Neural
Deformation Pyramid (NDP) non-rigidly warps the complete cloud onto the partial
scan. NDP fits a hierarchy of $\mathrm{Sim}(3)$ deformation fields (three levels,
up to $500$ Adam iterations per level, $6000$-point subsampling) by minimising a
truncated Chamfer distance between the warped source and the target. The result
is a warped cloud whose *visible* side matches the real geometry while its
*occluded* side retains the generator's completion. NDP is the default backend;
rigid Coherent Point Drift and TEASER++ remain available as alternatives.

**Occlusion-aware culling.** Inserting the entire warped cloud would double the
visible surface (the accurate scan already represents it). We therefore cull the
warped points against the existing scene with two complementary tests, keeping
only the genuinely occluded back of the object:

1. **Proximity de-duplication.** A warped point is removed if it lies within
   $\tau = \max(1.3\,\bar{s},\, 3\,\text{mm})$ of an existing scene Gaussian, where
   $\bar{s}$ is the local Gaussian spacing.
2. **In-front (occlusion) culling.** The accurate back-projected scan is rasterised
   into a depth buffer (with a small dilation); any warped point that projects
   *in front of* this surface along its ray is discarded, since the real scan owns
   the visible side. Points behind the surface — the unobserved back — are kept.

**Identity assignment.** The surviving warped points are inserted into
$\mathcal{G}$ with instance id $\ell = k$, the inserted and SAM3D-target flags set,
and the object flag left unset (the static model does not yet designate an
*active* tracked object — that choice is made in the dynamic phase). The instance
id is additionally propagated to the existing scene Gaussians that lie within the
object's mask slab, so that the visible front and completed back share one
identity.

After all objects are fused, the complete scene — Gaussian appearance parameters
plus all four identity buffers — is serialised to a single warm-start snapshot
`static_state.pt`.

> **Algorithm 1 — Scene Initialization**
> **Input:** RGB-D stream $\{(I_t, D_t, M_t, T_t)\}$, object text prompts $\{w_k\}$
> **Output:** warm-start Gaussian scene $\mathcal{G}_0$ with identity buffers
> 1. **for** each incoming frame $(I_t, D_t, T_t)$ *(concurrent with capture)*:
> 2.    **if** $\text{admit}(R_t, t_t)$ by the $\tau_t,\tau_r$ keyframe gate **then**
> 3.        $T_t \leftarrow \textsc{ICP-Refine}(D_t, T_t,\, \text{global model})$
> 4.        $\textsc{TSDF-Integrate}(I_t, D_t, T_t)$ at $2\,\text{mm}$ voxel
> 5. $\mathcal{P} \leftarrow \textsc{TSDF-Finalize}()$;  $\mathcal{P}' \leftarrow \textsc{AdaptiveThin}(\mathcal{P}, x_{\text{cam}})$
> 6. Initialise one Gaussian per point of $\mathcal{P}'$
> 7. **for** step $= 1 \dots 500$ (with EMA early-stop): optimise colour/scale/quat/opacity; **means LR $=0$**, no densification, no camera-opt
> 8. $A \leftarrow$ last sweep keyframe
> 9. $\{\text{mask}_k\} \leftarrow \textsc{FastSAM+CLIP}(A, \{w_k\})$ with automatic count/presence gating
> 10. **for** each object $k$:
> 11.    $\mathcal{S}_k, q_k^{\text{can}} \leftarrow \textsc{SAM3D}(A, \text{mask}_k, D_A)$ *(complete model)*
> 12.    $\mathcal{T}_k \leftarrow \Pi^{-1}(\text{mask}_k, D_A, K, T_A)$ *(accurate partial scan)*
> 13.    $\mathcal{S}_k \leftarrow \textsc{RigidInit}(\mathcal{S}_k, q_k^{\text{can}}, \mathcal{T}_k)$;  $\;\mathcal{S}_k \leftarrow \textsc{NDP}(\mathcal{S}_k \to \mathcal{T}_k)$
> 14.    $\mathcal{S}_k \leftarrow \textsc{ProximityCull} \circ \textsc{InFrontCull}(\mathcal{S}_k, \mathcal{G}, \mathcal{T}_k)$
> 15.    Insert $\mathcal{S}_k$ into $\mathcal{G}$ with $\ell = k$; propagate $\ell$ to the masked front
> 16. **return** serialise $\mathcal{G}$ and identity buffers to `static_state.pt`

---

## 2. Live Dynamic Scene Update

In the dynamic phase the robot manipulates one object while the camera continues
to observe the scene. The system loads the warm-start snapshot and, for every
incoming frame, (i) estimates the rigid 6-DoF pose increment of the grasped
object from sparse feature correspondences, (ii) rigidly transforms the object's
Gaussians by that pose, (iii) periodically detects regions of the live image that
the current model fails to explain, and (iv) feed-forward completes those regions
with new Gaussians. The phase is a pure inference-time *runtime*: **no per-step
gradient descent is performed on any Gaussian parameter** (all learning rates are
zero), so the scene changes only through the tracker's rigid update and
feed-forward insertion. This is essential — gradient descent during tracking would
fight the rigid estimator. Algorithm&nbsp;2 summarises one tick.

### 2.1 Warm Initialisation and Depth Conditioning

The snapshot `static_state.pt` is loaded, restoring the Gaussian appearance
parameters and the four identity buffers. One global runtime setting is enforced:
PyTorch's cuDNN autotuner (`cudnn.benchmark`) is **disabled**. The tracker
operates on a per-frame, object-cropped image whose resolution changes almost
every tick; with autotuning enabled, cuDNN re-benchmarks its convolution
algorithms on each new input shape, which we measured to inflate feature
extraction from $\sim$14&nbsp;ms to several hundred milliseconds per tick. The
frame source is abstracted so that the same runtime serves both a recorded
dataset (iterating stored frames) and a live stream (lock-free polling of a
shared-memory ring buffer fed by the camera-publisher process).

**Depth filtering.** Each incoming depth frame is cleaned by a hole-aware
$5\times5$ **median** followed by a weight-corrected **bilateral** filter (invalid
zero-depth pixels are held out so they are never bled into valid neighbours). The
median removes stereo *flying pixels* — silhouette pixels whose depth jumps to the
background and would otherwise back-project to a Gaussian floating off the surface
— and the bilateral pass smooths residual surface jitter while preserving edges.
The filter is enabled by default. Its placement is rate-aware: in recorded mode it
is applied once at the batch source, so the tracker *and* the feed-forward decoder
both consume the cleaned depth; in live mode it is applied **off the tracker
thread**, at the feed-forward site, so it never adds latency to the tracking loop
(the tracker there runs on the raw sensor depth, whose per-pixel jitter the robust
RANSAC solve already tolerates). The static seed is not filtered, for the reason
given in Section&nbsp;1.2.

### 2.2 Sparse Feature Tracking

We estimate object motion by matching learned sparse features between a set of
*anchor* views of the object and the current frame, then solving for a rigid
transform from the resulting 3D–3D correspondences.

**D0 bootstrap.** On the first dynamic frame the active object is selected — by
the operator through an interactive picker, or automatically as the instance
nearest the camera — and its instance id $\ell^\star$ fixes the object flag,
$o_i \leftarrow \mathbb{1}[\ell_i = \ell^\star]$. The current poses of the object's
Gaussians are stored as the **reference pose**
$\{(\mu_i^{\text{ref}}, q_i^{\text{ref}})\}_{o_i=1}$, against which all future
motion is composed. XFeat keypoints are then extracted on the *full* rendered
frame and filtered, after extraction, to those falling inside the object mask and
the gripper-keep region. (Extracting on a pre-masked image is avoided: zeroing
pixels outside the mask corrupts the convolutional descriptors of boundary
keypoints and destroys matching.) Each surviving keypoint is back-projected to a
3D world point via the depth map, and the collection forms the reference anchor
$\mathcal{A}_0$. An anchor stores the keypoints, their descriptors, their 3D
world points, the camera orientation at capture, and the object-to-world rotation
at capture.

**Per-tick estimation.** For a new frame the object's previous pose is projected
to predict its image bounding box; the RGB, depth, and camera intrinsics are
**cropped** to that box (padded by $150\,\text{px}$ and clamped to the frame), which
focuses extraction on the object and bounds cost. XFeat extracts up to
$1024$ keypoints on the crop, which are matched against the most appropriate
anchor (Section&nbsp;2.3) by the LighterGlue graph matcher (with confidence-based
early exit disabled, since varying the exit layer per tick changes the match set
and induces pose jitter). Matches are filtered to (a) those landing inside the
gripper-keep region and (b) those landing inside the rendered object footprint
dilated by a search radius — the object-mask post-filter that prevents the pose
from "pinning" to static background once the object is grasped and lifted.
Matched keypoints are back-projected to 3D, giving correspondences
$\{(p_j^{a}, p_j)\}$ between the anchor's object frame and the current world frame.

**Robust rigid solve.** The rigid transform between the matched 3D point sets is
estimated by RANSAC over closed-form Kabsch fits. Each iteration draws a minimal
3-point sample, solves

$$
(R_\ell, t_\ell) = \arg\min_{R \in SO(3),\, t}\ \textstyle\sum_j \lVert R\,p_j^{a} + t - p_j\rVert^2
$$

in closed form via the SVD of the cross-covariance
$\sum_j (p_j^{a}-\bar{p}^{a})(p_j-\bar{p})^\top = U\Sigma V^\top$, with
$R_\ell = V\,\mathrm{diag}(1,1,\det(VU^\top))\,U^\top$ and
$t_\ell = \bar{p} - R_\ell\,\bar{p}^{a}$, and counts inliers within
$25\,\text{mm}$. We use $32$ RANSAC iterations (the estimator is deterministically
seeded, so additional iterations do not reduce jitter), refit on the inlier set,
and require at least $12$ inliers for a successful estimate. The per-tick anchor
result $(R_\ell, t_\ell)$ is composed onto the anchor's stored object pose
$(R_a, t_a)$ to yield the cumulative object pose relative to the reference frame:

$$
R \leftarrow R_\ell\,R_a, \qquad t \leftarrow R_\ell\,t_a + t_\ell.
$$

### 2.3 Multi-Anchor Keyframe Management

A single reference view is insufficient once the object rotates away from its D0
appearance, because sparse matching is viewpoint-dependent. The tracker therefore
maintains a growing pool of anchors and selects, each tick, the one whose stored
viewpoint is closest to the current predicted viewpoint. Crucially, the relevant
quantity is the object's orientation **as seen from the camera**, not its absolute
world orientation, since matching degrades whenever the *relative* viewpoint
changes — whether the object moved, the camera moved, or both. We define the
relative orientation

$$
R_{\text{rel}} = R_{\text{cam}}^{\top}\,R_{\text{obj}},
$$

and select the anchor minimising the geodesic distance between its relative
orientation and the predicted one, optionally penalised by an on-screen scale
ratio. A **new anchor** is captured whenever no existing anchor covers the current
view, i.e. when the current relative orientation differs by more than
$22.5^\circ$ from every anchor, or the camera-to-object distance ratio exceeds
$1.3$. A fresh anchor reuses the *current* frame's already-extracted full-image
keypoints (post-filtered to the object), back-projects them at the current
cumulative pose, and appends them to the pool. Because anchors are only created on
successful ticks, the pool densifies exactly along the trajectory the object
actually traverses.

### 2.4 Output Pose Regularisation

The raw RANSAC/Kabsch pose exhibits a few-millimetre, few-degree per-tick wander
on a stationary object, caused by tick-to-tick variation in the matched point set
rather than by real motion. Two output-side regularisers are available; both act
**only** on the pose returned to the scene-update step, never on the tracker's
internal cumulative pose used for anchor selection and next-tick prediction, so
that smoothing can never destabilise tracking itself.

The **static-hold filter** (enabled by default) addresses the stationary wander
without introducing motion lag. It keeps a sliding window of the last $15$
successful poses; when the *net* trend across the window is below
$18\,\text{mm}$ and $6^\circ$ — i.e. the object is genuinely stationary — it
outputs the per-axis median pose over the window instead of the current estimate,
collapsing the residual jitter to near zero. Real motion accumulates in the
window and passes through unchanged once it exceeds the trend gate, so the only
cost is a small onset dead-band. A constant-velocity SE(3) error-state **Kalman
filter** (12-state: position, velocity, rotation error, angular velocity; with an
innovation "snap" gate for reacquisitions) is also implemented but disabled by
default, as it could not achieve zero-lag smoothing on the jerky pick-and-place
motions of our setting.

### 2.5 Rigid Scene Update

Given the (regularised) object pose $(R, t)$ relative to the reference frame, the
object's Gaussians are transformed in lockstep. For every Gaussian with $o_i = 1$,
the mean and orientation are recomputed from the stored reference:

$$
\mu_i \leftarrow R\,\mu_i^{\text{ref}} + t, \qquad
q_i \leftarrow \mathrm{normalise}\big(q(R) \otimes q_i^{\text{ref}}\big),
$$

where $q(R)$ is the quaternion of $R$ and $\otimes$ is the Hamilton product.
Composing every update against the *fixed* reference pose (rather than
incrementally on the previous tick) prevents drift accumulation across the
trajectory. The update is performed under a lock shared with the visualisation
thread so that rendering never observes a half-transformed object. The background
Gaussians ($o_i = 0$) and feed-forward inserts (instance id $999$, Section&nbsp;2.7)
are left untouched.

### 2.6 Change Detection

As the object moves it reveals workspace surfaces that the static model never
observed (e.g. the table beneath the object, or the object's previously-occluded
back). A change-detection module (CDN) localises these regions by comparing the
*rendered* scene to the *live* image at the current camera. The model is rendered
at the live pose to obtain $\hat{I}_t$; the live image $I_t$ and render are
compared over the **valid region** only — excluding the gripper (via $M_t$) and
the tracked object's footprint, both of which are expected to differ. A pixel that
the model renders as empty background (accumulated alpha below $0.5$) is, in
general, ambiguous: it may be a *fillable hole* — a real near surface the static
model simply never reconstructed — or *genuine void* beyond the scene. The two are
disambiguated by the **live sensor depth**: a low-coverage pixel is admitted to
the change region only when the live depth is a valid near surface
($0.05$–$3.0\,\text{m}$); true void (zero/sky/out-of-range depth) is gated out. This
lets the system flag and later fill the workspace surfaces revealed under a lifted
object without firing spuriously on the empty space above the scene. Comparison
uses a multi-scale structural-dissimilarity score on the luminance channel, built
as a three-level pyramid with **coarse-heavy** weights $(0.15, 0.30, 0.55)$ from
fine to coarse. The coarse weighting is deliberate: the rendered scene is
inherently softer than the live sensor image (frozen geometry, low-order SH,
thinned seed), so a full-resolution structural comparison reads this sharpness
mismatch as "change" everywhere; weighting the coarse bands suppresses that
texture noise while genuine content changes survive.

Pixels whose dissimilarity exceeds $0.07$ form a raw change mask, which is cleaned
by morphological closing (radius $10$), opening (radius $3$), and a minimum-area
connected-component filter ($76\,\text{px}$ at the pooled comparison grid, which
keeps every component above the area floor rather than only the largest), then
re-intersected with the valid region. To prevent a compounding feedback loop — where a few spurious specks are
inserted, render worse on the next tick, and flag a larger region — the cleanup
returns an **empty** mask rather than the raw mask when cleaning removes
everything. Finally the dilated object footprint is subtracted, so only true
background/revealed regions remain.

### 2.7 Feed-Forward Surface Completion

When change is detected (evaluated on a fixed cadence — every $10$ ticks, subject
to a minimum wall-clock gap — rather than every frame, to bound cost), the changed
region is completed with new Gaussians on a background thread. The default decoder
is a feed-forward Gaussian predictor (AnySplat); a direct RGB-D back-projection
decoder is also available.

The cleaned change mask is first **cull-cleaned**: any existing Gaussian that
projects in front of the live sensor depth by more than $2\,\text{mm}$ inside the
change region — and is *not* part of a tracked object — is deleted, since it is a
stale occluder the sensor now sees through; the change mask is then re-rendered.
The mask is decomposed into its top-$3$ connected components by area. For each
component the decoder predicts a set of Gaussians for the revealed surface; in the
AnySplat path the predicted cloud is reprojected into the world frame using the
*scene* intrinsics (after an ICP refinement of the local scene-to-sensor
alignment), with per-Gaussian scales enlarged by a factor of $2.0$ and voxel
de-duplication disabled, a combination tuned to produce gap-free yet smooth fills.
Every inserted batch passes a scale-hygiene filter: any Gaussian whose largest axis
exceeds $5\,\text{cm}$ is **uniformly shrunk** to that cap (preserving aspect, so
anisotropic splats are not flattened), which prevents the occasional far/background
prediction with a huge scale from smearing the scene. In the RGB-D path, each
masked pixel is instead back-projected to a disc-shaped Gaussian oriented by the
local depth normal, with depth-discontinuity ("cliff") and sensor-leak pixels
removed.

The decoded Gaussians are inserted into $\mathcal{G}$ with a reserved instance id
($999$), with the object and inserted flags set. This identity makes them
**frozen but static**: the rigid scene update (Section&nbsp;2.5) skips them because
their instance id is not the tracked $\ell^\star$, and the zero learning rates
mean they are never optimised. They simply persist as part of the scene from the
moment of insertion onward, progressively completing the workspace as
manipulation proceeds.

### 2.8 Visualisation

The live scene is streamed to a browser client following the server-side
rasterise-and-push pattern. The browser holds no native splat handles; instead a
background render thread reads each connected client's 6-DoF camera pose, renders
the current Gaussian scene server-side (the same rasteriser used for change
detection), and pushes the resulting full-frame RGB image to that client as a
background image. One atomic full-frame replacement per push means there is no
intermediate scene-rebuild state, eliminating the per-write flicker that the
alternative client-side splat-handle path exhibited (its WebGL renderer remounts
the canvas on every property write). A shared model lock — held by the pipeline
around every tracker write and feed-forward insertion, and by the render thread
around each render — prevents the viewer from observing a half-updated scene. The
trade-off is that the same GPU now also serves viewer frames; this is acceptable
because the render is decoupled onto its own thread and only fires while a client
is connected.

> **Algorithm 2 — Per-Frame Dynamic Update**
> **Input:** live/recorded frame $(I_t, D_t, M_t, T_t)$, warm scene $\mathcal{G}$, anchor pool $\mathcal{A}$
> **Output:** updated scene $\mathcal{G}$
> 1. **if** first dynamic frame **then**
> 2.    $\ell^\star \leftarrow \textsc{SelectObject}()$;  $\;o_i \leftarrow \mathbb{1}[\ell_i = \ell^\star]$
> 3.    store reference $\{(\mu_i^{\text{ref}}, q_i^{\text{ref}})\}_{o_i=1}$;  $\;\mathcal{A} \leftarrow \{\textsc{SeedAnchor}(I_t, D_t, T_t)\}$;  **return**
> 4. $\text{box} \leftarrow$ project previous object pose;  crop $(I_t, D_t, K)$ to $\text{box}{+}150\,\text{px}$
> 5. $\mathcal{K} \leftarrow \textsc{XFeat}(I_t^{\text{crop}})$ *(up to 1024 keypoints)*
> 6. $a \leftarrow \textsc{SelectAnchor}(\mathcal{A}, R_{\text{rel}})$;  $\;\mathcal{M} \leftarrow \textsc{LighterGlue}(\mathcal{A}_a, \mathcal{K})$
> 7. filter $\mathcal{M}$ to gripper-keep $\cap$ dilated object footprint;  back-project to 3D
> 8. $(R_\ell, t_\ell) \leftarrow \textsc{RANSAC-Kabsch}(\mathcal{M})$ *(32 iters, 25 mm, $\ge$12 inliers)*
> 9. **if** success **then** $R \leftarrow R_\ell R_a,\ t \leftarrow R_\ell t_a + t_\ell$;  $\;(R,t)\leftarrow\textsc{StaticHold}(R,t)$
> 10.    **for** $i$ with $o_i = 1$: $\mu_i \leftarrow R\mu_i^{\text{ref}} + t,\ \ q_i \leftarrow \mathrm{norm}(q(R)\otimes q_i^{\text{ref}})$
> 11.    **if** view moved $> 22.5^\circ$ / scale $> 1.3$ from all anchors **then** append new anchor to $\mathcal{A}$
> 12. **if** feed-forward due (every 10 ticks) **then** *(background thread)*
> 13.    $\hat I_t \leftarrow \textsc{Render}(\mathcal{G}, T_t)$;  $\;C \leftarrow \textsc{MS-SSIM-Change}(\hat I_t, I_t)$ over valid region
> 14.    $C \leftarrow \textsc{Clean}(C)$;  subtract object footprint;  cull stale occluders;  re-render $C$
> 15.    **for** each of top-3 components of $C$: $\,\mathcal{G} \mathrel{+}= \textsc{Decode}(I_t, D_t, \text{component})$ with id $999$
> 16. $\textsc{PushToViewer}(\mathcal{G})$

---

## Notation and Default Parameters

| Symbol / setting | Meaning | Default |
|---|---|---|
| $\tau_t,\ \tau_r$ | keyframe translation / rotation gate | $2\,\text{cm}$ / $20^\circ$ |
| TSDF voxel, truncation | fusion grid resolution | $2\,\text{mm}$ / $8\,\text{mm}$ |
| ICP stages, fitness gate | coarse-to-fine point-to-plane ICP | $\{(5\text{cm},6),(2\text{cm},12)\}$, $0.30$ |
| $r_{\text{near}},\ v_{\text{far}}$ | adaptive-thin near radius / far voxel | $1\,\text{m}$ / $1\,\text{cm}$ |
| depth working range | seed + training-mask cap (min / max) | $0.05\,\text{m}$ / $2.0\,\text{m}$ |
| sim depth noise | $\sigma_0$ / $k$ in $\sigma_z=\sigma_0+kz^2$ | $1.5\,\text{mm}$ / $0.5\,\text{mm/m}^2$ |
| depth filter | median / bilateral, on by default | $5\times5$ / edge-preserving |
| static steps, early-stop | Splatfacto budget / EMA loss gate | $500$ / $\bar{\mathcal L}<0.02$ for $8$ |
| means LR (static / dynamic) | Gaussian-position learning rate | $0$ / $0$ |
| background colour | render/composite sky colour | $(0.86, 0.92, 1.0)$ |
| static hygiene | scale-shrink cap / opacity purge | $5\to1\,\text{cm}$ / $\alpha<0.05$ |
| CLIP gate | count cliff ratio / presence margin | $2.5$ / $0.04$ |
| NDP | levels / iters / subsample | $3$ / $500$ / $6000$ |
| cull $\tau$ / in-front band | proximity / occlusion cull | $\max(1.3\bar s, 3\text{mm})$ / $0\,\text{mm}$ |
| XFeat top-$k$ / crop pad | per-tick keypoints / bbox padding | $1024$ / $150\,\text{px}$ |
| RANSAC iters / inlier / min | rigid solve robustness | $32$ / $25\,\text{mm}$ / $12$ |
| anchor gates | new-anchor rotation / scale | $22.5^\circ$ / $1.3$ |
| static-hold | window / trans / rot gate | $15$ / $18\,\text{mm}$ / $6^\circ$ |
| CDN | pyramid weights / threshold / min-area | $(0.15,0.30,0.55)$ / $0.07$ / $76\,\text{px}$ |
| CDN coverage gate | live-depth near-surface band | $0.05$–$3.0\,\text{m}$ |
| feed-forward | cadence / top-$N$ / AnySplat scale | every $10$ ticks / $3$ / $\times 2.0$ |
| FF insert clamp | max Gaussian axis (uniform shrink) | $5\,\text{cm}$ |
