#!/usr/bin/env python3
"""Manual tuning sweep: overlay elbow masks on the real ZED RGB while sweeping
(a) lateral offset Y & Z (perpendicular to viewing axis -> shifts silhouette) and
(b) orientation pitch/yaw/roll (degrees -> rotates silhouette).
Joints need NOT match the frame: the NEAR wrist hardware is rigid to the camera, so its
alignment is meaningful across configs. Pick the tile that lines up the near hardware.
"""
import os, sys, importlib.util
from pathlib import Path
import numpy as np
for a,r in (("float",float),("int",int),("bool",bool),("object",object)):
    if not hasattr(np,a): setattr(np,a,r)
import cv2
os.environ["DGS_MASK_RENDER_SCALE"]="1.0"
ROS_MASK=Path("dynamic_gs2/ros_mask.py").resolve()
spec=importlib.util.spec_from_file_location("ros_mask",ROS_MASK)
rm=importlib.util.module_from_spec(spec); sys.modules["ros_mask"]=rm; spec.loader.exec_module(rm)
OUT=Path("real_hw/mask_out/tune"); OUT.mkdir(parents=True,exist_ok=True)
rgb=cv2.imread("dynamic_gs2/verify/zedm_rgb_realcam.png",cv2.IMREAD_COLOR); H,W=rgb.shape[:2]
intr=rm.CameraIntrinsics(width=W,height=H,fx=732.544,fy=732.544,cx=633.254,cy=362.258)
JOINTS={"EL_FLE":2.3118,"FA_ROT":-1.3555,"SH_FLE":0.2388,"SH_ROT":-1.1818,"WRIST_1":-1.2689,"WRIST_2":0.6095,"finger_joint":0.0625}
gen=rm.RobotMaskGenerator(intr,[0.0,1.0],[JOINTS,JOINTS])
fk=gen.robot.link_fk(cfg={k:v for k,v in JOINTS.items() if k in gen.actuated_joint_names},use_names=True)
T_elbow=fk[rm.REAL_HW_CAMERA_PARENT_LINK].astype(np.float32)
base_xyz=rm.REAL_HW_CAMERA_XYZ.copy(); base_R=rm.REAL_HW_CAMERA_ROT.copy()

def rotd(ax,deg):
    t=np.radians(deg); c,s=np.cos(t),np.sin(t)
    if ax=="x": return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if ax=="y": return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def render(xyz,R):
    off=np.eye(4,dtype=np.float32); off[:3,:3]=R; off[:3,3]=xyz
    gen._ensure_renderer(); gen._update_robot_poses(fk)
    gen.scene.set_pose(gen.camera_node,pose=gen._build_render_camera_pose((T_elbow@off).astype(np.float32),optical=True))
    _,d=gen.renderer.render(gen.scene)
    m=(d==0).astype(np.uint8)*255
    if m.shape[:2]!=(H,W): m=cv2.resize(m,(W,H),interpolation=cv2.INTER_NEAREST)
    return m

def overlay(m,label):
    robot=(m==0).astype(np.uint8)*255
    edge=cv2.morphologyEx(robot,cv2.MORPH_GRADIENT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    ov=rgb.copy(); fill=ov.copy(); fill[robot>0]=(0,0,255); ov=cv2.addWeighted(ov,0.75,fill,0.25,0); ov[edge>0]=(0,255,0)
    cv2.putText(ov,label,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    return ov

def montage(tiles,name,cols=3):
    th,tw=tiles[0].shape[:2]; sc=520.0/tw
    sm=[cv2.resize(t,(int(tw*sc),int(th*sc))) for t in tiles]; sh,sw=sm[0].shape[:2]
    rows=(len(sm)+cols-1)//cols; g=np.full((rows*sh,cols*sw,3),30,np.uint8)
    for i,t in enumerate(sm): r,c=divmod(i,cols); g[r*sh:(r+1)*sh,c*sw:(c+1)*sw]=t
    cv2.imwrite(str(OUT/name),g); print("wrote",name)

# (1) lateral Y sweep (ELBOW Y), +/-30mm step 10
tilesY=[]
for dy in [-30,-20,-10,0,10,20,30]:
    xyz=base_xyz.copy(); xyz[1]+=dy/1000.0
    tilesY.append(overlay(render(xyz,base_R),"dY=%+dmm"%dy))
montage(tilesY,"sweep_Y.png")
# (2) lateral Z sweep (ELBOW Z)
tilesZ=[]
for dz in [-30,-20,-10,0,10,20,30]:
    xyz=base_xyz.copy(); xyz[2]+=dz/1000.0
    tilesZ.append(overlay(render(xyz,base_R),"dZ=%+dmm"%dz))
montage(tilesZ,"sweep_Z.png")
# (3) yaw sweep (rotate about ELBOW Z -> pans silhouette L/R)
tilesYaw=[]
for dyaw in [-6,-4,-2,0,2,4,6]:
    R=rotd("z",dyaw)@base_R
    tilesYaw.append(overlay(render(base_xyz,R),"yaw=%+ddeg"%dyaw))
montage(tilesYaw,"sweep_yaw.png")
# (4) pitch sweep (rotate about ELBOW Y)
tilesPit=[]
for dp in [-6,-4,-2,0,2,4,6]:
    R=rotd("y",dp)@base_R
    tilesPit.append(overlay(render(base_xyz,R),"pitch=%+ddeg"%dp))
montage(tilesPit,"sweep_pitch.png")
print("done")
