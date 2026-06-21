#!/usr/bin/env python3
"""Render the elbow mask with WRIST_2 (gripper-rotation joint) +45 and -45 deg, overlaid on the
real ZED RGB. Tells which sign of the 45-deg constant offset matches the real gripper.
Real ZED K. Current camera offset + current joints, only WRIST_2 changed."""
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
OUT=Path("real_hw/mask_out"); OUT.mkdir(parents=True,exist_ok=True)
rgb=cv2.imread("dynamic_gs2/verify/zedm_rgb_realcam.png",cv2.IMREAD_COLOR); H,W=rgb.shape[:2]
intr=rm.CameraIntrinsics(width=W,height=H,fx=732.544,fy=732.544,cx=633.254,cy=362.258)
BASE={"EL_FLE":2.3118,"FA_ROT":-1.3555,"SH_FLE":0.2388,"SH_ROT":-1.1818,"WRIST_1":-1.2689,"WRIST_2":0.6095,"finger_joint":0.0625}

def render_overlay(wrist2, tag):
    J=dict(BASE); J["WRIST_2"]=wrist2
    gen=rm.RobotMaskGenerator(intr,[0.0,1.0],[J,J])
    fk=gen.robot.link_fk(cfg={k:v for k,v in J.items() if k in gen.actuated_joint_names},use_names=True)
    T_elbow=fk[rm.REAL_HW_CAMERA_PARENT_LINK].astype(np.float32)
    off=np.eye(4,dtype=np.float32); off[:3,:3]=rm.REAL_HW_CAMERA_ROT; off[:3,3]=rm.REAL_HW_CAMERA_XYZ
    gen._ensure_renderer(); gen._update_robot_poses(fk)
    gen.scene.set_pose(gen.camera_node,pose=gen._build_render_camera_pose((T_elbow@off).astype(np.float32),optical=True))
    _,d=gen.renderer.render(gen.scene)
    m=(d==0).astype(np.uint8)*255
    if m.shape[:2]!=(H,W): m=cv2.resize(m,(W,H),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT/("mask_%s.png"%tag)), m)
    robot=(m==0).astype(np.uint8)*255
    edge=cv2.morphologyEx(robot,cv2.MORPH_GRADIENT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    ov=rgb.copy(); fill=ov.copy(); fill[robot>0]=(0,0,255); ov=cv2.addWeighted(ov,0.75,fill,0.25,0); ov[edge>0]=(0,255,0)
    cv2.putText(ov,"WRIST_2 %s (%.4f rad)"%(tag,wrist2),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.imwrite(str(OUT/("overlay_%s.png"%tag)), ov)
    print("wrote overlay_%s.png  WRIST_2=%.4f cover%%=%.1f"%(tag,wrist2,100*(robot>0).mean()))
    return ov

d45=np.radians(45.0)
ovp=render_overlay(BASE["WRIST_2"]+d45,"wrist_plus45")
ovm=render_overlay(BASE["WRIST_2"]-d45,"wrist_minus45")
ov0=render_overlay(BASE["WRIST_2"],"wrist_base")
# side-by-side: minus | base | plus
sb=np.hstack([ovm,ov0,ovp]); cv2.imwrite(str(OUT/"wrist45_compare.png"), sb); print("wrote wrist45_compare.png (minus | base | plus)")
