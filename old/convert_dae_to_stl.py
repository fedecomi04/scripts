from pathlib import Path
import pymeshlab

out_root = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/stl")
out_root.mkdir(parents=True, exist_ok=True)

dae_files = [
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/dynaarm_description/meshes/0_base_ballbot_mesh.dae",
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/dynaarm_description/meshes/100_shoulder_mesh.dae",
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/dynaarm_description/meshes/200_upperarm_mesh.dae",
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/dynaarm_description/meshes/300_elbow_mesh.dae",
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/dynaarm_description/meshes/400_forearm_mesh.dae",
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/dynaarm_description/meshes/500_wrist1_mesh.dae",
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/robotiq/robotiq_2f_85_gripper_visualization/meshes/collision/robotiq_arg2f_85_outer_knuckle.dae",
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/robotiq/robotiq_2f_85_gripper_visualization/meshes/collision/robotiq_arg2f_85_outer_finger.dae",
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/robotiq/robotiq_2f_85_gripper_visualization/meshes/collision/robotiq_arg2f_85_inner_finger.dae",
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/robotiq/robotiq_2f_85_gripper_visualization/meshes/collision/robotiq_arg2f_85_inner_knuckle.dae",
]

for src in dae_files:
    src = Path(src)
    dst = out_root / (src.stem + ".stl")
    print(f"Converting {src} -> {dst}")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src))
    ms.save_current_mesh(str(dst))