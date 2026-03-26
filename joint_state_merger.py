#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetJointProperties

class JointStateMerger:
    def __init__(self):
        self.pub = rospy.Publisher("/joint_states_full", JointState, queue_size=10)
        self.sub = rospy.Subscriber("/joint_states", JointState, self.cb, queue_size=10)

        rospy.wait_for_service("/gazebo/get_joint_properties")
        self.get_joint = rospy.ServiceProxy("/gazebo/get_joint_properties", GetJointProperties)

        self.candidates = rospy.get_param(
            "~gripper_joint_candidates",
            ["finger_joint", "dynaarm_arm::finger_joint", "Dynaarm_Arm::finger_joint"],
        )
        self.last_good_name = None

    def read_finger_joint(self):
        names = [self.last_good_name] if self.last_good_name else []
        names += [n for n in self.candidates if n != self.last_good_name]

        for name in names:
            try:
                res = self.get_joint(name)
                if res.position:
                    self.last_good_name = name
                    return float(res.position[0])
            except Exception:
                pass
        raise RuntimeError("Could not read finger_joint from Gazebo")

    def upsert(self, msg, name, value):
        if name in msg.name:
            i = msg.name.index(name)
            msg.position[i] = value
            if len(msg.velocity) == len(msg.name):
                msg.velocity[i] = 0.0
            if len(msg.effort) == len(msg.name):
                msg.effort[i] = 0.0
        else:
            msg.name.append(name)
            msg.position.append(value)
            if len(msg.velocity) == len(msg.name) - 1:
                msg.velocity.append(0.0)
            if len(msg.effort) == len(msg.name) - 1:
                msg.effort.append(0.0)

    def cb(self, msg):
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)

        try:
            q = self.read_finger_joint()
        except Exception as e:
            rospy.logwarn_throttle(2.0, str(e))
            return

        # Root joint used by the URDF
        self.upsert(out, "finger_joint", q)

        # Safer: also publish the mimic joints explicitly
        self.upsert(out, "left_inner_knuckle_joint", q)
        self.upsert(out, "left_inner_finger_joint", -q)
        self.upsert(out, "right_inner_knuckle_joint", q)
        self.upsert(out, "right_inner_finger_joint", -q)
        self.upsert(out, "right_finger_joint", q)

        self.pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("joint_state_merger")
    JointStateMerger()
    rospy.spin()