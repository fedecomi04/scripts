#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState

class JointStateMerger:
    def __init__(self):
        input_topic = rospy.get_param("~input_topic", "/joint_states")
        output_topic = rospy.get_param("~output_topic", "joint_states_full")

        self.pub = rospy.Publisher(output_topic, JointState, queue_size=10)
        self.sub = rospy.Subscriber(input_topic, JointState, self.cb, queue_size=10)

        # Finger-joint SOURCE: "gazebo" reads it from the Gazebo physics service (sim, default);
        # "topic" subscribes to a real gripper joint_states topic (real robot — the service is absent).
        # Either way the value lands in self.last_finger_joint and cb() upserts it + the 5 mimics.
        self.source = rospy.get_param("~gripper_source", "gazebo").lower()
        self.last_finger_joint = 0.0

        if self.source == "topic":
            self.gripper_topic = rospy.get_param("~gripper_topic", "/arm_1/gripper/joint_states")
            self.gripper_joint_name = rospy.get_param("~gripper_joint_name", "finger_joint")
            self.gripper_sub = rospy.Subscriber(
                self.gripper_topic, JointState, self._on_gripper_state, queue_size=10,
            )
            rospy.loginfo("[joint_state_merger] finger source=TOPIC %s (joint '%s')",
                          self.gripper_topic, self.gripper_joint_name)
        else:
            from gazebo_msgs.srv import GetJointProperties
            rospy.wait_for_service("/gazebo/get_joint_properties")
            self.get_joint = rospy.ServiceProxy("/gazebo/get_joint_properties", GetJointProperties)
            self.candidates = rospy.get_param(
                "~gripper_joint_candidates",
                ["finger_joint", "dynaarm_arm::finger_joint", "Dynaarm_Arm::finger_joint"],
            )
            self.finger_poll_hz = rospy.get_param("~finger_poll_hz", 75.0)
            self.last_good_name = None
            self.read_finger_joint_once()
            rospy.Timer(rospy.Duration(1.0 / self.finger_poll_hz), self.poll_finger_joint)
            rospy.loginfo("[joint_state_merger] finger source=GAZEBO (service @ %.0f Hz)",
                          self.finger_poll_hz)

    # -------- real-robot source: cache the finger angle from the gripper topic --------
    def _on_gripper_state(self, msg):
        if not msg.name or not msg.position:
            return
        try:
            i = list(msg.name).index(self.gripper_joint_name)
        except ValueError:
            rospy.logwarn_throttle(
                2.0, "[joint_state_merger] joint '%s' not in %s (names=%s)"
                % (self.gripper_joint_name, self.gripper_topic, list(msg.name)))
            return
        self.last_finger_joint = float(msg.position[i])

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

    def read_finger_joint_once(self):
        try:
            self.last_finger_joint = self.read_finger_joint()
        except Exception as e:
            rospy.logwarn_throttle(2.0, str(e))

    def poll_finger_joint(self, _event):
        self.read_finger_joint_once()

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

        q = self.last_finger_joint

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
