#!/usr/bin/env python
import copy
import sys
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Quaternion, Point, TransformStamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_geometry_msgs import do_transform_pose
import tf2_ros

from moveit_msgs.srv import GetPlanningScene
from std_srvs.srv import Empty, EmptyRequest

from gpnet.srv import GPnetGraspPlanner
import moveit_commander
from moveit_commander import PlanningSceneInterface, MoveGroupCommander


class GraspObject(object):
    def __init__(self):
        rospy.loginfo("Initalizing...")
        self.gs = GraspServer()

        self.pick_gui = rospy.Service("/grasp_obj", Empty, self.start_grasp_obj)

        rospy.loginfo("Initiate MoveGroupCommander")
        self.group = MoveGroupCommander('arm_torso')

        rospy.loginfo("Setting up publishers to gripper and head controller...")
        self.head_cmd = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=1)
        self.gripper_cmd = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=1)

        self.scene = PlanningSceneInterface()
        rospy.loginfo("Connecting to /get_planning_scene service")
        self.scene_srv = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
        self.scene_srv.wait_for_service()

        rospy.loginfo("Connecting to clear octomap service...")
        self.clear_octomap_srv = rospy.ServiceProxy('/clear_octomap', Empty)
        self.clear_octomap_srv.wait_for_service()

        rospy.wait_for_service('/gripper_controller/grasp')
        self.gripper_controller = rospy.ServiceProxy('gripper_controller/grasp', Empty)

        rospy.loginfo("Grasp client initialised.")

    def start_grasp_obj(self, req):
        self.grasp_object()
        return {}

    def grasp_object(self):
        # Move TIAGo to pregrasp_position
        head_tilt = -0.8
        self.move_head(tilt=head_tilt)
        self.move_to_pregrasp(height=0.3)

        # Clear octomap
        self.clear_octomap_srv.call(EmptyRequest())

        raw_input("Plan grasps.")
        possible_grasps, possible_grasp_poses = self.gs.predict_grasps()

        rospy.loginfo("Received {} grasp proposals.".format(len(possible_grasps)))
        success = False
        i = 0
        self.check_for_table(head_tilt)
        while not success and i < len(possible_grasps):
            grasp_pose = possible_grasps[i][0]
            grasp_width = possible_grasps[i][2]

            rospy.loginfo("Attempting grasp #{}".format(i))
            i += 1
            if grasp_width > 0.08:
                rospy.loginfo("Grasp #{} width exceeds gripper dimensions with {}".format(i, grasp_width))
                continue

            success = self.execute_grasp(grasp_pose)

        # Open gripper
        self.open_gripper()

    def execute_grasp(self, grasp_pose):
        """ Execute grasp with TIAGo. We move the arm to a pre-grasp position, then move the wrist forward linearly
            to approach the grasp pose, close the gripper, and lift the object in positive z in base coordinates.

            Parameters
            -------
            grasp_pose (Pose): Pose of the grasp TCP in the 'base_footprint' coordinate frame.

        """
        # Publish the poses so they can be viewed in RVIZ
        self.gs.publish_poses([grasp_pose,
                               self.gs.calculate_pre_grasp_pose(grasp_pose,
                                                                dist=0.13,
                                                                end_effector=False)],
                              'base_footprint')

        # Pose is given in TCP, but moveit uses the coordinate frame of the wrist for path planning. Transforming
        # from the TCP to the wrist
        end_effector_pose = self.gs.transform_tcp_to_end_effector(grasp_pose)
        pre_grasp_pose = self.gs.calculate_pre_grasp_pose(end_effector_pose, dist=0.13)

        # Plan path with moveit
        self.group.clear_pose_targets()
        self.group.set_pose_target(pre_grasp_pose)
        plan = self.group.plan()
        # Check if a plan was found
        if plan.joint_trajectory.points:
            resp = raw_input("Can reach pose. continue? y/n: ")
            if resp != 'y':
                rospy.loginfo("User opted to not go to grasp pose. Check next grasp pose")
                return False
            # Move arm to pre-grasping pose
            self.group.execute(plan)
            self.group.clear_pose_targets()
        else:
            rospy.loginfo("Could not plan path to approach position. Abort.")
            return False

        # Calculate a linear path from the pre-grasping-pose to the actual grasping pose
        pre_grasp_pose_2 = self.gs.calculate_pre_grasp_pose(end_effector_pose, dist=0.05)

        (grasp_plan, fraction) = self.group.compute_cartesian_path([pre_grasp_pose_2, end_effector_pose],
                                                                   0.01,
                                                                   0.0,
                                                                   avoid_collisions=False)
        self.group.execute(grasp_plan)

        # Close gripper
        self.gripper_controller(EmptyRequest())

        # Move up (change z coordinate of pose, since it is in base-coordinates)
        post_grasp_pose = copy.deepcopy(end_effector_pose)
        post_grasp_pose.position.z += 0.15

        (grasp_plan, fraction) = self.group.compute_cartesian_path([post_grasp_pose],
                                                                   0.01,
                                                                   0.0,
                                                                   avoid_collisions=False)
        self.group.execute(grasp_plan)
        return True

    def move_to_pregrasp(self, height=0.15):
        """ Move to a pregrasp position with TIAGo's arm unfolded and pointing upwards

            Parameters
            -------
            height (float): Height of TIAGo's torso joint

        """
        rospy.loginfo("Resetting TIAGo's pose to initial pose")

        joint_goal = self.group.get_current_joint_values()
        joint_goal[0] = height
        joint_goal[1] = 0.08
        joint_goal[2] = -0.07
        joint_goal[3] = -3.0
        joint_goal[4] = 1.5
        joint_goal[5] = -1.57
        joint_goal[6] = 0.2
        joint_goal[7] = 0.0
        self.group.go(joint_goal, wait=True)
        self.group.stop()

    def open_gripper(self):
        """ Open TIAGo's gripper fully
        """
        jt = JointTrajectory()
        jt.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [0.04, 0.04]
        pt.time_from_start = rospy.Duration(0.5)
        jt.points.append(pt)
        self.gripper_cmd.publish(jt)
        rospy.sleep(1.0)

    def check_for_table(self, tilt):
        """ Let Tiago move his head.

            Parameters
            -------
            tilt (float): New rotational angle for the head tilting joint.
                            Negative is looking down, positive looking up.
            turn (float): New rotational angle for the head turning joint.

        """
        jt = JointTrajectory()
        jt.joint_names = ['head_1_joint', 'head_2_joint']
        for cnt, x in enumerate([1.0, 0.0, -1.0, 0.0]):
            jtp = JointTrajectoryPoint()
            jtp.positions = [x, tilt]
            jtp.time_from_start = rospy.Duration(float(cnt) + 1.0)
            jt.points.append(jtp)
        # Will not wait until command is finished!
        self.head_cmd.publish(jt)
        rospy.sleep(float(cnt))

    def move_head(self, tilt=0.0, turn=0.0):
        """ Let Tiago move his head.

            Parameters
            -------
            tilt (float): New rotational angle for the head tilting joint.
                            Negative is looking down, positive looking up.
            turn (float): New rotational angle for the head turning joint.

        """
        rospy.loginfo("GraspServer: Moving head")
        jt = JointTrajectory()
        jt.joint_names = ['head_1_joint', 'head_2_joint']
        jtp = JointTrajectoryPoint()
        jtp.positions = [turn, tilt]
        jtp.time_from_start = rospy.Duration(1.0)
        jt.points.append(jtp)
        # Will not wait until command is finished!
        self.head_cmd.publish(jt)


class GraspServer(object):
    def __init__(self):
        rospy.loginfo("Initializing GraspServer...")

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_l = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_b = tf2_ros.TransformBroadcaster()

        self.poses_pub = rospy.Publisher('/grasp_poses', PoseArray, latch=True, queue_size=3)

        rospy.loginfo("GraspServer initialized!")

    def publish_poses(self, possible_poses, frame_id):
        pa = PoseArray()
        pa.header.frame_id = frame_id
        pa.header.stamp = rospy.Time.now()
        for pose in possible_poses:
            pa.poses.append(pose)
        self.poses_pub.publish(pa)

    def calculate_pre_grasp_pose(self, orig_pose, dist, end_effector=True):
        pose = copy.deepcopy(orig_pose)
        position = np.expand_dims(np.array((pose.position.x,
                                            pose.position.y,
                                            pose.position.z)),
                                  axis=-1)
        orientation = self.quaternion_rotation_matrix(pose.orientation)

        transform = np.vstack((np.hstack((orientation, position)),
                               np.array((0, 0, 0, 1))))
        if end_effector:
            translation = np.expand_dims(np.array((-dist, 0, 0, 1)), -1)
        else:
            translation = np.expand_dims(np.array((0, 0, -dist, 1)), -1)

        pre_grasp_position = np.matmul(transform, translation).squeeze()

        pre_grasp_pose = pose
        pre_grasp_pose.position.x = pre_grasp_position[0]
        pre_grasp_pose.position.y = pre_grasp_position[1]
        pre_grasp_pose.position.z = pre_grasp_position[2]

        return pre_grasp_pose

    def transform_tcp_to_end_effector(self, orig_pose):
        pose = copy.deepcopy(orig_pose)
        # Construct transform tcp
        position = np.expand_dims(np.array((pose.position.x,
                                            pose.position.y,
                                            pose.position.z)),
                                  axis=-1)
        orientation = self.quaternion_rotation_matrix(pose.orientation)

        tf_tcp = np.vstack((np.hstack((orientation, position)),
                            np.array((0, 0, 0, 1))))
        # Construct translation and rotation transform (rotate around y axis)
        tf = np.eye(4)
        tf[2, 3] = -0.22  # translation in z
        tf[0, 0] = 0
        tf[0, 2] = -1
        tf[2, 0] = 1
        tf[2, 2] = 0

        # Apply transform
        tf_end_effector = np.matmul(tf_tcp, tf).squeeze()

        # Generate new pose message
        end_effector_pose = pose
        end_effector_pose.position.x = tf_end_effector[0, 3]
        end_effector_pose.position.y = tf_end_effector[1, 3]
        end_effector_pose.position.z = tf_end_effector[2, 3]

        new_orientation = self.matrix_to_quaternion(tf_end_effector[:3, :3])

        end_effector_pose.orientation.x = new_orientation[0]
        end_effector_pose.orientation.y = new_orientation[1]
        end_effector_pose.orientation.z = new_orientation[2]
        end_effector_pose.orientation.w = new_orientation[3]

        return end_effector_pose

    @staticmethod
    def matrix_to_quaternion(m=np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))):
        # q0 = qw
        t = np.matrix.trace(m)
        q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if t > 0:
            t = np.sqrt(t + 1)
            q[3] = 0.5 * t
            t = 0.5 / t
            q[0] = (m[2, 1] - m[1, 2]) * t
            q[1] = (m[0, 2] - m[2, 0]) * t
            q[2] = (m[1, 0] - m[0, 1]) * t

        else:
            i = 0
            if (m[1, 1] > m[0, 0]):
                i = 1
            if (m[2, 2] > m[i, i]):
                i = 2
            j = (i + 1) % 3
            k = (j + 1) % 3

            t = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1)
            q[i] = 0.5 * t
            t = 0.5 / t
            q[3] = (m[k, j] - m[j, k]) * t
            q[j] = (m[j, i] + m[i, j]) * t
            q[k] = (m[k, i] + m[i, k]) * t

        return q

    @staticmethod
    def quaternion_rotation_matrix(Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q.w
        q1 = Q.x
        q2 = Q.y
        q3 = Q.z

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])

        return rot_matrix

    def strip_leading_slash(self, s):
        return s[1:] if s.startswith("/") else s

    def predict_grasps(self):
        rospy.wait_for_service('gpnet_grasp_planner')
        plan_grasp = rospy.ServiceProxy('gpnet_grasp_planner', GPnetGraspPlanner)

        # read images
        depth_im = rospy.wait_for_message('/camera/depth_image', Image)
        camera_intr = rospy.wait_for_message('/camera/info', CameraInfo)
        response = plan_grasp(depth_im, camera_intr)
        rospy.loginfo("grasp_server: Received grasp proposals from GPnet. Transform coordinate frames.")
        planned_grasps = response.grasp_proposals

        frame_id = self.strip_leading_slash(planned_grasps.header.frame_id)
        rospy.loginfo("grasp_server: Transforming from frame: " + frame_id + " to 'base_footprint'")
        grasps = []
        poses = []
        for idx, grasp in enumerate(planned_grasps.grasps):
            ps = PoseStamped()
            ps.pose.position = grasp.pose.position
            ps.pose.orientation = grasp.pose.orientation

            ps.header.stamp = self.tfBuffer.get_latest_common_time("base_footprint", frame_id)
            ps.header.frame_id = frame_id

            transformed_ps = self.transform_to_base_footprint(ps, frame_id)
            grasps.append([transformed_ps.pose, grasp.quality, grasp.width])
            poses.append(transformed_ps.pose)
        return sorted(grasps, key=lambda l: - l[0].position.z), poses

    def transform_to_base_footprint(self, ps, frame_id):
        transform_ok = False
        while not transform_ok and not rospy.is_shutdown():
            try:
                transform = self.tfBuffer.lookup_transform("base_footprint",
                                                           ps.header.frame_id,
                                                           rospy.Time(0))
                transformed_ps = do_transform_pose(ps, transform)
                transform_ok = True
            except tf2_ros.ExtrapolationException as e:
                rospy.logwarn("Exception on transforming point... trying again \n(" + str(e) + ")")
                rospy.sleep(0.01)
                ps.header.stamp = self.tfBuffer.get_latest_common_time("base_footprint", frame_id)
        return transformed_ps


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('grasping_demo')
    grasping = GraspObject()
    rospy.spin()
