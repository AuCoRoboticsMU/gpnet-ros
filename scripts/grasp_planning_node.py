#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np

from detection import GPnet

from gpnet.srv import GPnetGraspPlanner
from gpnet.msg import GPnetGrasp, GraspProposals

class GraspPlanner(object):
    def __init__(self, cv_bridge, grasp_proposal_network):
        """
        Parameters
        cv_bridge: :obj:`CvBridge`
            ROS `CvBridge`.
        grasp_proposal_network: :obj:`GraspingPolicy`
            Grasping policy to use.
        """
        self.cv_bridge = cv_bridge
        self.gpnet = grasp_proposal_network

    def read_images(self, req):
        """Reads images from a ROS service request.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for grasp planner service.
        """
        # Get the raw depth image as ROS `Image` objects.
        raw_depth = req.depth_image

        # Get the raw camera info as ROS `CameraInfo`.
        raw_camera_info = req.camera_info

        # Unpacking the ROS depth image using ROS `CvBridge`
        try:
            depth_im = self.cv_bridge.imgmsg_to_cv2(
                raw_depth, desired_encoding="passthrough")
        except:
            rospy.logerr("Could not convert ROS depth image.")
        depth_image = depth_im.copy()
        depth_image[np.isnan(depth_image)] = 0.0

        return depth_image, raw_camera_info.K, raw_camera_info.header.frame_id

    def plan_grasp(self, req):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """
        depth_im, K, frame = self.read_images(req)
        # rospy.loginfo(K)
        rospy.loginfo("Planning Grasp")
        return self._find_grasps(depth_im, self.gpnet, K, frame)

    @staticmethod
    def matrix_to_quaternion(m):
        # q0 = qw
        t = np.matrix.trace(m)
        q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if (t > 0):
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

    def _find_grasps(self, depth_im, net, K, grasp_frame):
        """Executes a grasping policy on an `RgbdImageState`.

        Parameters
        ----------
        depth_im: :obj:`RgbdImageState`
            `RgbdImageState` from BerkeleyAutomation/perception to encapsulate
            depth and color image along with camera intrinsics.
        net: :obj:`GraspingPolicy`
            Grasping policy to use.
        grasp_frame: :obj:`str`
            Frame of reference to publish pose in.
        """
        # Execute the policy's action.
        rospy.loginfo("Grasp Planning Node: Predict grasps")
        grasps, scores, toc = net(depth_im, K)

        # Create `GraspProposals` return msg and populate it.
        grasp_proposals = GraspProposals()
        grasp_proposals.header.frame_id = grasp_frame
        grasp_proposals.header.stamp = rospy.Time.now()
        for grasp, quality in zip(grasps, scores):
            gpnet_grasp = GPnetGrasp()
            gpnet_grasp.quality = quality
            pose = grasp.pose.squeeze()
            gpnet_grasp.pose.position.x = pose[0, 3]
            gpnet_grasp.pose.position.y = pose[1, 3]
            gpnet_grasp.pose.position.z = pose[2, 3]
            q = self.matrix_to_quaternion(pose[:3, :3])
            gpnet_grasp.pose.orientation.x = q[0]
            gpnet_grasp.pose.orientation.y = q[1]
            gpnet_grasp.pose.orientation.z = q[2]
            gpnet_grasp.pose.orientation.w = q[3]
            gpnet_grasp.width = grasp.width
            grasp_proposals.grasps.append(gpnet_grasp)
        return grasp_proposals


if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.init_node("Grasp_Proposal_Server")

    # Initialize `CvBridge`.
    cv_bridge = CvBridge()

    # Get configs.
    model_name = rospy.get_param("~model_name")
    model_dir = rospy.get_param("~model_dir")
    if model_dir.lower() == "default":
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "../models")
    model_dir = os.path.join(model_dir, model_name)

    # Create a grasping policy.
    rospy.loginfo("Load Grasp Proposal Network")
    grasp_proposal_network = GPnet(model_dir)

    # Create a grasp planner.
    grasp_planner = GraspPlanner(cv_bridge, grasp_proposal_network)

    # Initialize the ROS service.
    grasp_planning_service = rospy.Service("gpnet_grasp_planner", GPnetGraspPlanner,
                                           grasp_planner.plan_grasp)
    rospy.loginfo("Grasp Planner Initialized")

    # Spin forever.
    rospy.spin()
