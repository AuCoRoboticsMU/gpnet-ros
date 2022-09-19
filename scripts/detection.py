import time

import numpy as np
import torch

from utils import Grasp, depth_encoding
from model import load_network
from skimage.feature import peak_local_max


class GPnet(object):
    def __init__(self, model_path, rviz=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)
        self.rviz = rviz

    def __call__(self, depth_im, K):
        tic = time.time()
        qual_pred, rot_pred, width_pred = predict(depth_im, self.net, self.device)
        grasps, scores, poses = select_grasps(qual_pred.copy(), rot_pred, width_pred, depth_im, K)
        toc = time.time() - tic
        grasps, scores = np.asarray(grasps), np.asarray(scores)

        return grasps, scores, toc

def predict(depth_image, net, device):
    x = depth_encoding(depth_image)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(x)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().detach().numpy()
    rot_vol = rot_vol.cpu().detach().numpy()
    width_vol = width_vol.cpu().detach().numpy()
    return qual_vol, rot_vol, width_vol

def select_grasps(pred_qual, pred_quat, pred_width, depth_im, K, n=5):
    indices = peak_local_max(pred_qual.squeeze(), min_distance=4, threshold_abs=0.1, num_peaks=n)
    grasps = []
    qualities = []
    poses = []

    for index in indices:
        quaternion = pred_quat.squeeze()[:, index[0], index[1]]
        quality = pred_qual.squeeze()[index[0], index[1]]
        width = pred_width.squeeze()[index[0], index[1]]

        contact = (index[1], index[0])
        grasp, T_camera_tcp = reconstruct_grasp_from_variables(depth_im, contact, quaternion, width, K)

        if grasp is None:
            continue

        grasps.append(grasp)
        qualities.append(quality)
        poses.append(T_camera_tcp)
    return grasps, qualities, poses

def reconstruct_grasp_from_variables(depth_im, contact, quaternion, width, K):
    # Deproject from depth image into image coordinates
    # Note that homogeneous coordinates have the image coordinate order (x, y), while accessing the depth image
    # works with numpy coordinates (row, column)
    homog = np.array((contact[0], contact[1], 1)).reshape((3, 1))
    if depth_im[contact[1], contact[0]] == 0.0:
        return None, None
    point = depth_im[contact[1], contact[0]] * np.linalg.inv(np.array(K).reshape(3, 3)).dot(homog)
    point = point.squeeze()

    # Transform the quaternion into a rotation matrix
    rot = quaternion_rotation_matrix(quaternion)

    # Move from contact to grasp centre by traversing 0.5*grasp width in grasp axis direction
    centre_point = point + width / 2 * rot.T[0, :]

    # Construct transform Camera --> gripper
    T_camera_tcp = np.r_[np.c_[rot, centre_point], [[0, 0, 0, 1]]]

    return Grasp(T_camera_tcp, width), T_camera_tcp

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
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

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
