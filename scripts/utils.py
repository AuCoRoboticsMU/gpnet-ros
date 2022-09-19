import numpy as np
import matplotlib.pyplot as plt


def depth_encoding(image):
    norm = plt.Normalize(vmin=0.4, vmax=1.4)
    colored_image = plt.cm.jet(norm(image))[:, :, :-1]
    return np.array((colored_image[:, :, 0], colored_image[:, :, 1], colored_image[:, :, 2]), dtype=np.float32)

class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width
