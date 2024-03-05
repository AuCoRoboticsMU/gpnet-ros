# Proposing Grasps for Mobile Manipulators

This is a ROS package for GP-net to be used on mobile manipulators. It uses a 
GP-net model to propose up to 5 grasps based on a depth image. A pre-trained model
for a robot with a PAL parallel jaw gripper is available at [zenodo](https://zenodo.org/record/7589237).
If you want to use GP-net for alternative grippers, you have to train a new model by
generating a new training dataset with [the dataset generation code](https://github.com/AuCoRoboticsMU/gpnet-data) and training
a new model with our code available on [github](https://github.com/AuCoRoboticsMU/GP-net).


-----
### Installation

This code has been tested with a ROS kinetic installation and python 2.7.
Besides the ROS installation, the package requirements are listed in `requirements.txt`.
We recommend an installation via docker or a virtual environment.

After cloning this package into the `src` folder in your catkin workspace, build the workspace
and `source catkin_ws/devel/setup.bash` in order to use the package.

---
### Use GP-net to grasp objects

GP-net can be used by launching the `grasp_planning_node.py` and use the planning service by
sending a depth image and the camera info to the node, e.g.:

```
plan_grasp = rospy.ServiceProxy('gpnet_grasp_planner', GPnetGraspPlanner)
depth_im = rospy.wait_for_message('/camera/depth_image', Image)
camera_intr = rospy.wait_for_message('/camera/info', CameraInfo)
grasp_response = plan_grasp(depth_im, camera_intr) 
```

An example usage script is given in `scripts/tiago_example.py`, which can be used with

`roslaunch gpnet tiago_example.launch`

Note that you will have to adjust the `model_dir` in the launch file to the path were you
your GP-net model is stored. A pretrained model is available at [zenodo](https://zenodo.org/record/7589237)

----------------
If you use this code, please cite

A. Konrad, J. McDonald and R. Villing, "GP-Net: Flexible Viewpoint Grasp Proposal," in 21st International Conference on Advanced Robotics (ICAR), (pp. 317-324), 2023.

--------
### Acknowledgements

This publication has emanated from research supported in part by Grants from Science Foundation Ireland under 
Grant numbers 18/CRT/6049 and 16/RI/3399.
The opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do 
not necessarily reflect the views of the Science Foundation Ireland.
