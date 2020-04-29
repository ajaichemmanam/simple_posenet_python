# simple_posenet_python
A simple and minimal posenet inference in python

# Getting Started
1. Install requirements
2. Download tfjs models for posenet.
3. Set path to models and image for inference in .py files
4. python3 eval_singleposemodel.py (Image with single person) or python3 eval_multiposemodel.py (Image with single/multiple person)

# Observed Results

SINGLE POSE OUTPUT
![SinglePose BBOX](https://raw.githubusercontent.com/ajaichemmanam/simple_posenet_python/master/assets/singlepose_bbox.png)
![SinglePose Keypoints](https://raw.githubusercontent.com/ajaichemmanam/simple_posenet_python/master/assets/singlepose_keypoints.png)
![SinglePose Connected Keypoints](https://raw.githubusercontent.com/ajaichemmanam/simple_posenet_python/master/assets/singlepose_connectedkeypoints.png)

MULTIPOSE OUTPUT
![MultiPose BBOX](https://raw.githubusercontent.com/ajaichemmanam/simple_posenet_python/master/assets/multipose_bbox.png)
![MultiPose Keypoints](https://raw.githubusercontent.com/ajaichemmanam/simple_posenet_python/master/assets/multipose_keypoints.png)
![MultiPose Connected Keypoints](https://raw.githubusercontent.com/ajaichemmanam/simple_posenet_python/master/assets/multipose_connectedkeypoints.png)

Multipose  gives better results than singlepose even for single person. Due to graph tree based refinement.

# ACKNOWLEDGEMENT
Thanks to https://github.com/patlevin for support functions