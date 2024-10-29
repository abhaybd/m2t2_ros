# M2T2_ROS

This repository provides an easy-to-use ROS wrapper around the [M2T2](https://m2-t2.github.io/) grasp predictor.

## Installation

1. Clone this project into your catkin workspace.
2. Ensure M2T2 is installed by following [these instructions](https://github.com/NVlabs/M2T2?tab=readme-ov-file#installation).
3. Download the model weights from [HuggingFace](https://huggingface.co/wentao-yuan/m2t2) and place it in the project root, named `m2t2.pth`.
4. Build your workspace with catkin.

Note: you may use a virtual environment if you wish, the provided ROS node and service automatically loads the system python libraries to use ROS.

## Usage

There are two ways to run this, as a ROS node or a standalone service.

### As a ROS node

When running this way, a ROS node will subscribe to depth and RGB images, run inference realtime, and publish predicted grasps to a `pred_grasps` topic.
This may be useful if you want a stream of the latest grasp predictions.

To do so, simply run:
```bash
rosrun m2t2_ros m2t2_node.py <camera_name>
```

`<camera_name>` is the name of the camera namespace, which contains the `rgb/image_raw`, `depth_to_rgb/image_raw`, and `rgb/camera_info` topics.
Additionally, the frame `<camera_name>/rgb_camera_link` should be published. The world frame is assumed to be `world`, but can be modified with the `-b` flag.


### As a ROS service

When running this way, you can call the ROS service with the depth and RGB images (and camera properties) and receive back all the generated grasp predictions.
This may be useful if you want to run on non-realtime data, or if you want explicit syncing between images and predictions.

To do so, simply run:
```bash
rosrun m2t2_ros m2t2_service.py
```

This provides the `grasp_prediction` service, which is defined [here](srv/GraspPrediction.srv).
When calling the service, you will provide the depth and RGB images, as well as the camera intrinsics matrix and the camera pose relative to the world frame.
The intrinsics should be specified as a 9-dimensional array, representing the flattened matrix in row-major order.
