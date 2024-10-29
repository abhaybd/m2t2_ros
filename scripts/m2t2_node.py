#!/home/prior/miniconda3/envs/torch/bin/python

import numpy as np
import os
import argparse

import sys ; sys.path.append("/usr/lib/python3/dist-packages")
import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge
import tf

import omegaconf
from scipy.spatial.transform import Rotation as scipyR

from m2t2_ros.msg import GraspCandidate, GraspCandidates
from m2t2_model import M2T2Model

def get_args():
    parser = argparse.ArgumentParser(description='M2T2 ROS Node')
    parser.add_argument("camera")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-b", "--base-frame", default="world")
    return parser.parse_args()

def main():
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    args = get_args()
    rospy.init_node('m2t2_node')

    cam_params: CameraInfo = rospy.wait_for_message(f"/{args.camera}/rgb/camera_info", CameraInfo, timeout=5.0)
    cam_intrinsics = np.array(cam_params.K).reshape(3, 3)
    tf_listener = tf.TransformListener()
    bridge = CvBridge()

    cfg = omegaconf.OmegaConf.load(args.config)
    m2t2 = M2T2Model(cfg, cam_intrinsics)

    pub = rospy.Publisher("pred_grasps", GraspCandidates, queue_size=10)

    def get_cam_pose():
        trans, quat = tf_listener.lookupTransform(args.base_frame, f"{args.camera}/rgb_camera_link", rospy.Time())
        rotmat = scipyR.from_quat(quat).as_matrix()
        trf = np.eye(4)
        trf[:3,:3] = rotmat
        trf[:3,-1] = trans
        return trf

    def camera_callback(depth_msg: Image, rgb_msg: Image):
        depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        cam_pose = get_cam_pose()
        grasps, conf = m2t2.generate_grasps(cam_pose, depth, rgb)

        gcs = GraspCandidates()
        gcs.header.stamp = depth_msg.header.stamp
        gcs.header.frame_id = m2t2.base_frame
        for grasp, c in zip(grasps, conf):
            gc = GraspCandidate()
            pos, quat = gc.pose.position, gc.pose.orientation
            pos.x, pos.y, pos.z = grasp[:3, 3]
            quat.x, quat.y, quat.z, quat.w = scipyR.from_matrix(grasp[:3, :3]).as_quat()
            gc.confidence = c
            gcs.candidates.append(gc)
        pub.publish(gcs)

    depth_sub = message_filters.Subscriber(f"/{args.camera}/depth_to_rgb/image_raw", Image)
    rgb_sub = message_filters.Subscriber(f"/{args.camera}/rgb/image_raw", Image)
    ts = message_filters.ApproximateTimeSynchronizer([depth_sub, rgb_sub], 1, 0.05)
    ts.registerCallback(camera_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
