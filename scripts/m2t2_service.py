#!/home/prior/miniconda3/envs/torch/bin/python

import numpy as np
import os
import argparse

import sys ; sys.path.append("/usr/lib/python3/dist-packages")
import rospy
from cv_bridge import CvBridge

import omegaconf
from scipy.spatial.transform import Rotation as scipyR

from m2t2_ros.msg import GraspCandidate
from m2t2_ros.srv import GraspPrediction, GraspPredictionResponse, GraspPredictionRequest
from m2t2_model import M2T2Model


def get_args():
    parser = argparse.ArgumentParser(description='M2T2 ROS Service')
    parser.add_argument("config", default="config.yaml")
    return parser.parse_args()

def main():
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    args = get_args()
    rospy.init_node('m2t2_service')

    cfg = omegaconf.OmegaConf.load(args.config)
    m2t2 = M2T2Model(cfg, None)
    bridge = CvBridge()

    def handle_grasp_prediction(req: GraspPredictionRequest):
        cam_pose = np.eye(4)
        pos, quat = req.cam_pose.position, req.cam_pose.orientation
        cam_pose[:3,:3] = scipyR.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        cam_pose[:3,-1] = [pos.x, pos.y, pos.z]
        
        cam_intrinsics = np.array([req.cam_intrinsics]).reshape(3, 3)
        depth = bridge.imgmsg_to_cv2(req.depth, desired_encoding="passthrough")
        rgb = bridge.imgmsg_to_cv2(req.rgb, desired_encoding="rgb8")
        m2t2.cam_intrinsics = cam_intrinsics
        grasps, conf = m2t2.generate_grasps(cam_pose, depth, rgb)

        res = GraspPredictionResponse()
        res.candidates.header.stamp = req.depth.header.stamp
        res.candidates.header.frame_id = req.depth.header.frame_id
        for grasp, c in zip(grasps, conf):
            gc = GraspCandidate()
            pos, quat = gc.pose.position, gc.pose.orientation
            pos.x, pos.y, pos.z = grasp[:3, 3]
            quat.x, quat.y, quat.z, quat.w = scipyR.from_matrix(grasp[:3, :3]).as_quat()
            gc.confidence = c
            res.candidates.candidates.append(gc)
        return res

    s = rospy.Service('grasp_prediction', GraspPrediction, handle_grasp_prediction)
    rospy.spin()

if __name__ == "__main__":
    main()
