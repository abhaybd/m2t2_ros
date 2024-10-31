import numpy as np
from collections import defaultdict

import torch

from m2t2.m2t2 import M2T2
from m2t2.meshcat_utils import create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
from m2t2.plot_utils import get_set_colors
from m2t2.dataset_utils import normalize_rgb, sample_points


class M2T2Model(object):
    def __init__(self, cfg, cam_intrinsics: np.ndarray):
        self.cfg = cfg
        self.m2t2 = M2T2.from_config(cfg.m2t2)
        ckpt = torch.load(cfg.eval.checkpoint)
        self.m2t2.load_state_dict(ckpt['model'])
        self.m2t2 = self.m2t2.cuda().eval()

        self.cam_intrinsics = cam_intrinsics

    def generate_grasps(self, cam_pose: np.ndarray, depth_img: np.ndarray, rgb_img: np.ndarray, visualize=False):
        """Returns predicted grasps and confidence scores for the given depth and rgb images"""
        assert depth_img.shape[:2] == rgb_img.shape[:2], "depth and rgb images must be the same resolution"

        # normalize rgb
        rgb = normalize_rgb(rgb_img).permute(1, 2, 0).numpy()

        # Convert depth image to point cloud
        height, width = depth_img.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
        mask = depth_img > 0
        uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
        uvd *= np.expand_dims(depth_img / 1000.0, axis=-1)
        uvd = uvd[mask]
        xyz = np.linalg.solve(self.cam_intrinsics, uvd.T).T

        # Transform point cloud to world frame
        xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]
        inputs = np.concatenate([xyz - np.mean(xyz, axis=0), rgb[mask]], axis=1)

        data = {
            "cam_pose": torch.from_numpy(cam_pose).float(),
            "object_points": torch.rand(100, 3),
            "object_inputs": torch.rand(100, 6),
            "object_center": torch.zeros(3),
            "ee_pose": torch.eye(4),
            "bottom_center": torch.zeros(3),
            "task_is_pick": torch.tensor(True),
            "task_is_place": torch.tensor(False)
        }
        data = {k: v.unsqueeze(0).cuda() for k, v in data.items()}
        
        outputs = defaultdict(list)
        for _ in range(self.cfg.eval.num_runs):
            pt_idx = sample_points(xyz, self.cfg.data.num_points).numpy()
            data["inputs"] = torch.from_numpy(inputs[pt_idx]).float().unsqueeze(0).cuda()
            data["points"] = torch.from_numpy(xyz[pt_idx]).float().unsqueeze(0).cuda()
            with torch.no_grad():
                out = self.m2t2.infer(data, self.cfg.eval)
                for k in ["grasps", "grasp_confidence", "grasp_contacts"]:
                    outputs[k].extend(out[k][0])

        pred_grasps = torch.cat(outputs["grasps"], dim=0).cpu().numpy()
        pred_conf = torch.cat(outputs["grasp_confidence"], dim=0).cpu().numpy()

        if visualize:
            vis = create_visualizer()
            make_frame(vis, 'camera', T=cam_pose)
            visualize_pointcloud(vis, "scene", xyz, rgb_img[mask], size=0.005)
            for i, (grasps, conf, contacts, color) in enumerate(zip(
                outputs['grasps'],
                outputs['grasp_confidence'],
                outputs['grasp_contacts'],
                get_set_colors()
            )):
                print(f"object_{i:02d} has {grasps.shape[0]} grasps")
                conf = conf.cpu().numpy()
                conf_colors = (np.stack([
                    1 - conf, conf, np.zeros_like(conf)
                ], axis=1) * 255).astype('uint8')
                visualize_pointcloud(
                    vis, f"object_{i:02d}/contacts",
                    contacts.cpu().numpy(), conf_colors, size=0.01
                )
                grasps = grasps.cpu().numpy()
                for j, grasp in enumerate(grasps):
                    visualize_grasp(
                        vis, f"object_{i:02d}/grasps/{j:03d}",
                        grasp, color, linewidth=0.2
                    )

        return pred_grasps, pred_conf
