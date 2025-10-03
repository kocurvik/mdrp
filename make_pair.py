import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform.rotation import Rotation


import rerun as rr
import rerun.blueprint as rrb

# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
# import open3d.ml.torch as ml3d

import poselib
from tqdm import tqdm

import cv2
import re

class PairMaker():
    def __init__(self):
        self.parse_args()

        print("Loading MoGe")
        self.mde_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda()
        print("Loading SuperPoint")
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        print("Loading LightGlue")
        self.matcher = LightGlue(features='superpoint').eval().cuda()

        self.ransac_dict = {'max_iterations': 1000, 'max_epipolar_error': self.args.threshold, 'progressive_sampling': False,
                            'min_iterations': 1000, 'lo_iterations': 25, 'max_reproj_error': self.args.reproj_threshold,
                            'solver_scale': True, 'solver_shift': False, 'use_reproj': False, 'use_p3p': True,
                            'optimize_hybrid': True, 'optimize_shift': False, 'use_ours': False,
                            'weight_sampson': self.args.weight_sampson}

        self.bundle_dict = {'loss_type':'TRUNCATED_CAUCHY'}
        # self.bundle_dict = {}

        name = os.path.basename(self.args.image_1_path).split('.')[0] + '-' + os.path.basename(self.args.image_2_path).split('.')[0]

        if self.args.cache_path is None:
            self.cache_dir = os.path.join('pair_cache', name)
        else:
            self.cache_dir = self.args.cache_path
        os.makedirs(self.cache_dir, exist_ok=True)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--threshold', type=float, default=2.0)
        parser.add_argument('-r', '--reproj_threshold', type=float, default=16.0)
        parser.add_argument('-w', '--weight_sampson', type=float, default=0.1)
        parser.add_argument('--subsample_rate', type=float, default=0.5)
        parser.add_argument('--use_cache', action='store_true', default=False)
        parser.add_argument('-v', '--verbose', action='store_true', default=False)
        parser.add_argument('--cache_path', type=str, default=None)
        parser.add_argument('image_1_path')
        parser.add_argument('image_2_path')

        self.args = parser.parse_args()


    def load_image(self, filename):
        return cv2.imread(filename)

    def preprocess_image(self, image):
        cv_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.tensor(cv_image1 / 255, dtype=torch.float32, device='cuda').permute(2, 0, 1)

    def infer_on_initial_anchor(self, image):
        self.since_last_anchor = 0
        self.anchor_R = np.eye(3)
        self.anchor_t = np.zeros(3)
        self.anchor_scale = 1.0
        self.anchor_mde_out = self.mde_model.infer(image)
        self.anchor_feats = self.extractor.extract(image)
        self.anchor_K, self.anchor_cam_dict = self.get_camera_data(image, self.anchor_mde_out)

        return self.get_pointcloud(np.linalg.inv(self.anchor_K), self.anchor_mde_out, image)

    def infer_images(self, image1, image2):
        # Infer Moge
        mde_out1 = self.mde_model.infer(image1)        
        mde_out2 = self.mde_model.infer(image2)
        
        # Extract SuperPoint features
        feats1_out = self.extractor.extract(image1)
        feats2_out = self.extractor.extract(image2)

        # Match the features using LightGlue
        matches01 = self.matcher({'image0': feats1_out, 'image1': feats2_out})
        feats1, feats2, matches01 = [rbd(x) for x in [feats1_out, feats2_out, matches01]]

        # conver the matches to numpy
        matches = matches01['matches']
        points1 = feats1['keypoints'][matches[..., 0]].cpu().numpy()
        points2 = feats2['keypoints'][matches[..., 1]].cpu().numpy()

        # extract depths for each keypoint
        depth_map1 = mde_out1["depth"].cpu().numpy()
        depth_map2 = mde_out2["depth"].cpu().numpy()
        depths1 = depth_map1[points1[:, 1].astype(int), points1[:, 0].astype(int)]
        depths2 = depth_map2[points2[:, 1].astype(int), points2[:, 0].astype(int)]

        l = ~np.logical_and(np.isinf(depths1), np.isinf(depths2))

        K2, cam_dict_2 = self.get_camera_data(image2, mde_out2)        
        K1, cam_dict_1 = self.get_camera_data(image1, mde_out1)

        d = np.column_stack([depths1, depths2])


        pose, info = poselib.estimate_relative_pose_w_mono_depth(points1[l], points2[l], d[l],
                                                                 cam_dict_1, cam_dict_2, self.ransac_dict,
                                                                 self.bundle_dict)

        if self.args.verbose:
            print(info['inlier_ratio'])
            print("R: ")
            print(pose.R)
            print("t: ", pose.t)
            print("scale: ", pose.scale)

        self.pcd_1, self.colors_1 = self.get_pointcloud(np.linalg.inv(K1), mde_out1, image1)
        self.pose = pose
        # self.pcd_1 = (pose.R @ self.pcd_1.T).T + pose.t
        # self.pcd_1 /= pose.scale

        self.pcd_2, self.colors_2 = self.get_pointcloud(np.linalg.inv(K2), mde_out2, image2)

        self.K1 = K1
        self.K2 = K2

    def postprocess(self):
        print("Processing video")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        img_ours = cv2.imread(os.path.join(self.cache_dir, f'frame_ours_{0:05d}.png'))
        new_dims = [img_ours.shape[1], img_ours.shape[0]]
        self.writer_ours = cv2.VideoWriter(os.path.join(self.cache_dir, 'ours.mp4'), fourcc, 25.0, new_dims)


        for i in range(360//3):
            img_pcd = cv2.imread(os.path.join(self.cache_dir, f'frame_ours_{i:05d}.png'))
            self.writer_ours.write(img_pcd)

        self.writer_ours.release()
        print("Video processed")

    def run_visualizer(self, key):
        ctr = self.vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(os.path.join(self.cache_dir, 'camera_position.json'), param)

        # y_axis = np.copy(param.extrinsic[:3, 1])
        y_axis = np.copy(param.extrinsic[1, :3])
        y_axis /= np.linalg.norm(y_axis)
        y_axis *= 2 * np.pi / 360 * 3
        print("yaxis: ", y_axis)
        R = Rotation.from_rotvec(y_axis).as_matrix()
        # R = self.geometry_1.get_rotation_matrix_from_xyz((0, 3 / 360 * 2 * np.pi, 0))

        self.vis.get_view_control().reset_camera_local_rotate()

        for i in tqdm(range(360//3)):
            self.vis.capture_screen_image(os.path.join(self.cache_dir, f'frame_ours_{i:05d}.png'), False)
            self.geometry_1.rotate(R, self.center)
            self.geometry_2.rotate(R, self.center)
            self.vis_camera_1.rotate(R, self.center)
            self.vis_camera_2.rotate(R, self.center)

            self.vis.update_geometry(self.geometry_1)
            self.vis.update_geometry(self.geometry_2)
            self.vis.update_geometry(self.vis_camera_1)
            self.vis.update_geometry(self.vis_camera_2)

            self.vis.poll_events()
            self.vis.update_renderer()

        self.postprocess()

    def process_pair(self):
        image_1 = cv2.imread(self.args.image_1_path)
        image_2 = cv2.imread(self.args.image_2_path)
        
        self.infer_images(self.preprocess_image(image_1), self.preprocess_image(image_2))

        self.geometry_1 = o3d.geometry.PointCloud()
        l = np.random.rand(len(self.pcd_1)) < self.args.subsample_rate
        l = np.logical_and(l, ~np.any(np.isinf(self.pcd_1), axis=1))
        self.geometry_1.points = o3d.utility.Vector3dVector(self.pcd_1[l].astype(np.float64))
        self.geometry_1.colors = o3d.utility.Vector3dVector(self.colors_1[l])

        self.geometry_1.rotate(self.pose.R, np.zeros(3))
        self.geometry_1.translate(self.pose.t)
        self.geometry_1.scale(1 / self.pose.scale, np.zeros(3))
        
        self.geometry_2 = o3d.geometry.PointCloud()
        l = np.random.rand(len(self.pcd_2)) < self.args.subsample_rate
        l = np.logical_and(l, ~np.any(np.isinf(self.pcd_2), axis=1))
        self.geometry_2.points = o3d.utility.Vector3dVector(self.pcd_2[l].astype(np.float64))
        self.geometry_2.colors = o3d.utility.Vector3dVector(self.colors_2[l])

        Rt = np.eye(4)
        self.vis_camera_2 = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_2.shape[1],
                                                                            view_height_px=image_2.shape[0],
                                                                            intrinsic=self.K2,
                                                                            extrinsic=Rt)
        self.vis_camera_2.scale(0.05 * np.median(np.linalg.norm(self.pcd_2, axis=1)), np.zeros(3))

        self.vis_camera_1 = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_1.shape[1],
                                                                             view_height_px=image_1.shape[0],
                                                                             intrinsic=self.K1,
                                                                             extrinsic=Rt)
        self.vis_camera_1.scale(0.05 * np.median(np.linalg.norm(self.pcd_1, axis=1)), np.zeros(3))
        self.vis_camera_1.rotate(self.pose.R)
        self.vis_camera_1.translate(self.pose.t / self.pose.scale)

        print("First point cloud loaded!")

        self.width = 1280
        self.height = 720

        self.center = (np.nanmedian(self.pcd_1, axis=0) + np.nanmedian(self.pcd_2, axis=0)) / 2

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord(' '), self.run_visualizer)
        self.vis.create_window(width=self.width, height=self.height)
        self.vis.get_render_option().background_color = [255, 255, 255]
        self.vis.get_render_option().point_size = 1
        self.vis.get_render_option().line_width = 10

        self.vis.add_geometry(self.geometry_1)
        self.vis.add_geometry(self.geometry_2)
        self.vis.add_geometry(self.vis_camera_1)
        self.vis.add_geometry(self.vis_camera_2)
        self.vis.poll_events()
        self.vis.update_renderer()

        self.vis.get_view_control().set_lookat(self.center)
        self.vis.run()


    def get_camera_data(self, image, mde_out):
        h, w = image.size(-2), image.size(-1)
        instrinsics = mde_out['intrinsics'].cpu().numpy()

        # MoGe also provides estimated focal length
        f, px, py = max(w, h) * instrinsics[0, 0], w * instrinsics[0, 2], h * instrinsics[1, 2]

        ## Since focals are known we can use the calibrated solver.
        poselib_camera_dict = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height': -1, 'params': [f, px, py]}
        K = np.array([[f, 0, px], [0, f, py],[0, 0, 1]])

        return K, poselib_camera_dict


    def get_pointcloud(self, K_inv, mde_out, image):
        depth_map = mde_out["depth"].cpu().numpy()
        color_map = image.cpu().numpy().transpose(1, 2, 0).reshape(-1, 3)
        H, W = depth_map.shape
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)

        z = depth_map.flatten()
        uu = uu.flatten()
        vv = vv.flatten()
        proj = np.vstack([uu, vv, np.ones_like(uu)])

        flattened_colors = color_map.reshape([-1, 3])

        pcd = z[:, np.newaxis] * (K_inv @ proj).T

        l = ~np.any(np.logical_or(np.isinf(pcd), np.isnan(pcd)), axis=1)

        return pcd[l], flattened_colors[l]


if __name__ == '__main__':
    v = PairMaker()
    v.process_pair()