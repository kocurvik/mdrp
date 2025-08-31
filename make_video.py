import time

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import open3d as o3d
# import open3d.ml.torch as ml3d

import poselib
from tqdm import tqdm


class VideoMaker():
    def __init__(self):
        self.mde_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda()
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features='superpoint').eval().cuda()

        self.ransac_dict = {'max_iterations': 1000, 'max_epipolar_error': 2.0, 'progressive_sampling': False,
                            'min_iterations': 10000, 'lo_iterations': 25, 'max_reproj_error': 16.0,
                            'solver_scale': True, 'solver_shift': True,
                            'optimize_hybrid': True, 'optimize_shift': False, 'use_ours': True, 'weight_sampson': 2.0}
        
        self.anchor_K = None
        self.anchor_cam_dict = None
        self.anchor_mde_out = None
        self.anchor_feats = None


    def load_image(self, filename):
        return cv2.imread(filename)

    def preprocess_image(self, image):
        cv_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.tensor(cv_image1 / 255, dtype=torch.float32, device='cuda').permute(2, 0, 1)

    def infer_on_anchor(self, image):
        self.anchor_mde_out = self.mde_model.infer(image)
        self.anchor_feats = self.extractor.extract(image)
        self.anchor_K, self.anchor_cam_dict = self.get_camera_data(image, self.anchor_mde_out)

        return self.get_pointcloud(np.linalg.inv(self.anchor_K), self.anchor_mde_out, image)

    def infer_images(self, image1, image2, feats2=None):
        # Infer Moge
        mde_out1 = self.mde_model.infer(image1)
        if self.anchor_mde_out is None:
            mde_out2 = self.mde_model.infer(image2)
        else:
            mde_out2 = self.anchor_mde_out

        # Extract SuperPoint features
        feats1 = self.extractor.extract(image1)
        if self.anchor_feats is None:
            feats2 = self.extractor.extract(image2)
        else:
            feats2 = self.anchor_feats

        # Match the features using LightGlue
        matches01 = self.matcher({'image0': feats1, 'image1': feats2})
        feats1, feats2, matches01 = [rbd(x) for x in [feats1, feats2, matches01]]

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

        if self.anchor_K is None:
            K2, cam_dict_2 = self.get_camera_data(image2, mde_out2)
        else:
            K2, cam_dict_2 = self.anchor_K, self.anchor_cam_dict

        K1, cam_dict_1 = self.get_camera_data(image1, mde_out1)

        d = np.column_stack([depths1, depths2])

        pose, info = poselib.estimate_relative_pose_w_mono_depth(points1[l], points2[l], d[l],
                                                                 cam_dict_1, cam_dict_2, self.ransac_dict, {})

        print(info['inlier_ratio'])
        print("R: ")
        print(pose.R)
        print("t: ", pose.t)
        print("scale: ", pose.scale)

        pcd_1, colors_1 = self.get_pointcloud(np.linalg.inv(K1), mde_out1, image1)
        pcd_1 /= pose.scale
        pcd_1 = (pose.R @ pcd_1.T).T + pose.t

        return pcd_1, colors_1

    def step_visualizer(self, key):

        if not self.updated:
            print("wating for previous updated")
            return
        print("Updating")

        self.updated = False
        ret, frame = self.cap.read()

        if not ret:
            self.vis.close()

        self.i += 1

        xyz, colors = self.infer_images(self.preprocess_image(frame), None)

        # pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(xyz[::50])
        self.pcd.colors = o3d.utility.Vector3dVector(colors[::50])
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.capture_screen_image(f'video/frame_{self.i:05d}.png', False)
        print("Updated")
        self.updated = True

    def postprocess(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter('output.avi', fourcc, 25.0, (self.new_width, 2 * self.new_height))

        for i in range(self.video_length):
            img_pcd = cv2.imread(f'video/frame_{i:05d}.png')
            ret, frame = self.cap.read()
            if ret is None or img_pcd is None:
                break

            frame_resize = cv2.resize(frame, (self.new_width, self.new_height))

            output_frame = np.concatenate([frame_resize, img_pcd], axis=0)
            self.writer.write(output_frame)

        self.cap.release()
        self.writer.release()

    def run_visualizer(self, key):
        # capture first frame
        self.vis.capture_screen_image(f'video/frame_{self.i:05d}.png', False)


        if not self.updated:
            print("wating for previous updated")
            return
        print("Running forever")


        self.updated = False
        while True:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                print("Done")
                self.vis.close()
                self.postprocess()
                return

            self.i += 1

            xyz, colors = self.infer_images(self.preprocess_image(frame), None)

            # pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(np.copy(xyz[::20]))
            self.pcd.colors = o3d.utility.Vector3dVector(np.copy(colors[::20]))

            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
            self.vis.capture_screen_image(f'video/frame_{self.i:05d}.png', False)
            print(f"Updated {self.i} / {self.video_length}")

    def process_video(self, video_path):
        self.i = 0
        self.cap = cv2.VideoCapture(video_path)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_width = self.video_width // 2
        self.new_height = self.video_height // 2

        # self.postprocess()
        # return

        ret, frame = self.cap.read()

        xyz, colors = self.infer_on_anchor(self.preprocess_image(frame))

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.copy(xyz[::20]))
        self.pcd.colors = o3d.utility.Vector3dVector(np.copy(colors[::20]))

        # pcd_init = o3d.geometry.PointCloud()
        # pcd_init.points = o3d.utility.Vector3dVector(xyz)
        # pcd_init.colors = o3d.utility.Vector3dVector(colors)

        print("First point cloud loaded!")

        # self.pcds = [pcd]
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        # self.vis = ml3d.vis.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord('n'), self.step_visualizer)
        self.vis.register_key_callback(ord(' '), self.run_visualizer)
        self.vis.create_window(width=self.new_width, height=self.new_height)
        self.vis.get_render_option().background_color = [0, 0, 0]

        # self.vis.add_geometry(pcd_init)
        self.vis.add_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.updated = True
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

        return z[:, np.newaxis] * (K_inv @ proj).T, flattened_colors


if __name__ == '__main__':
    v = VideoMaker()
    v.process_video("trains_iphone.MOV")