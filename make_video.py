import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch
import open3d as o3d
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


class VideoCaptureProxy:
    """
    A class that acts as a proxy for cv2.VideoCapture, allowing it to handle
    both video files and directories of sorted image files.

    It provides the same core methods as cv2.VideoCapture:
    - read()
    - isOpened()
    - release()
    - get()
    """

    def __init__(self, source):
        """
        Initializes the VideoCaptureProxy with a video file or a directory path.

        Args:
            source (str): The path to a video file or a directory containing images.
        """
        self.is_video_file = os.path.isfile(source)
        self.source = source
        self.frame_count = 0
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame_index = -1
        self.image_files = []
        self.is_open = False

        # Define a list of common image extensions
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        if self.is_video_file:
            # Handle a video file using the standard cv2.VideoCapture
            self.cap = cv2.VideoCapture(source)
            self.is_open = self.cap.isOpened()
        elif os.path.isdir(source):
            # Handle a directory of images
            self.cap = None  # No underlying cv2.VideoCapture object

            # Use natural sorting to handle filenames like frame_1.png, frame_10.png
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

            # Filter and sort image files
            files = [f for f in os.listdir(source) if f.lower().endswith(self.image_extensions)]
            self.image_files = sorted(files, key=natural_sort_key)
            self.frame_count = len(self.image_files)

            if self.frame_count > 0:
                # Read the first image to get dimensions
                first_frame_path = os.path.join(source, self.image_files[0])
                first_frame = cv2.imread(first_frame_path)
                if first_frame is not None:
                    self.frame_height, self.frame_width = first_frame.shape[:2]
                    self.is_open = True
                    self.current_frame_index = 0
                else:
                    print(f"Error: Could not read first image from directory {source}.")
            else:
                print(f"Error: No image files found in directory {source}.")
        else:
            print(f"Error: Source '{source}' is not a valid video file or directory.")

    def isOpened(self):
        """
        Returns True if the video source is open and ready to be read, False otherwise.
        """
        if self.is_video_file:
            return self.cap.isOpened()
        else:
            return self.is_open

    def read(self):
        """
        Reads the next frame from the video source.

        Returns:
            tuple: A tuple containing a boolean (True if a frame was read successfully, False otherwise)
                   and the frame (a numpy array) or None.
        """
        if not self.isOpened():
            return False, None

        if self.is_video_file:
            return self.cap.read()
        else:
            if self.current_frame_index < self.frame_count:
                frame_path = os.path.join(self.source, self.image_files[self.current_frame_index])
                frame = cv2.imread(frame_path)
                self.current_frame_index += 1
                return True, frame
            else:
                return False, None

    def get(self, propId):
        """
        Returns the specified property of the video source.

        Args:
            propId (int): The property identifier.

        Returns:
            float: The value of the property.
        """
        if self.is_video_file:
            return self.cap.get(propId)
        else:
            if propId == cv2.CAP_PROP_FRAME_COUNT:
                return self.frame_count
            elif propId == cv2.CAP_PROP_FRAME_WIDTH:
                return self.frame_width
            elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
                return self.frame_height
            elif propId == cv2.CAP_PROP_FPS:
                # FPS is not inherent for an image sequence. Return a default value.
                return 30.0
            elif propId == cv2.CAP_PROP_POS_FRAMES:
                return self.current_frame_index
            else:
                # For any other unsupported property, return a default value
                return 0.0

    def set(self, propId, value):
        """
        Sets a property of the video source. Currently only supports
        cv2.CAP_PROP_POS_FRAMES for image directories.

        Args:
            propId (int): The property identifier.
            value (float): The value to set the property to.

        Returns:
            bool: True if the property was set successfully, False otherwise.
        """
        if self.is_video_file:
            return self.cap.set(propId, value)
        else:
            if propId == cv2.CAP_PROP_POS_FRAMES:
                # Ensure the value is an integer and within the valid range
                frame_idx = int(value)
                if 0 <= frame_idx < self.frame_count:
                    self.current_frame_index = frame_idx
                    return True
                else:
                    print(f"Warning: Cannot set frame position to {frame_idx}. It is out of bounds [0, {self.frame_count-1}].")
                    return False
            else:
                print(f"Warning: Property {propId} is not supported for image directories.")
                return False


    def release(self):
        """
        Releases the video source.
        """
        if self.is_video_file and self.cap:
            self.cap.release()
        self.is_open = False

class VideoMaker():
    def __init__(self):
        self.parse_args()

        print("Loading MoGe")
        self.mde_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda()
        print("Loading SuperPoint")
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        print("Loading LightGlue")
        self.matcher = LightGlue(features='superpoint').eval().cuda()

        self.ransac_dict = {'max_iterations': 1000, 'max_epipolar_error': self.args.threshold, 'progressive_sampling': False,
                            'min_iterations': 10000, 'lo_iterations': 25, 'max_reproj_error': self.args.reproj_threshold,
                            'solver_scale': True, 'solver_shift': True, 'use_reproj': True,
                            'optimize_hybrid': True, 'optimize_shift': False, 'use_ours': True,
                            'weight_sampson': self.args.weight_sampson}

        if self.args.cache_path is None:
            self.cache_dir = os.path.join('video_cache', os.path.basename(self.args.video_path).split('.')[0])
        else:
            self.cache_dir = self.args.cache_path
        os.makedirs(self.cache_dir, exist_ok=True)

        if self.args.output_path is None:
            self.output_path = os.path.join(self.cache_dir, 'output.avi')
        else:
            self.output_path = self.args.output_path

        self.anchor_K = None
        self.anchor_cam_dict = None
        self.anchor_mde_out = None
        self.anchor_feats = None

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--threshold', type=float, default=2.0)
        parser.add_argument('-r', '--reproj_threshold', type=float, default=16.0)
        parser.add_argument('-w', '--weight_sampson', type=float, default=1.0)
        parser.add_argument('--subsample_rate', type=float, default=0.05)
        parser.add_argument('--keyframes', type=int, default=0)
        parser.add_argument('--use_cache', action='store_true', default=False)
        parser.add_argument('-v', '--verbose', action='store_true', default=False)
        parser.add_argument('--cache_path', type=str, default=None)
        parser.add_argument('--output_path', type=str, default=None)
        parser.add_argument('video_path')

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
        if self.anchor_mde_out is None:
            mde_out2 = self.mde_model.infer(image2)
        else:
            mde_out2 = self.anchor_mde_out

        # Extract SuperPoint features
        feats1_out = self.extractor.extract(image1)
        if self.anchor_feats is None:
            feats2_out = self.extractor.extract(image2)
        else:
            feats2_out = self.anchor_feats

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

        if self.anchor_K is None:
            K2, cam_dict_2 = self.get_camera_data(image2, mde_out2)
        else:
            K2, cam_dict_2 = self.anchor_K, self.anchor_cam_dict

        K1, cam_dict_1 = self.get_camera_data(image1, mde_out1)

        d = np.column_stack([depths1, depths2])


        pose, info = poselib.estimate_relative_pose_w_mono_depth(points1[l], points2[l], d[l],
                                                                 cam_dict_1, cam_dict_2, self.ransac_dict, {})


        if self.args.verbose:
            print(info['inlier_ratio'])
            print("R: ")
            print(pose.R)
            print("t: ", pose.t)
            print("scale: ", pose.scale)

        pcd_1, colors_1 = self.get_pointcloud(np.linalg.inv(K1), mde_out1, image1)
        # pcd_1 /= pose.scale
        # pcd_1 = (pose.R @ pcd_1.T).T + pose.t

        if self.args.keyframes:
            if info['inlier_ratio'] > 0.75 and self.since_last_anchor > self.args.keyframes \
                    and np.sum(info['inliers']) > 200:
                if self.args.verbose:
                    print("Setting new keyframe")
                self.anchor_K = K1
                self.anchor_feats = feats1_out
                self.anchor_cam_dict = cam_dict_1
                self.anchor_mde_out = mde_out1

                self.anchor_R = pose.R @ self.anchor_R
                self.anchor_t = pose.R @ self.anchor_t + pose.t
                self.anchor_scale *= pose.scale

                self.since_last_anchor = 0

                return pcd_1, colors_1, K1, self.anchor_R, self.anchor_t, self.anchor_scale

            self.since_last_anchor += 1

            return pcd_1, colors_1, K1, \
                pose.R @ self.anchor_R, \
                pose.R @ self.anchor_t + pose.t, \
                self.anchor_scale * pose.scale

        return pcd_1, colors_1, K1, pose.R, pose.t, pose.scale

    def postprocess(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, 25.0, (3 * self.new_width, self.new_height))

        for i in range(self.video_length):
            img_pcd = cv2.imread(os.path.join(self.cache_dir, f'frame_ours_{i:05d}.png'))
            img_moge = cv2.imread(os.path.join(self.cache_dir, f'frame_naive_{i:05d}.png'))

            ret, frame = self.cap.read()
            if ret is None or img_pcd is None or img_moge is None:
                break

            frame_resize = cv2.resize(frame, (self.new_width, self.new_height))

            output_frame = np.concatenate([frame_resize, img_moge, img_pcd], axis=1)
            self.writer.write(output_frame)

        self.cap.release()
        self.writer.release()

    def process_video_rr(self, video_path):
        self.i = 0
        self.cap = cv2.VideoCapture(video_path)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_width = self.video_width // 2
        self.new_height = self.video_height // 2

        ret, frame = self.cap.read()

        xyz, colors = self.infer_on_initial_anchor(self.preprocess_image(frame))

        blueprint = rrb.Blueprint(rrb.Spatial3DView())
        rr.init("points3d_camera", default_blueprint=blueprint, spawn=True)
        # rr.init("rerun_example_points3d", spawn=True)

        rr.set_time("time", duration=self.i)

        rr.log("world/camera/image", rr.Pinhole(image_from_camera=self.anchor_K, width=self.video_width, height=self.video_height, camera_xyz=rr.ViewCoordinates.RDF,))
        rr.log("world/camera/image", rr.Transform3D(translation=np.zeros(3), mat3x3=np.eye(3)))
        rr.log("world/camera/image/rgb", rr.Image(frame, color_model="bgr"))

        rr.log("world/points", rr.Points3D(xyz, colors=colors))


        while True:
            ret, frame = self.cap.read()
            msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            if not ret or frame is None:
                print("Done")
                return

            self.i += 1
            xyz, colors, K, R, t, scale = self.infer_images(self.preprocess_image(frame), None)

            rr.set_time("time", duration=msec/1000)

            rr.log("world/camera/image",
                   rr.Pinhole(image_from_camera=K, width=self.video_width, height=self.video_height,
                              camera_xyz=rr.ViewCoordinates.RDF))
            rr.log("world/camera/image", rr.Transform3D(translation=t, mat3x3=R))
            rr.log("world/camera/image/rgb", rr.Image(frame, color_model="bgr"))

            xyz = (R @ (xyz / scale).T) + t
            rr.log("world/points", rr.Points3D(xyz, colors=colors))

            print(f"Updated {self.i} / {self.video_length}")



    def run_visualizer(self, key):
        self.vis.capture_screen_image(os.path.join(self.cache_dir, f'frame_ours_{self.i:05d}.png'), False)
        self.vis.capture_screen_image(os.path.join(self.cache_dir, f'frame_naive_{self.i:05d}.png'), False)

        with tqdm(total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
            pbar.update(1)
            while True:
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    print("Done")
                    self.vis.close()
                    self.postprocess()
                    return

                self.i += 1

                xyz, colors, K, R, t, scale = self.infer_images(self.preprocess_image(frame), None)

                l = np.random.rand(len(xyz)) < self.args.subsample_rate
                l = np.logical_and(l, ~np.any(np.isinf(xyz), axis=1))
                self.pcd.points = o3d.utility.Vector3dVector(xyz[l].astype(np.float64))
                self.pcd.colors = o3d.utility.Vector3dVector(colors[l])

                self.vis.update_geometry(self.pcd)
                self.vis.update_geometry(self.vis_camera)
                self.vis.poll_events()
                self.vis.update_renderer()
                self.vis.capture_screen_image(os.path.join(self.cache_dir, f'frame_naive_{self.i:05d}.png'), False)

                self.pcd.scale(1 / scale, np.zeros(3))
                self.pcd.rotate(R)
                self.pcd.translate(t)
                self.vis_camera.rotate(R)
                self.vis_camera.translate(t)

                self.vis.update_geometry(self.pcd)
                self.vis.update_geometry(self.vis_camera)
                self.vis.poll_events()
                self.vis.update_renderer()
                self.vis.capture_screen_image(os.path.join(self.cache_dir, f'frame_ours_{self.i:05d}.png'), False)

                # print(f"Updated {self.i} / {self.video_length}")

                self.vis_camera.translate(-t)
                self.vis_camera.rotate(R.T)
                pbar.update(1)

    def process_video(self):
        self.i = 0
        self.cap = VideoCaptureProxy(self.args.video_path)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_width = self.video_width #// 2
        self.new_height = self.video_height #// 2

        # we have images in cahce so we can skip
        if self.args.use_cache:
            self.postprocess()
            return

        ret, frame = self.cap.read()

        xyz, colors = self.infer_on_initial_anchor(self.preprocess_image(frame))

        self.pcd = o3d.geometry.PointCloud()
        l = np.random.rand(len(xyz)) < self.args.subsample_rate
        l = np.logical_and(l, ~np.any(np.isinf(xyz), axis=1))
        self.pcd.points = o3d.utility.Vector3dVector(xyz[l].astype(np.float64))
        self.pcd.colors = o3d.utility.Vector3dVector(colors[l])

        Rt = np.eye(4)
        self.vis_camera = o3d.geometry.LineSet.create_camera_visualization(view_width_px=self.video_width,
                                                                           view_height_px=self.video_height,
                                                                          intrinsic=self.anchor_K,
                                                                          extrinsic=Rt)
        self.vis_camera.scale(0.05 * np.max(np.linalg.norm(xyz, axis=1)), np.zeros(3))
        print("First point cloud loaded!")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord(' '), self.run_visualizer)
        self.vis.create_window(width=self.new_width, height=self.new_height)
        self.vis.get_render_option().background_color = [0, 0, 0]
        self.vis.get_render_option().point_size = 1

        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.vis_camera)
        self.vis.poll_events()
        self.vis.update_renderer()
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
    v.process_video()