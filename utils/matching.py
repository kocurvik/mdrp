import os

import numpy as np
import lightglue
from kornia.feature import LoFTR
import cv2
import torch

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

class LoFTRMatcher():
    matcher_str = 'loftr'

    def __init__(self, weights='outdoor', device='cuda', max_dim=512):
        self.loftr = LoFTR(weights).to(device)
        self.device = device
        self.max_dim = max_dim

    def enforce_dim(self, img):
        if self.max_dim is None:
            return img, 1.0

        h, w = img.shape[:2]

        gr = max(h, w)

        if gr > self.max_dim:
            scale_factor = self.max_dim / gr
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            return img, scale_factor
        else:
            return img, 1.0

    def match(self, img_1, img_2):
        img1_b = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img2_b = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        img1_b, s1 = self.enforce_dim(img1_b)
        img2_b, s2 = self.enforce_dim(img2_b)

        data = {'image0': frame2tensor(img1_b, self.device), 'image1': frame2tensor(img2_b, self.device)}
        pred = self.loftr(data)
        kp_1 = pred['keypoints0'].detach().cpu().numpy() / s1
        kp_2 = pred['keypoints1'].detach().cpu().numpy() / s2
        conf = pred['confidence'].detach().cpu().numpy()

        return conf, kp_1, kp_2


def get_matcher_string(args):
    if args.resize is None:
        resize_str = 'noresize'
    else:
        resize_str = str(args.resize)

    return f'features_{args.features}_{resize_str}_{args.max_features}'


def get_extractor(args):
    if args.features == 'superpoint':
        extractor = lightglue.SuperPoint(max_num_keypoints=args.max_features).eval().cuda()
    elif args.features == 'disk':
        extractor = lightglue.DISK(max_num_keypoints=args.max_features).eval().cuda()
    elif args.features == 'sift':
        extractor = lightglue.SIFT(max_num_keypoints=args.max_features).eval().cuda()
    else:
        raise NotImplementedError

    return extractor


def read_loftr_image(img_dir_path, img, cameras):
    img_path = os.path.join(img_dir_path, img.name)
    image_array = cv2.imread(img_path)
    cam = cameras[img.camera_id]

    if cam.width != image_array.shape[1]:
        if cam.width == image_array.shape[0]:
            image_array = np.swapaxes(image_array, -2, -1)
        else:
            print(f"Image dimensions do not comply with camera width and height for: {img_path} - skipping!")
            return None
    return image_array
