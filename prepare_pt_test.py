import argparse
import glob
import itertools
import ntpath
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import lightglue
from lightglue.utils import load_image

from utils.matching import LoFTRMatcher, get_matcher_string, get_extractor, read_loftr_image
from utils.read_write_colmap import cam_to_K, read_model, Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples', type=int, default=None)
    parser.add_argument('-s', '--seed', type=int, default=100)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-f', '--features', type=str, default='superpoint')
    parser.add_argument('-mf', '--max_features', type=int, default=2048)
    parser.add_argument('-r', '--resize', type=int, default=None)
    parser.add_argument('-m', '--monodepth_method', type=str, default='moge')
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('out_path')
    parser.add_argument('dataset_path')

    return parser.parse_args()


def create_gt_h5(dataset_path, image_list, out_dir, args):
    exist = [os.path.exists(os.path.join(out_dir, f'{x}.h5')) for x in ['K', 'R', 'T']]
    if not False in exist and not args.recalc:
        print(f"GT info exists in {out_dir} - not creating it anew")
        return

    print(f"Writing GT info to {out_dir}")
    fK = h5py.File(os.path.join(out_dir, 'K.h5'), 'w')
    fR = h5py.File(os.path.join(out_dir, 'R.h5'), 'w')
    fT = h5py.File(os.path.join(out_dir, 'T.h5'), 'w')

    for image in image_list:
        calib_path = os.path.join(dataset_path, 'set_100', 'calibration', f'calibration_{image}.h5')

        f = h5py.File(calib_path)

        R = np.array(f['R'])
        t = np.array(f['T'])
        K = np.array(f['K'])

        fR.create_dataset(image, shape=(3, 3), data=R)
        fT.create_dataset(image, shape=(3, 1), data=t.reshape(3,1))
        fK.create_dataset(image, shape=(3, 3), data=K)


def extract_features(dataset_path, image_list, out_dir, args):
    # extractor = lightglue.SuperPoint(max_num_keypoints=2048).eval().cuda()
    extractor = get_extractor(args)
    out_path = os.path.join(out_dir, f"{get_matcher_string(args)}.pt")

    if os.path.exists(out_path) and not args.recalc:
        print(f"Features already found in {out_path}")
        return

    print("Extracting features")
    feature_dict = {}

    for image in tqdm(image_list):
        img_path = os.path.join(dataset_path, 'set_100', 'images', f'{image}.jpg')
        image_tensor = load_image(img_path).cuda()

        kp_tensor = extractor.extract(image_tensor, resize=args.resize)
        feature_dict[image] = kp_tensor

    torch.save(feature_dict, out_path)
    print("Features saved to: ", out_path)

# def extract_depth(dataset_path, image_list, out_dir, args):
#     # extractor = get_depth_extractor(args)
#
#     monodepth_dir = os.path.join(out_dir, f"{args.monodepth_extractor}")


def add_depths(h5_file, depth_dir, img_1, img_2, coords_1, coords_2):
    depth_methods = os.listdir(depth_dir)

    for depth_method in depth_methods:
        depth_method_dir = os.path.join(depth_dir, depth_method)
        depths_1 = np.load(glob.glob(f'{depth_method_dir}/*{img_1}*.npy')[0])
        depths_2 = np.load(glob.glob(f'{depth_method_dir}/*{img_2}*.npy')[0])

        out_array = np.empty([len(coords_1), 2])
        for i in range(len(coords_1)):
            x1, y1 = coords_1[i]
            x2, y2 = coords_2[i]
            out_array[i][0] = depths_1[y1, x1]
            out_array[i][1] = depths_2[y2, x2]

        h5_file.create_dataset(f'{img_1}-{img_2}-{depth_method}', shape=out_array.shape, data=out_array)

def create_pairs(dataset_path, image_list, out_dir, args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    features = torch.load(os.path.join(out_dir, f"{get_matcher_string(args)}.pt"))
    depth_dir = os.path.join(dataset_path, 'depths')

    matcher = lightglue.LightGlue(features=args.features).eval().cuda()

    h5_path_str = f'pairs-{get_matcher_string(args)}-LG.h5'
    txt_path_str = f'pairs-{get_matcher_string(args)}-LG.txt'
    h5_path = os.path.join(out_dir, h5_path_str)
    h5_file = h5py.File(h5_path, 'w')
    print("Writing matches to: ", h5_path)

    pairs = list(itertools.combinations(image_list, 2))

    txt_path = os.path.join(out_dir, txt_path_str)
    print("Writing list of pairs to: ", txt_path)

    print("Writing list of triplets to: ", txt_path)
    with open(txt_path, 'w') as f:
        f.writelines(f'{img_1} {img_2}\n' for img_1, img_2 in pairs)

    for img_1, img_2 in tqdm(pairs):

        label = f'{img_1}-{img_2}'

        if label in h5_file:
            continue

        feats_1 = features[img_1]
        feats_2 = features[img_2]

        out = matcher({'image0': feats_1, 'image1': feats_2})
        scores = out['matching_scores0'][0].detach().cpu().numpy()
        matches = out['matches0'][0].detach().cpu().numpy()

        idxs = []

        for idx_1, idx_2 in enumerate(matches):
            if idx_2 != -1:
                idxs.append((idx_1, idx_2))

        out_array = np.empty([len(idxs), 5], dtype=float)

        coords_1 = []
        coords_2 = []

        for i, idx in enumerate(idxs):
            idx_1, idx_2 = idx
            point_1 = feats_1['keypoints'][0, idx_1].detach().cpu().numpy()
            point_2 = feats_2['keypoints'][0, idx_2].detach().cpu().numpy()

            coords_1.append(np.floor(point_1).astype(int))
            coords_2.append(np.floor(point_2).astype(int))

            score = scores[idx_1]
            out_array[i] = np.array([*point_1, *point_2, score])

        h5_file.create_dataset(label, shape=out_array.shape, data=out_array)
        add_depths(h5_file, depth_dir, img_1, img_2, coords_1, coords_2)


def prepare_single(args, subset):
    dataset_path = os.path.join(args.dataset_path, subset)

    with open(os.path.join(dataset_path, 'set_100', 'images.txt'), 'r') as f:
        image_list = [ntpath.normpath(x).split(ntpath.sep)[1].split('.')[0] for x in f.readlines()]

    out_dir = os.path.join(args.out_path, subset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    create_gt_h5(dataset_path, image_list, out_dir, args)
    extract_features(dataset_path, image_list, out_dir, args)
    # extract_depth(dataset_path, image_list, out_dir, args)
    create_pairs(dataset_path, image_list, out_dir, args)

def run_im(args):
    dataset_path = Path(args.dataset_path)
    dir_list = [x for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))]

    for subset in dir_list:
        prepare_single(args, subset)

if __name__ == '__main__':
    args = parse_args()
    run_im(args)