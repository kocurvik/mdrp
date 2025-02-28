import argparse
import json
import os

import h5py
import numpy as np

from scipy.spatial.transform import Rotation

from utils.data import R_err_fun, t_err_fun

from utils.eval_utils import print_results_focal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float, default=1.0)
    parser.add_argument('-r', '--reproj_threshold', type=float, default=16.0)
    parser.add_argument('-s', '--shared', default=False, action='store_true')
    parser.add_argument('dataset_path')
    parser.add_argument('master_result_path')

    return parser.parse_args()


def extract_relative_pose(filename):
    """
    Extract the relative rotation matrix and translation vector between two camera views from a file.

    Args:
        filename (str): Path to the file containing camera pose information

    Returns:
        tuple: (R_relative, t_relative) where:
            - R_relative: 3x3 numpy array representing the relative rotation matrix
            - t_relative: 3x1 numpy array representing the relative translation vector
    """
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extract camera poses from the relevant lines (skip comment lines starting with #)
    camera_poses = []
    for line in lines:
        if not line.startswith('#'):
            parts = line.strip().split()
            if len(parts) >= 8:  # Ensure line has enough data
                # Extract quaternion (QW, QX, QY, QZ) and translation (TX, TY, TZ)
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])

                # Store pose information
                camera_poses.append({
                    'quaternion': np.array([qw, qx, qy, qz]),
                    'translation': np.array([tx, ty, tz])
                })

    # Ensure we have at least two camera poses
    if len(camera_poses) < 2:
        raise ValueError("File must contain at least two camera poses")

    # Convert quaternions to rotation matrices using SciPy
    # Note: SciPy uses quaternions in the form [x, y, z, w] so we need to reorder
    q1 = camera_poses[0]['quaternion']
    q2 = camera_poses[1]['quaternion']

    # Reorder from [w, x, y, z] to [x, y, z, w] for SciPy
    q1_scipy = np.array([q1[1], q1[2], q1[3], q1[0]])
    q2_scipy = np.array([q2[1], q2[2], q2[3], q2[0]])

    # Create rotation objects and get rotation matrices
    R1 = Rotation.from_quat(q1_scipy).as_matrix()
    R2 = Rotation.from_quat(q2_scipy).as_matrix()

    # Get translation vectors
    t1 = camera_poses[0]['translation'].reshape(3, 1)
    t2 = camera_poses[1]['translation'].reshape(3, 1)

    # Calculate relative rotation: R_relative = R2 * R1^T
    R_relative = np.dot(R2, R1.T)

    # Calculate relative translation: t_relative = t2 - R_relative * t1
    t_relative = t2 - np.dot(R_relative, t1)

    return R_relative, t_relative


def extract_focal(filename):
    """
    Extract the first parameter for both cameras from a file with the given format.

    Args:
        filename (str): Path to the file containing camera parameters

    Returns:
        list: First parameter values for each camera [param_camera0, param_camera1, ...]
    """
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Initialize list to store the first parameter for each camera
    focals = []

    # Extract camera parameters from the relevant lines (skip comment lines starting with #)
    for line in lines:
        if not line.startswith('#'):
            parts = line.strip().split()
            if len(parts) >= 6:  # Ensure line has enough data
                # The first parameter is at index 4 after CAMERA_ID, MODEL, WIDTH, HEIGHT
                first_param = float(parts[4])
                focals.append(first_param)

    # Return the list of first parameters
    return focals

def get_result_dict(info, R_est, t_est, f1_est, f2_est, R_gt, t_gt, f1_gt, f2_gt):
    out = {}

    # out['R_err'] = rotation_angle(R_est.T @ R_gt)
    # out['t_err'] = angle(t_est, t_gt)
    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = t_est.tolist()
    out['t_gt'] = t_gt.tolist()
    out['f1_gt'] = f1_gt
    out['f1'] = f1_est
    out['f2_gt'] = f2_gt
    out['f2'] = f2_est

    out['R_err'] = R_err_fun(out)
    out['t_err'] = t_err_fun(out)
    # out['P_err'] = max(out['R_err'], out['t_err'])
    # out['P_err'] = out['R_err']

    out['f1_err'] = np.abs(out['f1'] - f1_gt) / f1_gt
    out['f2_err'] = np.abs(out['f2'] - f2_gt) / f2_gt
    out['f_err'] = np.sqrt(out['f1_err'] * out['f2_err'])

    info['inliers'] = []
    out['info'] = info

    return out

def load_master_results(dataset_path, dir, min_cor=7):

    runtimes = np.loadtxt(os.path.join(dir, 'runtimes.txt')).tolist()

    H5_file = h5py.File(dataset_path)

    prelim_pairs = [x.split('corr_')[1] for x in H5_file.keys() if 'corr_' in x]

    pairs = [(pair.split('_o_')[0] + '_o', pair.split('_o_')[1]) for pair in prelim_pairs]

    results = []

    for runtime, pair in zip(runtimes, pairs):
        img_name_1, img_name_2 = pair
        Rt = np.array(H5_file[f'pose_{img_name_1}_{img_name_2}'])
        R_gt = Rt[:3, :3]
        t_gt = Rt[:, 3]

        K1_gt = np.array(H5_file[f'K_{img_name_1}'])
        f1_gt = (K1_gt[0, 0] + K1_gt[1, 1]) / 2
        K2_gt = np.array(H5_file[f'K_{img_name_2}'])
        f2_gt = (K2_gt[0, 0] + K2_gt[1, 1]) / 2

        data = np.array(H5_file[f'corr_{img_name_1}_{img_name_2}'])

        if len(data) < min_cor:
            continue

        if "_o" in img_name_1:
            pose_file = os.path.join(dir, 'pose', f'{img_name_1.split("_o")[0]}_frame{img_name_2.split("_o")[0].split("frame")[1]}.txt')
            calib_file = os.path.join(dir, 'camera',
                                  f'{img_name_1.split("_o")[0]}_frame{img_name_2.split("_o")[0].split("frame")[1]}.txt')
        else:
            pose_file = os.path.join(dir, 'pose',
                                     f'{img_name_1}_{img_name_2}.txt')
            calib_file = os.path.join(dir, 'camera',
                                      f'{img_name_1}_{img_name_2}.txt')

        R, t = extract_relative_pose(pose_file)
        f1, f2 = extract_focal(calib_file)

        info = {'runtime': 1000 * runtime, 'inlier_ratio': 1.0}
        rd = get_result_dict(info, R, t, f1, f2, R_gt, t_gt, f1_gt, f2_gt)
        rd['experiment'] = 'mast3r'
        results.append(rd)

    return results

if __name__ == '__main__':
    args = parse_args()

    basename = os.path.basename(args.dataset_path).split('.')[0]
    if args.threshold != 1.0:
        basename = f'{basename}-{args.threshold}t'
    if args.reproj_threshold != 16.0:
        basename = f'{basename}-{args.reproj_threshold}r'

    min_samples = 6 if args.shared else 7
    master_results = load_master_results(args.dataset_path, os.path.join('mast3r_results', args.master_result_path),
                                         min_samples)

    if args.shared:
        json_string = f'shared_focal-{basename}.json'
    else:
        json_string = f'varying_focal-{basename}.json'

    json_path = os.path.join('results', json_string)

    print("Loading: ", json_string)
    with open(json_path, 'r') as f:
        results = json.load(f)

    results = [x for x in results if x['experiment'] != 'mast3r']
    results.extend(master_results)

    experiments = sorted(list(set([x['experiment'] for x in results])))
    print_results_focal(experiments, results)

