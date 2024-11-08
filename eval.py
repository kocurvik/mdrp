import argparse
import json
import os
from multiprocessing import Pool
from time import perf_counter

import h5py
import numpy as np
import poselib
# import pykitti
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm

from utils.geometry import rotation_angle, angle, get_camera_dicts, force_inliers
from utils.vis import draw_results_pose_auc_10


# from utils.vis import draw_results_pose_auc_10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-i', '--force_inliers', type=float, default=None)
    parser.add_argument('-t', '--threshold', type=float, default=1.0)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('-s', '--shuffle', type=float, default=0.0)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('-o', '--overwrite', action='store_true', default=False)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()

# def get_pairs(file):
#     return [tuple(x.split('-')) for x in file.keys() if 'feat' not in x and 'desc' not in x]

def get_pairs(file):
    with open(file, 'r') as f:
        pairs = f.readlines()
    return [tuple(x.strip().split(' ')) for x in pairs]

def get_result_dict(info, pose_est, R_gt, t_gt):
    out = {}

    R_est, t_est = pose_est.R, pose_est.t

    out['R_err'] = rotation_angle(R_est.T @ R_gt)
    out['t_err'] = angle(t_est, t_gt)
    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = R_est.tolist()
    out['t_gt'] = R_gt.tolist()

    out['P_err'] = max(out['R_err'], out['t_err'])
    out['P_err'] = out['R_err']

    info['inliers'] = []
    out['info'] = info

    return out


def eval_experiment(x):
    iters, experiment, kp1, kp2, d, R_gt, t_gt, K1, K2, t = x

    lo_iterations = 0 if 'nLO' in experiment else 25

    if iters is None:
        ransac_dict = {'max_iterations': 5000, 'max_epipolar_error': t, 'progressive_sampling': False,
                       'min_iterations': 50, 'lo_iterations': lo_iterations}
    else:
        ransac_dict = {'max_iterations': iters, 'max_epipolar_error': t, 'progressive_sampling': False,
                       'min_iterations': iters, 'lo_iterations': lo_iterations}

    ransac_dict['all_permutations'] = True
    ransac_dict['use_astermark'] = 'reldepth' in experiment

    bundle_dict = {'max_iterations': 0 if lo_iterations == 0 else 100}

    camera1 = {'model': 'PINHOLE', 'width': -1, 'height': -1, 'params': [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]}
    camera2 = {'model': 'PINHOLE', 'width': -1, 'height': -1, 'params': [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]]}

    if '5p_nister' in experiment:
        start = perf_counter()
        pose_est, info = poselib.estimate_relative_pose(kp1, kp2, camera1, camera2, ransac_dict, bundle_dict)
        info['runtime'] = 1000 * (perf_counter() - start)
    else:
        start = perf_counter()
        pose_est, info = poselib.estimate_relative_pose_w_mono_depth(kp1, kp2, d, camera1, camera2, ransac_dict, bundle_dict)
        info['runtime'] = 1000 * (perf_counter() - start)


    result_dict = get_result_dict(info, pose_est, R_gt, t_gt)
    result_dict['experiment'] = experiment

    return result_dict


def print_results(experiments, results, eq_only=False):
    tab = PrettyTable(['solver', 'median pose err', 'mean pose err',
                       'Pose AUC@5', 'Pose AUC@10', 'Pose AUC@20',
                       'median time', 'mean time', 'median inliers', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        p_errs[np.isnan(p_errs)] = 180
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 21)])

        times = np.array([x['info']['runtime'] for x in exp_results])
        inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

        exp_name = exp


        tab.add_row([exp_name, np.median(p_errs), np.mean(p_errs),
                     100*np.mean(p_res[:5]), 100*np.mean(p_res[:10]), 100*np.mean(p_res),
                     np.median(times), np.mean(times),
                     np.median(inliers), np.mean(inliers)])
    print(tab)

    print('latex')

    print(tab.get_formatted_string('latex'))

def draw_cumplots(experiments, results):
    plt.figure()
    plt.xlabel('Pose error')
    plt.ylabel('Portion of samples')

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]
        exp_name = exp
        label = f'{exp_name}'

        R_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        R_res = np.array([np.sum(R_errs < t) / len(R_errs) for t in range(1, 180)])
        plt.plot(np.arange(1, 180), R_res, label = label)

    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('k error')
    plt.ylabel('Portion of samples')

def shuffle_portion(kp: np.ndarray, s: float) -> np.ndarray:
    num_rows_to_shuffle = int(s * kp.shape[0])
    indices_to_shuffle = np.random.choice(kp.shape[0], num_rows_to_shuffle, replace=False)
    rows_to_shuffle = kp[indices_to_shuffle]
    np.random.shuffle(rows_to_shuffle)
    shuffled_kp = kp.copy()
    shuffled_kp[indices_to_shuffle] = rows_to_shuffle

    return shuffled_kp


def eval(args):
    experiments = ['5p_nister', '3dp_monodepth+moge', '3dp_monodepth+ml-depth-pro-bm', '3dp_monodepth+metric3d', '3dp_monodepth+marigold-old', '3dp_monodepth+marigold-bm']
    # experiments = ['5p_nister', '3dp_monodepth+moge', '3dp_monodepth+marigold-bm', '3dp_reldepth+moge', '3dp_reldepth+marigold-bm']
    experiments = [f'nLO-{x}' for x in experiments]

    dataset_path = args.dataset_path
    basename = os.path.basename(dataset_path)

    if args.force_inliers is not None:
        basename = f'{basename}-{args.force_inliers:.1f}inliers'

    if args.threshold != 1.0:
        basename = f'{basename}-{args.threshold}t'

    matches_basename = os.path.basename(args.feature_file)

    if args.graph:
        basename = f'{basename}-graph'
        iterations_list = [10, 20, 50, 100, 200, 500, 1000]
    else:
        iterations_list = [args.iters]

    if args.shuffle == 0.0:
        json_string = f'twoview-{basename}-{matches_basename}.json'
    else:
        json_string = f'twoview-{basename}-{matches_basename}-{args.shuffle}s.json'


    if args.load:
        print("Loading: ", json_string)
        with open(os.path.join('results', json_string), 'r') as f:
            results = json.load(f)

    else:
        R_file = h5py.File(os.path.join(dataset_path, 'R.h5'))
        T_file = h5py.File(os.path.join(dataset_path, 'T.h5'))
        K_file = h5py.File(os.path.join(dataset_path, 'K.h5'))
        C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))
        pairs = get_pairs(os.path.join(dataset_path, f'{args.feature_file}.txt'))
        if args.first is not None:
            pairs = pairs[:args.first]

        def gen_data():
            for img_name_1, img_name_2 in pairs:
                for experiment in experiments:
                    for iterations in iterations_list:
                        R1 = np.array(R_file[img_name_1])
                        t1 = np.array(T_file[img_name_1])
                        R2 = np.array(R_file[img_name_2])
                        t2 = np.array(T_file[img_name_2])
                        K1 = np.array(K_file[img_name_1])
                        K2 = np.array(K_file[img_name_2])

                        R_gt = R2 @ R1.T
                        t_gt = t2 - R_gt @ t1

                        matches = np.array(C_file[f'{img_name_1}-{img_name_2}'])
                        kp1 = matches[:, :2]
                        kp2 = matches[:, 2:4]

                        if '+' in experiment:
                            depth_method = experiment.split('+')[1]
                            d = C_file[f'{img_name_1}-{img_name_2}-{depth_method}']
                        else:
                            d = np.ones_like(kp1)

                        yield iterations, experiment, np.copy(kp1), np.copy(kp2), \
                            np.copy(d), R_gt, t_gt, K1, K2, args.threshold

        total_length = len(experiments) * len(iterations_list) * len(pairs)

        print(f"Total runs: {total_length} for {len(pairs)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        os.makedirs('results', exist_ok=True)

        if args.append:
            print(f"Appending from: {os.path.join('results', json_string)}")
            with open(os.path.join('results', json_string), 'r') as f:
                prev_results = json.load(f)

            if args.overwrite:
                prev_results = [x for x in prev_results if x['experiment'] not in experiments]

            results.extend(prev_results)

        with open(os.path.join('results', json_string), 'w') as f:
            json.dump(results, f)

        print("Done")

    print_results(experiments, results)
    draw_cumplots(experiments, results)

    if args.graph:
        draw_results_pose_auc_10(results, experiments, iterations_list)


if __name__ == '__main__':
    args = parse_args()
    eval(args)