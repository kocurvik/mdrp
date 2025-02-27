import argparse
import json
from multiprocessing import Process, Queue
import time
import os
import signal
from time import perf_counter

import h5py
import numpy as np
import poselib
import madpose
# import pykitti
from tqdm import tqdm

from utils.data import depth_indices, R_err_fun, t_err_fun, get_valid_depth_mask
from utils.eval_utils import print_results, NoDaemonProcessPool, get_exception_result_dict
from utils.madpose import madpose_opt_from_dict
from utils.vis import draw_results_pose_auc_10, draw_cumplots


# from utils.vis import draw_results_pose_auc_10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-i', '--force_inliers', type=float, default=None)
    parser.add_argument('-t', '--threshold', type=float, default=1.0)
    parser.add_argument('-r', '--reproj_threshold', type=float, default=16.0)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('-o', '--overwrite', action='store_true', default=False)
    parser.add_argument('--graduated', action='store_true', default=False)
    parser.add_argument('--faster', action='store_true', default=False)
    parser.add_argument('--nlo', action='store_true', default=False)
    parser.add_argument('--nmad', action='store_true', default=False)
    parser.add_argument('--madours', action='store_true', default=False)
    parser.add_argument('--madonly', action='store_true', default=False)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('dataset_path')

    return parser.parse_args()

def get_result_dict(info, pose_est, R_gt, t_gt):
    out = {}

    R_est, t_est = pose_est.R, pose_est.t

    # out['R_err'] = rotation_angle(R_est.T @ R_gt)
    # out['t_err'] = angle(t_est, t_gt)
    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = t_est.tolist()
    out['t_gt'] = t_gt.tolist()

    out['R_err'] = R_err_fun(out)
    out['t_err'] = t_err_fun(out)
    # out['P_err'] = max(out['R_err'], out['t_err'])
    # out['P_err'] = out['R_err']

    info['inliers'] = []
    out['info'] = info

    return out
def get_result_dict_madpose(stats, pose_est, R_gt, t_gt):
    out = {}

    R_est, t_est = pose_est.R(), pose_est.t()

    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = t_est.tolist()
    out['t_gt'] = t_gt.tolist()

    out['R_err'] = R_err_fun(out)
    out['t_err'] = t_err_fun(out)

    info = {}

    info['num_inliers'] = stats.best_num_inliers
    info['inlier_ratio'] = stats.inlier_ratios[0]
    out['info'] = info

    return out


def eval_experiment(x):
    iters, experiment, kp1, kp2, d, R_gt, t_gt, K1, K2, t, r = x

    lo_iterations = 0 if 'nLO' in experiment else 25

    if iters is None:
        ransac_dict = {'max_iterations': 1000, 'max_epipolar_error': t, 'progressive_sampling': False,
                       'min_iterations': 1000, 'lo_iterations': lo_iterations, 'max_reproj_error': r}
    else:
        ransac_dict = {'max_iterations': iters, 'max_epipolar_error': t, 'progressive_sampling': False,
                       'min_iterations': iters, 'lo_iterations': lo_iterations, 'max_reproj_error': r}

    ransac_dict['all_permutations'] = True

    ransac_dict['use_reldepth'] = 'reldepth' in experiment
    ransac_dict['use_p3p'] = 'p3p' in experiment

    ransac_dict['use_ours'] = 'ours' in experiment
    ransac_dict['solver_shift'] = 'shift' in experiment
    ransac_dict['solver_scale'] = 'scale' in experiment

    ransac_dict['use_reproj'] = 'reproj' in experiment
    ransac_dict['optimize_shift'] = 'reproj-s' in experiment

    ransac_dict['graduated_steps'] = 3 if 'GLO' in experiment else 0

    bundle_dict = {'max_iterations': 0 if lo_iterations == 0 else 100}

    camera1 = {'model': 'PINHOLE', 'width': -1, 'height': -1, 'params': [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]}
    camera2 = {'model': 'PINHOLE', 'width': -1, 'height': -1, 'params': [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]]}

    if '5p' in experiment:
        start = perf_counter()
        pose_est, info = poselib.estimate_relative_pose(kp1, kp2, camera1, camera2, ransac_dict, bundle_dict)
        info['runtime'] = 1000 * (perf_counter() - start)
    # elif 'reldepth' in experiment:
    #     start = perf_counter()
    #     pose_est, info = poselib.estimate_relative_pose_w_relative_depth(kp1, kp2, d[:, 1] / d[:, 0], camera1, camera2, ransac_dict, bundle_dict)
    #     info['runtime'] = 1000 * (perf_counter() - start)
    elif 'madpose' in experiment:
        opt, est_config = madpose_opt_from_dict(ransac_dict)
        start = perf_counter()
        pose, info = madpose.HybridEstimatePoseScaleOffset(kp1, kp2, d[:, 0], d[:, 1], [d[:, 0].min(), d[:, 1].min()],
            K1, K2, opt, est_config)
        result_dict = get_result_dict_madpose(info, pose, R_gt, t_gt)
        result_dict['info']['runtime'] = 1000 * (perf_counter() - start)
        result_dict['experiment'] = experiment
        return result_dict
    else:
        start = perf_counter()
        pose_est, info = poselib.estimate_relative_pose_w_mono_depth(kp1, kp2, d, camera1, camera2, ransac_dict, bundle_dict)
        info['runtime'] = 1000 * (perf_counter() - start)


    result_dict = get_result_dict(info, pose_est, R_gt, t_gt)
    result_dict['experiment'] = experiment

    return result_dict


def eval_experiment_wrapper(x, result_queue):
    pid = os.getpid()

    try:
        result = eval_experiment(x)
        result_queue.put((result, pid))
    except Exception as e:
        print(f"Process {pid}: Error in experiment: {e}")
        result_queue.put((get_exception_result_dict(x), pid))

def eval_experiment_wrapper(x, result_queue):
    pid = os.getpid()

    try:
        result = eval_experiment(x)
        result_queue.put(result)
    except Exception as e:
        print(f"Process {pid}: Error in experiment: {e}")
        result_queue.put(get_exception_result_dict(x))

def run_with_timeout(x, timeout=20):
    result_queue = Queue()
    process = Process(target=eval_experiment_wrapper, args=(x, result_queue))
    process.start()
    process_pid = process.pid
    process.join(timeout)

    if process.is_alive():
        print(f"Process {process_pid} timed out after {timeout} seconds. Terminating...")
        process.terminate()
        time.sleep(0.1)
        if process.is_alive():
            print(f"Process {process_pid} didn't terminate. Sending SIGKILL...")
            try:
                os.kill(process.pid, signal.SIGKILL)
            except OSError:
                pass
        process.join(1)
        return get_exception_result_dict(x)

    if not result_queue.empty():
        return result_queue.get()
    else:
        return get_exception_result_dict(x)

def eval(args):
    dataset_path = args.dataset_path
    basename = os.path.basename(dataset_path).split('.')[0]

    experiments = []
    # depths = range(1, 13)
    mdepths = [1, 2, 6, 10, 12]
    depths = [1, 2, 6, 10, 12]

    if 'mast3r' in basename:
        depths = [1]
        mdepths = [1]

    experiments.extend([f'3p_reldepth+{i}' for i in depths])
    experiments.extend([f'3p_ours_shift_scale+{i}' for i in depths])
    experiments.extend([f'3p_ours_shift_scale_reproj+{i}' for i in depths])
    experiments.extend([f'3p_ours_shift_scale_reproj-s+{i}' for i in depths])
    experiments.extend([f'mad_poselib_shift_scale+{i}' for i in depths])
    experiments.extend([f'mad_poselib_shift_scale_reproj+{i}' for i in depths])
    experiments.extend([f'mad_poselib_shift_scale_reproj-s+{i}' for i in depths])
    experiments.extend([f'p3p+{i}' for i in depths])
    experiments.extend([f'p3p_reproj+{i}' for i in depths])
    experiments.extend([f'p3p_reproj-s+{i}' for i in depths])
    if not args.nmad:
        experiments.extend([f'madpose+{i}' for i in mdepths])
    experiments.append('5p')

    if args.faster:
        experiments = []
        experiments.extend([f'3p_ours_shift_scale+{i}' for i in depths])
        experiments.extend([f'3p_ours_shift_scale_reproj+{i}' for i in depths])
        experiments.extend([f'3p_ours_shift_scale_reproj-s+{i}' for i in depths])
        experiments.extend([f'mad_poselib_shift_scale+{i}' for i in depths])
        experiments.extend([f'mad_poselib_shift_scale_reproj+{i}' for i in depths])
        experiments.extend([f'mad_poselib_shift_scale_reproj-s+{i}' for i in depths])

    if args.madours:
        experiments = []
        # experiments.extend([f'madpose+{i}' for i in mdepths])
        experiments.extend([f'madpose_ours_scale_shift+{i}' for i in mdepths])

    if args.madonly:
        experiments = []
        # experiments.extend([f'madpose+{i}' for i in mdepths])
        experiments.extend([f'madpose+{i}' for i in mdepths])

    if args.nlo:
        experiments = [f'nLO-{x}' for x in experiments]

    if args.graduated:
        experiments = [f'GLO-{x}' for x in experiments]

    if args.graph:
        experiments = []
        depths = [2, 6, 10, 11, 12]
        # experiments = [f'3p_monodepth+{i}' for i in depths]
        experiments = [f'3p_monodepth_p3p+{i}' for i in depths]
        # experiments.extend([f'3p_reldepth+{i}' for i in depths])
        # experiments.extend([f'p3p+{i}' for i in depths])
        # experiments.append('5p')

    if args.threshold != 1.0:
        basename = f'{basename}-{args.threshold}t'
        
    if args.reproj_threshold != 16.0:
        basename = f'{basename}-{args.reproj_threshold}r'

    if args.graph:
        basename = f'graph-{basename}'
        iterations_list = [10, 20, 50, 100, 200, 500, 1000]
    else:
        iterations_list = [args.iters]

    json_string = f'calibrated-{basename}.json'
    json_path = os.path.join('results', json_string)

    if args.load:
        print("Loading: ", json_string)
        with open(json_path, 'r') as f:
            results = json.load(f)

    else:
        H5_file = h5py.File(dataset_path)

        prelim_pairs = [x.split('corr_')[1] for x in H5_file.keys() if 'corr_' in x]

        pairs = [(pair.split('_o_')[0] + '_o', pair.split('_o_')[1]) for pair in prelim_pairs]

        if args.first is not None:
            pairs = pairs[:args.first]

        def gen_data():
            for img_name_1, img_name_2 in pairs:
                for experiment in experiments:
                    for iterations in iterations_list:

                        Rt = np.array(H5_file[f'pose_{img_name_1}_{img_name_2}'])
                        R_gt = Rt[:3, :3]
                        t_gt = Rt[:, 3]

                        K1 = np.array(H5_file[f'K_{img_name_1}'])
                        K2 = np.array(H5_file[f'K_{img_name_2}'])

                        data = np.array(H5_file[f'corr_{img_name_1}_{img_name_2}'])
                        kp1 = data[:, :2]
                        kp2 = data[:, 2:4]

                        if len(data) < 5:
                            continue

                        if '+' in experiment:
                            depth = int(experiment.split('+')[1])
                            d = data[:, depth_indices(depth)]
                            # d = data[:, 31] / data[:, 30]
                        else:
                            d = np.ones_like(kp1)

                        l = get_valid_depth_mask(d)
                        # l = ~np.logical_or(np.isinf(d[:, 0]), np.isinf(d[:, 1]))
                        d[l] = 1.0

                        yield iterations, experiment, np.copy(kp1), np.copy(kp2), \
                            np.copy(d), R_gt, t_gt, K1, K2, args.threshold, args.reproj_threshold

        total_length = len(experiments) * len(iterations_list) * len(pairs)

        print(f"Total runs: {total_length} for {len(pairs)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            pool = NoDaemonProcessPool(args.num_workers)
            results = [x for x in pool.imap(run_with_timeout, tqdm(gen_data(), total=total_length))]

        os.makedirs('results', exist_ok=True)

        if args.append:
            print(f"Appending from: {json_path}")
            try:
                with open(json_path, 'r') as f:
                    prev_results = json.load(f)
            except Exception:
                print("Prev results not found!")
                prev_results = []

            if args.overwrite:
                print("Overwriting old results")
                prev_results = [x for x in prev_results if not isinstance(x, tuple) and not isinstance(x, list)]
                prev_results = [x for x in prev_results if x['experiment'] not in experiments]

            results.extend(prev_results)

        with open(json_path, 'w') as f:
            json.dump(results, f)

        print("Done")

    print_results(experiments, results)
    draw_cumplots(experiments, results)

    if args.graph:
        draw_results_pose_auc_10(results, experiments, iterations_list)


if __name__ == '__main__':
    args = parse_args()
    eval(args)