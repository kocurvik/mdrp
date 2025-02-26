import multiprocessing.pool

import numpy as np

from utils.data import R_err_fun, t_err_fun
from prettytable import PrettyTable

def print_results_focal(experiments, results, eq_only=False):
    tab = PrettyTable(['solver', 'median pose err', 'median f err',
                       'pose mAA', 'f mAA', 'mean time', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        f_errs = np.array([r['f_err'] for r in exp_results])

        p_errs[np.isnan(p_errs)] = 180
        f_errs[np.isnan(f_errs)] = 1.0

        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 11)])
        f_res = np.array([np.sum(f_errs < t/100) / len(f_errs) for t in range(1, 11)])

        times = np.array([x['info']['runtime'] for x in exp_results])
        inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

        exp_name = exp


        tab.add_row([exp_name, np.median(p_errs), np.median(f_errs),
                     np.mean(p_res), np.mean(f_res),
                     np.mean(times),
                     np.mean(inliers)])
    print(tab)
    # print('latex')
    # print(tab.get_formatted_string('latex'))


def print_results(experiments, results, eq_only=False):
    tab = PrettyTable(['solver', 'median pose err', 'pose mAA', 'mean time', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        p_errs[np.isnan(p_errs)] = 180

        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 11)])

        times = np.array([x['info']['runtime'] for x in exp_results])
        inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

        exp_name = exp


        tab.add_row([exp_name, np.median(p_errs), np.mean(p_res),
                     np.mean(times),
                     np.mean(inliers)])
    print(tab)

    print('latex')

    print(tab.get_formatted_string('latex'))


def get_pairs(file):
    with open(file, 'r') as f:
        pairs = f.readlines()
    return [tuple(x.strip().split(' ')) for x in pairs]


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc


def get_exception_result_dict(x):
    iters, experiment, kp1, kp2, d, R_gt, t_gt, K1, K2, t, r = x
    out = {}

    R_est, t_est = np.eye(3), np.ones(3)

    f1_gt = (K1[0, 0] + K1[1, 1]) / 2
    f2_gt = (K2[0, 0] + K2[1, 1]) / 2
    out['f1_gt'] = f1_gt
    out['f1'] = 1.0
    out['f2_gt'] = f2_gt
    out['f2'] = 1.0

    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = t_est.tolist()
    out['t_gt'] = t_gt.tolist()

    out['R_err'] = R_err_fun(out)
    out['t_err'] = t_err_fun(out)

    out['f1_err'] = np.abs(out['f1'] - f1_gt) / f1_gt
    out['f2_err'] = np.abs(out['f2'] - f2_gt) / f2_gt
    out['f_err'] = np.sqrt(out['f1_err'] * out['f2_err'])

    info = {}

    info['num_inliers'] = 0
    info['inlier_ratio'] = 0.0
    info['runtime'] = 20000
    out['info'] = info
    out['experiment'] = experiment

    return out
