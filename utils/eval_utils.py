import numpy as np


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
