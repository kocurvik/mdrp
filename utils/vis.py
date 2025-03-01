import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import rc
import seaborn as sns
from tqdm import tqdm

from utils.data import get_basenames, err_fun_pose, get_experiments
from utils.geometry import rotation_angle

large_size = 24
small_size = 20

plt.rcParams.update({'figure.autolayout': True})

# plt.rcParams.update({'figure.autolayout': True})
# rc('font',**{'family':'serif','serif':['Times New Roman']})
# rc('font',**{'family':'serif'})
# rc('text', usetex=True)

# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Times New Roman'
# plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
# plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def get_colors_styles(experiments):
    # base_experiments = list(sorted(list(set([x.split(' ')[0] for x in experiments]))))
    # base_experiments = experiments
    # colors = {exp: sns.color_palette().as_hex()[i] for i, exp in enumerate(base_experiments)}
    colors = {exp: sns.color_palette("hls", len(experiments)).as_hex()[i] for i, exp in enumerate(experiments)}

    print(colors)

    styles = {}
    for exp in experiments:
        # if len(exp.split(' ')) == 1:
        #     styles[exp] = 'solid'
        # else:
        #     suffix = ' '.join(exp.split(' ')[1:])
        #     if 'ENM' in suffix:
        #         styles[exp] = 'dotted'
        #     else:
        #         styles[exp] = 'solid'
        # styles[exp] = 'dotted' if 'ENM' in exp else 'solid'
        styles[exp] = 'solid'

    return colors, styles

def draw_results(results, experiments, iterations_list, title=''):
    plt.figure()

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            # errs = np.array([max(1/3 *(out['R_12_err'] + out['R_13_err'] + out['R_23_err']), 1/3 * (out['t_12_err'] + out['t_13_err'] + out['t_23_err'])) for out in iter_results])
            errs = np.array([max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err'])) for out in iter_results])
            # errs = np.array([r['P_err'] for r in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))

            xs.append(mean_runtime)
            ys.append(AUC10)

        plt.semilogx(xs, ys, label=experiment, marker='*')

    title += f"Error: max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err']))"

    plt.title(title, fontsize=8)


    plt.xlabel('Mean runtime (ms)')
    plt.ylabel('AUC@10$\\deg$')
    plt.legend()
    plt.show()


def draw_results_pose_auc_10(results, experiments, iterations_list, err_fun=err_fun_pose, title=None):
    fig = plt.figure(frameon=True)

    colors, styles = get_colors_styles(experiments)

    aucs = {}
    rts = {}

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([err_fun(out) for out in iter_results])
            # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))
            # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

            xs.append(mean_runtime)
            ys.append(AUC10)

        aucs[experiment] = np.array(ys)
        rts[experiment] = np.array(xs)


        # colors = {exp: sns.color_palette("hls", len(experiments))[i] for i, exp in enumerate(experiments)}

        plt.semilogx(xs, ys, label=experiment, marker='*', color=colors[experiment], linestyle=styles[experiment])

    # plt.xlim(xlim)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('AUC@10$^\\circ$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        # plt.title(title)
        # plt.savefig(f'figs/{title}_pose.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'figs/{title}_pose.pdf')#, bbox_inches='tight', pad_inches=0)

        plt.legend()
        plt.savefig(f'figs/{title}_pose.png', bbox_inches='tight', pad_inches=0.1)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()

    return aucs, rts

def draw_results_pose_auc_10_mm(aucs, rts, experiments, title=None):
    fig = plt.figure(frameon=True)

    colors, styles = get_colors_styles(experiments)
    basenames = list(aucs.keys())

    for experiment in tqdm(experiments):
        xss = np.array([rts[b][experiment] for b in basenames])
        yss = np.array([aucs[b][experiment] for b in basenames])

        xs = np.mean(xss, axis=0)
        ys = np.mean(yss, axis=0)
        plt.semilogx(xs, ys, label=experiment, marker='*', color=colors[experiment], linestyle=styles[experiment])

    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('AUC@10$^\\circ$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        # plt.title(title)
        # plt.savefig(f'figs/{title}_pose.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'figs/{title}_pose.pdf')#, bbox_inches='tight', pad_inches=0)

        plt.legend()
        plt.savefig(f'figs/{title}_pose.png', bbox_inches='tight', pad_inches=0.1)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()

    return aucs, rts


def draw_results_pose_portion(results, experiments, iterations_list, title=None):
    plt.figure(frameon=False)

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        iter_results = experiment_results
        mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
        # errs = np.array([max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err'])) for out in iter_results])
        errs = np.array([out['t_13_err'] for out in iter_results])
        # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
        errs[np.isnan(errs)] = 180
        cum_err = np.array([np.sum(errs < t) / len(errs) for t in range(1, 181)])

        # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

        xs = np.arange(1, 181)
        ys = cum_err

        # plt.plot(xs, ys, label=experiment, marker='*', color=colors[experiment])
        plt.plot(xs, ys, label=experiment, marker='*')

    # plt.xlim([5.0, 1.9e4])
    plt.xlabel('Pose error', fontsize=large_size)
    plt.ylabel('Portion of samples', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        plt.savefig(f'figs/{title}_cumpose.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()


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

def draw_rotation_angle_f_err(experiments, results):
    plt.figure()
    plt.xlabel('Pose error')
    plt.ylabel('median f err')

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]
        exp_name = exp
        label = f'{exp_name}'

        lims = np.linspace(0, 60, 61)
        y_vals = []
        for i in range(len(lims) - 1):
            R_angles = np.array([r['f_err'] for r in exp_results if lims[i] < rotation_angle(r['R_gt']) < lims[i + 1]])
            y_vals.append(R_angles)

        plt.boxplot(y_vals, labels=lims[:60])
        plt.ylim([0, 1])
        plt.title(exp)


        plt.show()



    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('k error')
    plt.ylabel('Portion of samples')

def generate_graphs(dataset, results_type, all=True, basenames = None, prefix='', t='-2.0t',
                    features='splg', ylim=None, colors=None):
    if basenames is None:
        basenames = get_basenames(dataset)

    depths = [12]
    experiments = get_experiments(results_type, depths=depths)

    iterations_list = [10, 20, 50, 100, 200, 500, 1000]

    all_results = []
    aucs = {}
    rts = {}
    for basename in basenames:
        json_path = os.path.join('results', f'{results_type}-graph-{basename}_{features}{t}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            results = [x for x in json.load(f) if x['experiment'] in experiments]
            # aucs[basename], rts[basename] = draw_results_pose_auc_10(results, experiments, iterations_list,
            #                                           title=f'{prefix}{dataset}_{basename}_{results_type}',
            #                                           err_fun=t_err_fun)
            # draw_results_pose_auc_10(results, experiments, iterations_list,
            #                          f'maxerr_{dataset}_{basename}_{results_type}', err_fun=err_fun_max)
            if all:
               all_results.extend(results)

    if all:
        title = f'{dataset}_{results_type}-{features}'
        # draw_results_pose_auc_10_mm(aucs, rts, experiments, title=prefix + title)
        draw_results_pose_auc_10(all_results, experiments, iterations_list, title=prefix + title)

if __name__ == '__main__':
    generate_graphs('ScanNet', 'calibrated', all=True)
    generate_graphs('ETH', 'calibrated', all=True)
    generate_graphs('Phototourism', 'calibrated', all=True)

    generate_graphs('ScanNet', 'shared_focal', all=True)
    generate_graphs('ETH', 'shared_focal', all=True)

    generate_graphs('ScanNet', 'varying_focal', all=True)
    generate_graphs('Phototourism', 'varying_focal', all=True)

