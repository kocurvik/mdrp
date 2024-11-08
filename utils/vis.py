import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import rc
import seaborn as sns
from tqdm import tqdm

experiments = ['4p(HC)', '5p3v', '4p3v(M) + R + C', '4p3v(M-D) + R + C']

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
    base_experiments = experiments
    colors = {exp: sns.color_palette().as_hex()[i] for i, exp in enumerate(base_experiments)}
    # base_colors = {exp: sns.color_palette("hsl", len(base_experiments)).as_hex()[i] for i, exp in enumerate(base_experiments)}

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


def draw_results_pose_auc_10(results, experiments, iterations_list, title=None):
    fig = plt.figure(frameon=True)

    colors, styles = get_colors_styles(experiments)

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([out['P_err'] for out in iter_results])
            # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))
            # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

            xs.append(mean_runtime)
            ys.append(AUC10)


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

        plt.savefig(f'figs/{title}_pose.png', bbox_inches='tight', pad_inches=0.1)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()


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