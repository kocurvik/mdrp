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

large_size = 18
small_size = 16

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
    def get_base_exp_name(x):
        return x.replace('-reproj-s', '-reproj').replace('-reproj', '')

    base_experiments = list(set([get_base_exp_name(x) for x in experiments]))
    print(base_experiments)
    # base_experiments = experiments
    base_colors = {exp: sns.color_palette("hls", len(base_experiments)).as_hex()[i] for i, exp in enumerate(base_experiments)}
    colors = {exp: base_colors[get_base_exp_name(exp)] for exp in experiments}

    print(colors)

    styles = {}
    for exp in experiments:
        if 'reproj' not in exp:
            styles[exp] = 'solid'
        elif 'reproj-s' in exp:
            styles[exp] = 'dotted'
        else:
            styles[exp] = 'dashed'
    return colors, styles

def get_colors_styles_fixed(results_type):
    c = sns.color_palette()
    if 'calib' in results_type:
        colors = {
            '5p': c[5],
            '3p_reldepth': c[3],
            'p3p': c[6],
            'mad_poselib_shift_scale': c[2],
            '3p_ours_shift_scale': c[1],
            'madpose': c[2],
            'madpose_ours_scale_shift': c[1],
            'mast3r': c[4]
        }

        styles = {
            '5p': 'solid',
            '3p_reldepth': 'solid',
            'p3p': 'solid',
            'mad_poselib_shift_scale': 'solid',
            '3p_ours_shift_scale': 'solid',
            'madpose': 'dashed',
            'madpose_ours_scale_shift': 'dashed',
            'mast3r': 'dotted'
        }
        return colors, styles

    if 'shared' in results_type:
        colors = {
            '3p_ours_scale': c[0],
            'madpose_ours_scale': c[0],
            '4p_ours_scale_shift': c[1],
            'mad_poselib_shift_scale': c[2],
            'madpose': c[2],
            '3p_reldepth': c[3],
            'mast3r': c[4],
            '6p': c[5],
        }

        styles = {
            '3p_ours_scale': 'solid',
            'madpose_ours_scale': 'dashed',
            '4p_ours_scale_shift': 'solid',
            'mad_poselib_shift_scale': 'solid',
            'madpose': 'dashed',
            '3p_reldepth': 'solid',
            'mast3r': 'dotted',
            '6p': 'solid',
        }
        return colors, styles

    if 'varying' in results_type:
        colors = {
            'madpose_ours_scale': c[0],
            '3p_ours_scale_ff_fs': c[0],
            '3p_ours_scale': c[0],
            '4p_ours_scale_shift': c[1],
            'mad_poselib_shift_scale': c[2],
            'madpose': c[2],
            '4p4d': c[3],
            '7p': c[5],
            'mast3r': c[4]
        }

        styles = {
            'madpose_ours_scale': 'dashed',
            '3p_ours_scale': 'solid',
            '3p_ours_scale_ff_fs': 'dotted',
            '4p_ours_scale_shift': 'solid',
            'mad_poselib_shift_scale': 'solid',
            'madpose': 'dashed',
            '4p4d': 'solid',
            '7p': 'solid',
            'mast3r': 'dotted'
        }
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
    fig = plt.figure(figsize=(12,6), frameon=True)

    colors, styles = get_colors_styles(experiments)

    aucs = {}
    rts = {}

    for experiment in tqdm(experiments):
        print(experiment)
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info'].get('iterations', None) == iterations]
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
        # plt.plot(xs, ys, label=experiment, marker='*', color=colors[experiment], linestyle=styles[experiment])

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

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(title)
        plt.savefig(f'figs/{title}_pose.png', bbox_inches='tight', pad_inches=0.1)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()

    return aucs, rts

def draw_results_focal_auc_10(results, experiments, iterations_list, title=None):
    fig = plt.figure(figsize=(12,6), frameon=True)

    colors, styles = get_colors_styles(experiments)

    aucs = {}
    rts = {}

    for experiment in tqdm(experiments):
        print(experiment)
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info'].get('iterations', None) == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([out['f_err'] for out in iter_results])
            # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
            errs[np.isnan(errs)] = 1.0
            AUC10 = np.mean(np.array([np.sum(errs < t / 100) / len(errs) for t in range(1, 11)]))
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
    plt.ylabel('AUC@0.1-$f$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        # plt.title(title)
        # plt.savefig(f'figs/{title}_pose.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'figs/{title}_focal.pdf')#, bbox_inches='tight', pad_inches=0)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(title)
        plt.savefig(f'figs/{title}_focal.png', bbox_inches='tight', pad_inches=0.1)
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


def generate_graphs(dataset, results_type, t='-2.0t', features='splg', depth=12, master=False, ylim=None, xlim=None):
    basenames = get_basenames(dataset)


    depths = [depth]
    experiments = get_experiments(results_type, depths=depths, nmad=True, graph=True)
    experiments = [x for x in experiments if 'reproj' not in x]


    colors, styles = get_colors_styles_fixed(results_type)
    # colors, styles = None, None

    if master:
        experiments.append('mast3r')

    iterations_list = [50, 100, 200, 500, 1000]

    xs = np.empty([len(basenames), len(experiments), len(iterations_list)])
    ys = np.empty([len(basenames), len(experiments), len(iterations_list)])
    fs = np.empty([len(basenames), len(experiments), len(iterations_list)])

    for b, basename in enumerate(basenames):
        if 'varying' in results_type:
            json_path = os.path.join('results_new', f'{results_type}-{basename}_{features}{t}-graph.json')
        else:
            json_path = os.path.join('results_new', f'{results_type}-graph-{basename}_{features}{t}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            results = [x for x in json.load(f) if x['experiment'] in experiments]

        if master:
            master_json_path = os.path.join('results_new', f'{results_type}-mast3rgraph-{basename}.json')
            with open(master_json_path, 'r') as f:
                # master_results = json.load(f)
                master_results = [x for x in json.load(f) if x['experiment'] == 'mast3r']

            results.extend(master_results)

        calc_maa(b, experiments, iterations_list, results, fs, xs, ys)

    draw_all(experiments, fs, xs, ys, title=f'{results_type}-{dataset}-{features}', colors=colors, styles=styles, ylim=ylim, xlim=xlim)



def calc_maa(b, experiments, iterations_list, results, fs, xs, ys):
    for i, experiment in enumerate(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        for j, iterations in enumerate(iterations_list):
            iter_results = [x for x in experiment_results if x['info'].get('iterations', None) == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([err_fun_pose(out) for out in iter_results])
            errs[np.isnan(errs)] = 180.0
            AUC10 = 100 * np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))

            try:
                ferrs = np.array([out['f_err'] for out in iter_results])
                ferrs[np.isnan(ferrs)] = 1.0
                fAUC10 = 100 * np.mean(np.array([np.sum(ferrs < t / 100) / len(errs) for t in range(1, 11)]))
            except Exception:
                fAUC10 = 0.0

            xs[b, i, j] = mean_runtime
            ys[b, i, j] = AUC10
            fs[b, i, j] = fAUC10


def generate_eth_roma():
    basenames = get_basenames('ETH')

    experiments = []
    experiments.append('3p_ours_scale+12')
    # experiments.append('3p_ours+12')
    experiments.append('3p_reldepth+12')
    experiments.append('mad_poselib_shift_scale+12')
    experiments.append('6p')

    slow_experiments = []
    slow_experiments.append('madpose_ours_scale+12')
    slow_experiments.append('madpose+12')
    slow_experiments.append('mast3r+1')

    all_experiments = experiments + slow_experiments

    iterations_list = [50, 100, 200, 500, 1000]

    xs = np.empty([len(basenames), len(all_experiments), len(iterations_list)])
    ys = np.empty([len(basenames), len(all_experiments), len(iterations_list)])
    fs = np.empty([len(basenames), len(all_experiments), len(iterations_list)])

    for b, basename in enumerate(basenames):
        all_results = []
        json_path = os.path.join('results_new', f'shared_focal-graph-{basename}_roma-2.0t.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            results = [x for x in json.load(f) if x['experiment'] in experiments or x['experiment'] in slow_experiments]
        all_results.extend(results)

        json_path = os.path.join('results_new', f'shared_focal-{basename}_roma-2.0t.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            results = [x for x in json.load(f) if x['experiment'] in ['madpose+12', 'madpose_ours_scale+12']]
        all_results.extend(results)

        json_path = os.path.join('results_new', f'shared_focal-{basename}_mast3r-2.0t.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            results = [x for x in json.load(f) if x['experiment'] == 'mast3r+1']
        all_results.extend(results)

        json_path = os.path.join('results_new', 'mast3r_extended', f'{basename}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            results = [x for x in json.load(f) if x['experiment'] == 'mast3r']
        for x in results:
            x['experiment'] = 'mast3r+1'

        all_results.extend(results)

        calc_maa(b, all_experiments, iterations_list, all_results, fs, xs, ys)

    colors, styles = get_colors_styles_fixed('shared')
    draw_all(all_experiments, fs, xs, ys, title='eth_shared_roma', colors=colors, styles=styles, ylim=[80, 88])

def draw_all(experiments, fs, xs, ys, title=None, colors=None, styles=None, ylim=None, ylimf=None, xlim=None):
    fig = plt.figure(figsize=(8, 6), frameon=True)
    for i, experiment in enumerate(experiments):
        if colors is not None:
            experiment = experiment.split('+')[0]
            plt.semilogx(np.mean(xs[:, i, :], axis=0), np.mean(ys[:, i, :], axis=0), label=experiment, marker='*', color=colors[experiment], linestyle=styles[experiment])
        else:
            plt.semilogx(np.mean(xs[:, i, :], axis=0), np.mean(ys[:, i, :], axis=0), label=experiment, marker='*')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xlabel('$\\tau$ (ms)', fontsize=large_size)
    plt.ylabel('mAA', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        plt.savefig(f'figs/{title}_pose.pdf')  # , bbox_inches='tight', pad_inches=0)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(title)
        plt.savefig(f'figs/{title}_pose.png', bbox_inches='tight', pad_inches=0.0)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()

    fig = plt.figure(figsize=(8, 6), frameon=True)
    for i, experiment in enumerate(experiments):
        plt.semilogx(np.mean(xs[:, i, :], axis=0), np.mean(fs[:, i, :], axis=0), label=experiment, marker='*')

    # plt.xlim([15, 330])
    if ylimf is not None:
        plt.ylim(ylimf)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xlabel('$\\tau$ (ms)', fontsize=large_size)
    plt.ylabel('mAA$_f$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        plt.savefig(f'figs/{title}_focal.pdf')  # , bbox_inches='tight', pad_inches=0)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(title)
        plt.savefig(f'figs/{title}_focal.png', bbox_inches='tight', pad_inches=0.0)
        print(f'saved pose: {title}')
    else:
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # generate_eth_roma()
    
    for features in ['splg']:
    # for features in ['mast3r_moge']:
        # for depth in [1, 2, 6, 10, 12]:
        for depth in [10]:
            ...
            # generate_graphs('ScanNet', 'calibrated', features=features, depth=depth)
            # generate_graphs('ETH', 'calibrated', features=features, depth=depth)
            # generate_graphs('Phototourism', 'calibrated', features=features, depth=depth)
        
            # generate_graphs('ScanNet', 'shared_focal', features=features, depth=depth, master=True)
            # generate_graphs('ETH', 'shared_focal', features=features, depth=depth)
        
            # generate_graphs('ScanNet', 'varying_focal', features=features, depth=depth, master=False)
            generate_graphs('Phototourism', 'varying_focal', features=features, depth=depth, xlim=[9.5, 108])

