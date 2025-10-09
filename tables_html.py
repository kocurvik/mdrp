import json
import os
import subprocess

# from pylatex import Document, Section, Subsection, Command, Tabular, MultiColumn, MultiRow, NoEscape

import numpy as np

from utils.tables import method_opts, feature_names
from utils.data import basenames, R_err_fun, t_err_fun, err_fun_pose, \
    get_experiments, basenames_eth, basenames_pt, basenames_scannet
from utils.tables import get_median_errors, get_means

# depth_order = [1, 2, 3, 4, 5, 6, 7, 12, 8, 11, 9, 10]
depth_order = [10, 12]


depth_names = {10: 'MoGe', 12: 'UniDepth', 0: '-'}



def print_monodepth_rows(depth, methods, method_names, means, use_focal=False, cprint=print, master=False):
    num_rows = []
    for method in methods:
        if depth > 0:
            method = f'{method}+{depth}'
        if use_focal:
            metrics = ['median_pose_err', 'median_f_err', 'mAA_pose', 'mAA_f',
                       'mean_runtime']
            incdec = [1, 1, -1, -1, 1]
        else:
            metrics = ['median_pose_err', 'mAA_pose', 'mean_runtime']
            incdec = [1, -1, 1]

        incdec *= len(means)

        vals = []
        for dataset in means.keys():
            vals.extend([means[dataset][method][x] for x in metrics])

        num_rows.append(vals)

    text_rows = [[f'{x:0.2f}' for x in y] for y in num_rows]
    arr = np.array(num_rows)
    if depth > 0:
        for j in range(len(text_rows[0])):
            idxs = np.argsort(incdec[j] * arr[:, j])

            best_text_row = text_rows[idxs[0]][j]
            k = 0
            while text_rows[idxs[k]][j] == best_text_row:
                text_rows[idxs[k]][j] = '<strong>' + text_rows[idxs[k]][j] + '</strong>'
                k += 1

            # second_best_text_row = text_rows[idxs[k]][j]
            # while text_rows[idxs[k]][j] == second_best_text_row:
            #     text_rows[idxs[k]][j] = '\\underline{' + text_rows[idxs[k]][j] + '}'
            #     k += 1

    if master:
        cprint('<tr>')
        cprint(f'<td rowspan="{len(methods)}" style="vertical-align : middle;text-align:center;">MASt3R</td>')
    else:
        cprint(f'<td rowspan="{len(methods)}" style="vertical-align : middle;text-align:center;">{depth_names[depth]}</td>')
    for i, method in enumerate(methods):
        if i != 0:
            cprint('<tr>')
        cprint(f'{method_names[method]} {"".join([f"<td>{x}</td>" for x in text_rows[i]])}')
        cprint('</tr>')


def get_all_means(experiments, features, basenames, prefix='calibrated', calc_f_err=False, **kwargs):
    means = {}
    for feats in features:
        scene_errors = {}
        for scene_list in basenames.values():
            for scene in scene_list:
                print(f"Loading: {scene}")
                scene_errors[f"{scene}"] = get_median_errors(scene, experiments, features=feats, prefix=prefix,
                                                             calc_f_err=calc_f_err, **kwargs)

        print("Calculating Means")
        means.update({f'{k}-{feats}': get_means(scene_errors, v, experiments) for k, v in basenames.items()})
    return means


def generate_calib_table(cprint=print, prefix='', basenames=basenames, **kwargs):
    monodepth_methods = ['3p_reldepth',
                         'p3p',
                         'madpose',
                         '3p_ours_shift_scale_hybrid-s_ctruncated',
                         'p3p_hybrid_ctruncated']

    baseline_methods = ['5p']

    method_names = {'3p_reldepth' : '<td>3P-RelDepth</td><td></td><td></td>',
                    'p3p' : '<td>P3P</td><td></td><td></td>',
                    '3p_ours_shift_scale_hybrid-s_ctruncated': '<td>Ours</td><td>✔</td><td>✔</td>',
                    'p3p_hybrid_ctruncated': '<td>Ours</td><td>✔</td><td></td>',
                    'madpose': '<td>MADPose</td><td>✔</td><td>✔</td>',
                    '5p': '<td>5-Point</td><td></td><td></td>'}

    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

    experiments = []
    experiments.extend(baseline_methods)
    experiments.extend([f'{x}+10' for x in monodepth_methods])
    experiments.extend([f'{x}+12' for x in monodepth_methods])

    means = get_all_means(experiments, ['splg', 'roma'], basenames, **kwargs)

    cprint('<table>')
    cprint('<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="3" align="center">SP+LG</td><td align="center" colspan="3">RoMA</td></tr>')
    cprint('<tr><td>$\\epsilon(^\\circ)\\downarrow$</td><td>mAA $\\uparrow$</td><td>Runtime (ms)</td><td>$\\epsilon(^\\circ)\\downarrow$</td><td>mAA $\\uparrow$</td><td>Runtime (ms)</td></tr>')

    print_monodepth_rows(0, baseline_methods, method_names, means, cprint=cprint)
    for i in depth_order:
        print_monodepth_rows(i, monodepth_methods, method_names, means, cprint=cprint)
    cprint('</table>')

    # experiments.append('mast3r+1')
    # monodepth_methods.append('mast3r')

    experiments = []
    experiments.extend(baseline_methods)
    experiments.extend([f'{x}+1' for x in monodepth_methods])

    means_master = get_all_means(experiments, ['mast3r'], basenames, **kwargs)

    cprint('<table>')
    cprint('<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="3" align="center">MASt3R</td>')
    cprint(
        '<tr><td>$\\epsilon(^\\circ)\\downarrow$</td><td>mAA $\\uparrow$</td><td>Runtime (ms)</td></tr>')
    print_monodepth_rows(0, baseline_methods, method_names, means_master, cprint=cprint, master=False)
    print_monodepth_rows(1, monodepth_methods, method_names, means_master, cprint=cprint, master=True)
    cprint('</table>')

def generate_shared_table(cprint=print, prefix='', basenames=basenames, master=False, **kwargs):
    monodepth_methods = ['3p_reldepth',
                         'madpose',
                         '3p_ours_scale_hybrid_ctruncated']

    baseline_methods = ['6p']

    method_names = {'3p_reldepth': '<td>3P3D</td><td></td><td></td>',
                    '3p_ours_scale_hybrid_ctruncated': '<td>Ours</td><td>✔</td><td></td>',
                    'madpose': '<td>MADPose</td><td>✔</td><td>✔</td>',
                    '6p': '<td>6-Point</td><td></td><td></td>',
                    'mast3r': '<td>MASt3R Opt.</td><td></td><td></td>'}

    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

    experiments = []
    experiments.extend(baseline_methods)
    experiments.extend([f'{x}+10' for x in monodepth_methods])
    experiments.extend([f'{x}+12' for x in monodepth_methods])

    means = get_all_means(experiments, ['splg', 'roma'], basenames, prefix='shared_focal', calc_f_err=True, **kwargs)

    cprint('<table>')
    cprint(
        '<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">SP+LG</td><td align="center" colspan="5">RoMA</td></tr>')
    cprint(
        '<tr><td>$\\epsilon(^\\circ)\\downarrow$</td><td>$\\xi\\downarrow$</td><td>mAA $\\uparrow$</td><td>mAA$_f \\uparrow$</td><td>Runtime (ms)</td><td>$\\epsilon(^\\circ)\\downarrow$</td><td>$\\xi\\downarrow$</td><td>mAA $\\uparrow$</td><td>mAA$_f \\uparrow$</td><td>Runtime (ms)</td></tr>')

    print_monodepth_rows(0, baseline_methods, method_names, means, use_focal=True, cprint=cprint)
    for i in depth_order:
        print_monodepth_rows(i, monodepth_methods, method_names, means, use_focal=True, cprint=cprint)
    cprint('</table>')

    # experiments.append('mast3r+1')
    monodepth_methods.append('mast3r')

    experiments = []
    experiments.extend(baseline_methods)
    experiments.extend([f'{x}+1' for x in monodepth_methods])

    means_master = get_all_means(experiments, ['mast3r'], basenames, prefix='shared_focal', calc_f_err=True, **kwargs)

    cprint('<table>')
    cprint(
        '<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">MASt3R</td>')
    cprint(
        '<tr><td>$\\epsilon(^\\circ)\\downarrow$</td><td>$\\xi\\downarrow$</td><td>mAA $\\uparrow$</td><td>mAA$_f \\uparrow$</td><td>Runtime (ms)</td></tr>')
    print_monodepth_rows(0, baseline_methods, method_names, means_master, cprint=cprint, master=False)
    print_monodepth_rows(1, monodepth_methods, method_names, means_master, cprint=cprint, master=True)
    cprint('</table>')


def generate_varying_table(prefix='', cprint=print, basenames=basenames, master=False, **kwargs):
    monodepth_methods = ['4p4d',
                         'madpose',
                         '3p_ours_scale_hybrid_ctruncated']

    baseline_methods = ['7p']

    method_names = {'4p4d': '<td>4P4D</td><td></td><td></td>',
                    '3p_ours_scale_hybrid_ctruncated': '<td>Ours</td><td>✔</td><td></td>',
                    'madpose': '<td>MADPose</td><td>✔</td><td>✔</td>',
                    '7p': '<td>7-Point</td><td></td><td></td>',
                    'mast3r': '<td>MASt3R Opt.</td><td></td><td></td>'}

    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

    experiments = []
    experiments.extend(baseline_methods)
    experiments.extend([f'{x}+10' for x in monodepth_methods])
    experiments.extend([f'{x}+12' for x in monodepth_methods])

    means = get_all_means(experiments, ['splg', 'roma'], basenames, prefix='varying_focal', calc_f_err=True, **kwargs)

    cprint('<table>')
    cprint(
        '<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">SP+LG</td><td align="center" colspan="5">RoMA</td></tr>')
    cprint(
        '<tr><td>$\\epsilon(^\\circ)\\downarrow$</td><td>$\\xi\\downarrow$</td><td>mAA $\\uparrow$</td><td>mAA$_f \\uparrow$</td><td>Runtime (ms)</td><td>$\\epsilon(^\\circ)\\downarrow$</td><td>$\\xi\\downarrow$</td><td>mAA $\\uparrow$</td><td>mAA$_f \\uparrow$</td><td>Runtime (ms)</td></tr>')

    print_monodepth_rows(0, baseline_methods, method_names, means, cprint=cprint, use_focal=True)
    for i in depth_order:
        print_monodepth_rows(i, monodepth_methods, method_names, means, cprint=cprint, use_focal=True)
    cprint('</table>')

    # experiments.append('mast3r+1')
    monodepth_methods.append('mast3r')

    experiments = []
    experiments.extend(baseline_methods)
    experiments.extend([f'{x}+1' for x in monodepth_methods])

    means_master = get_all_means(experiments, ['mast3r'], basenames, prefix='varying_focal', calc_f_err=True, **kwargs)

    cprint('<table>')
    cprint(
        '<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">MASt3R</td>')
    cprint(
        '<tr><td>$\\epsilon(^\\circ)\\downarrow$</td><td>$\\xi\\downarrow$</td><td>mAA $\\uparrow$</td><td>mAA$_f \\uparrow$</td><td>Runtime (ms)</td></tr>')
    print_monodepth_rows(0, baseline_methods, method_names, means_master, use_focal=True, cprint=cprint, master=False)
    print_monodepth_rows(1, monodepth_methods, method_names, means_master, use_focal=True, cprint=cprint, master=True)
    cprint('</table>')


def init_html(title, destination, cprint=print):
    with open(destination, 'w') as file:
        file.write('<title>')
    cprint(title)
    cprint('</title>')
    cprint('<body>')
    cprint('<html>')

def close_html(cprint=print):
    cprint('</body>')
    cprint('</html>')

def type_table(table_func, prefix='', make_pdf=True, basenames=basenames, master=False, **kwargs):
    if not os.path.exists('html_tables'):
        os.makedirs('html_tables', exist_ok=True)
    kwarg_string = '-'.join([x for x in kwargs.values() if len(x) > 0])
    kwarg_string = f'{kwarg_string}-{"-".join(basenames.keys())}'
    destination = f'html_tables/sideways-{prefix}{table_func.__name__}{kwarg_string}.html'
    def cprint(*args):
        with open(destination, 'a', encoding="utf-8") as file:
            print(*args, file=file)

    init_html(f'{prefix}{table_func.__name__}{kwarg_string}', destination,cprint=cprint)
    print(prefix, table_func.__name__)
    table_func(prefix=prefix, cprint=cprint, master=master, basenames=basenames, **kwargs)
    close_html(cprint=cprint)



if __name__ == '__main__':
    # just print normally
    # cprint = print

    # Generate tables in SM
    type_table(generate_calib_table, basenames={'ScanNet': basenames_scannet}, make_pdf=True, t='2.0t')
    type_table(generate_shared_table, basenames={'ScanNet': basenames_scannet}, make_pdf=True, t='2.0t')
    type_table(generate_varying_table, basenames={'ScanNet': basenames_scannet}, make_pdf=True, t='2.0t')

    type_table(generate_calib_table, basenames={'ETH': basenames_eth}, make_pdf=True, t='2.0t')
    type_table(generate_shared_table, basenames={'ETH': basenames_eth}, make_pdf=True, t='2.0t')
    type_table(generate_varying_table, basenames={'ETH': basenames_eth}, make_pdf=True, t='2.0t')

    type_table(generate_calib_table, basenames={'Phototourism': basenames_pt}, make_pdf=True, t='2.0t')
    # type_table(generate_shared_table, basenames={'Phototourism': basenames_pt}, make_pdf=True, t='2.0t')
    type_table(generate_varying_table, basenames={'Phototourism': basenames_pt}, make_pdf=True, t='2.0t')

