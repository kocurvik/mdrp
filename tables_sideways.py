import json
import os
import subprocess

# from pylatex import Document, Section, Subsection, Command, Tabular, MultiColumn, MultiRow, NoEscape

import numpy as np

from utils.tables import depth_names, method_opts, method_names_calib, method_names_shared, method_names_varying, \
    init_latex, feature_names
from utils.data import basenames, R_err_fun, t_err_fun, err_fun_pose, \
    get_experiments, basenames_eth, basenames_pt, basenames_scannet
from utils.tables import get_median_errors, get_means

# depth_order = [1, 2, 3, 4, 5, 6, 7, 12, 8, 11, 9, 10]
depth_order = [1, 2, 6, 10, 12]

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
            text_rows[idxs[0]][j] = '\\textbf{' + text_rows[idxs[0]][j] + '}'
            text_rows[idxs[1]][j] = '\\underline{' + text_rows[idxs[1]][j] + '}'

    if master:
        cprint('\\multirow{', len(methods), '}{*}{Mast3r~\\cite{leroy2024grounding}}')
    else:
        cprint('\\multirow{', len(methods), '}{*}{', depth_names[depth], '}')
    for i, method in enumerate(methods):
        cprint(f'& {method_names[method]} & {method_opts(method)} & {"&".join(text_rows[i])} \\\\')


def generate_calib_table(cprint=print, prefix='', basenames=basenames, **kwargs):
    experiments = get_experiments('calib', master=False)
    # experiments = [x for x in experiments if 'reproj' not in x and 'mast3r' not in x]

    # monodepth_methods = ['3p_reldepth', 'p3p', 'mad_poselib_shift_scale', '3p_ours_shift_scale', 'madpose',
    #                      'madpose_ours_scale_shift']
    monodepth_methods = ['3p_reldepth', 'p3p', 'mad_poselib_shift_scale', '3p_ours_shift_scale',
                         'p3p_reproj', 'mad_poselib_shift_scale_reproj', '3p_ours_shift_scale_reproj',
                         'p3p_reproj-s', 'mad_poselib_shift_scale_reproj-s', '3p_ours_shift_scale_reproj-s',
                         'madpose', 'madpose_ours_scale_shift']

    baseline_methods = ['5p']

    experiments = [f'{prefix}{x}' for x in experiments]
    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

    means = get_all_means(experiments,  ['splg', 'roma'], basenames, kwargs)

    num_supercols = len(means)

    cprint('\\resizebox{\\linewidth}{!}{')
    column_alignment = 'clc' + 'c' * num_supercols * 3
    cprint('\\begin{tabular}{' + column_alignment + '}')
    cprint('\\toprule')
    header_row = ' \\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{{Solver}} & \\multirow{2.5}{*}{{Opt.}} '
    cmidrule_parts = []
    for i, part in enumerate(means.keys()):
        header_row += '& \\multicolumn{3}{c}{' + str(feature_names[part.split('-')[1]] ) + '}'
        cmidrule_parts.append('\\cmidrule(rl){' + str(3 * i + 4) + '-' + str(3 * i + 6) + '}')

    column_names = '& & & $\\epsilon(^\\circ)\\downarrow$ & mAA$\\uparrow$ & $\\tau (ms)\\downarrow$ '
    for i in range(num_supercols - 1):
        column_names += ' & $\\epsilon(^\\circ)\\downarrow$ & mAA $\\uparrow$ & $\\tau (ms)\\downarrow$ '

    cprint(header_row + '\\\\')
    cprint(' '.join(cmidrule_parts))
    cprint(column_names + '\\\\')
    cprint('\\midrule')

    print_monodepth_rows(0, baseline_methods, method_names_calib, means, cprint=cprint)
    cprint('\\hline')
    for i in depth_order:
        print_monodepth_rows(i, monodepth_methods, method_names_calib, means, cprint=cprint)
        cprint('\\hline')

    # experiments.append('mast3r+1')
    # monodepth_methods.append('mast3r')
    means_master = get_all_means(experiments, ['mast3r'], basenames, kwargs)
    cprint('\\\\')
    cprint('\\multicolumn{', str(3 + 3*num_supercols), '}{c}{\\begin{tabular}{clcccc}\\hline')

    cprint('\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{{Solver}} & \\multirow{2.5}{*}{{Opt.}} '
           '& \\multicolumn{3}{c}{Mast3r~\\cite{leroy2024grounding}} \\\\ \\cmidrule{4-6}')
    cprint('&&& $\\epsilon(^\\circ)\\downarrow$ & mAA $\\uparrow$ & $\\tau (ms)\\downarrow$ \\\\ \\cmidrule{1-6}')

    print_monodepth_rows(0, baseline_methods, method_names_calib, means_master, cprint=cprint, master=False)
    cprint('\\cmidrule{1-6}')
    print_monodepth_rows(1, monodepth_methods, method_names_calib, means_master, cprint=cprint, master=True)
    cprint('\\cmidrule{1-6}')
    cprint('\\end{tabular}}')
    cprint('\\end{tabular}}')


def get_all_means(experiments, features, basenames, kwargs):
    means = {}
    for feats in features:
        scene_errors = {}
        for scene_list in basenames.values():
            for scene in scene_list:
                print(f"Loading: {scene}")
                scene_errors[f"{scene}"] = get_median_errors(scene, experiments, features=feats, prefix='calibrated',
                                                             **kwargs)

        print("Calculating Means")
        means.update({f'{k}-{feats}': get_means(scene_errors, v, experiments) for k, v in basenames.items()})
    return means


def generate_shared_table(cprint=print, prefix='', master=False, **kwargs):
    # depths = range(1, 13)
    experiments = get_experiments('shared_focal', master=master)

    monodepth_methods = sorted(list(set([x.split('+')[0] for x in experiments]) - {'6p'}))
    baseline_methods = ['6p']

    experiments = [f'{prefix}{x}' for x in experiments]
    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

    scene_errors = {}
    for scene_list in basenames.values():
        for scene in scene_list:
            print(f"Loading: {scene}")
            scene_errors[scene] = get_median_errors(scene, experiments, calc_f_err=True, prefix='shared_focal',
                                                    **kwargs)

    print("Calculating Means")
    means = {k: get_means(scene_errors, v, experiments) for k, v in basenames.items()}

    cprint('\\resizebox{\\linewidth}{!}{')
    num_columns = 2 + 5 * len(basenames)
    # Generate the column alignment string for \begin{tabular}
    column_alignment = 'c' * num_columns
    # Create the base part of the table
    cprint('\\begin{tabular}{' + column_alignment + '}')
    # Start the table with a top rule
    cprint('\\toprule')
    # Initialize the first part of the header row dynamically
    header_row = '\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{Method} '
    cmidrule_parts = []
    for i, part in enumerate(basenames.keys()):
        header_row += '& \\multicolumn{5}{c}{' + str(part) + '}'
        cmidrule_parts.append('\\cmidrule(rl){' + str(5 * i + 3) + '-' + str(5 * i + 7) + '}')
    # Add the second row header for each part (the column names)
    column_names = '& & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ '
    for i in range(len(basenames) - 1):
        column_names += ' & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ '
    # Print the first and second header rows
    cprint(header_row + '\\\\')
    cprint(' '.join(cmidrule_parts))
    cprint(column_names + '\\\\')
    cprint('\\midrule')

    # table head
    # cprint('\\resizebox{\\linewidth}{!}{')
    # cprint('\\begin{tabular}{cccccccccccccccc}')
    # cprint('\\toprule')
    # cprint('\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{Method} & \\multicolumn{5}{c}{Phototourism} & \\multicolumn{5}{c}{ETH3D}  \\\\ \\cmidrule(rl){3-7} \\cmidrule(rl){8-12}')
    # cprint('\\cmidrule(rl){3-7} \\cmidrule(rl){8-12} & & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\  & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\ \\\\ \\midrule')

    print_monodepth_rows(0, baseline_methods, method_names_shared, means, use_focal=True,
                         cprint=cprint)
    if master:
        print_monodepth_rows(1, monodepth_methods, method_names_shared, means, use_focal=True,
                             cprint=cprint, master=True)
    else:
        for i in depth_order:
            print_monodepth_rows(i, monodepth_methods, method_names_shared, means, use_focal=True,
                                 cprint=cprint)
    cprint('\\end{tabular}}')


def generate_varying_table(prefix='', cprint=print, master=False, **kwargs):
    experiments = get_experiments('varying_focal', master=master)

    monodepth_methods = sorted(list(set([x.split('+')[0] for x in experiments]) - {'7p'}))
    baseline_methods = ['7p']

    experiments = [f'{prefix}{x}' for x in experiments]
    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

    cprint('\\resizebox{\\linewidth}{!}{')
    num_columns = 2 + 5 * len(basenames)
    # Generate the column alignment string for \begin{tabular}
    column_alignment = 'c' * num_columns
    # Create the base part of the table
    cprint('\\begin{tabular}{' + column_alignment + '}')
    # Start the table with a top rule
    cprint('\\toprule')
    # Initialize the first part of the header row dynamically
    header_row = '\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{Method} '
    cmidrule_parts = []
    for i, part in enumerate(basenames.keys()):
        header_row += '& \\multicolumn{5}{c}{' + str(part) + '}'
        cmidrule_parts.append('\\cmidrule(rl){' + str(5 * i + 3) + '-' + str(5 * i + 7) + '}')
    # Add the second row header for each part (the column names)
    column_names = '& & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ '
    for i in range(len(basenames) - 1):
        column_names += ' & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ '
    # Print the first and second header rows
    cprint(header_row + '\\\\')
    cprint(' '.join(cmidrule_parts))
    cprint(column_names + '\\\\')
    cprint('\\midrule')
    # table head
    # cprint('\\resizebox{\\linewidth}{!}{')
    # cprint('\\begin{tabular}{cccccccccccccccc}')
    # cprint('\\toprule')
    # cprint('\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{Method} & \\multicolumn{5}{c}{Phototourism} & \\multicolumn{5}{c}{ETH3D}  \\\\ \\cmidrule(rl){3-7} \\cmidrule(rl){8-12}')
    # cprint('\\cmidrule(rl){3-7} \\cmidrule(rl){8-12} & & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\  & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\ \\\\ \\midrule')

    scene_errors = {}
    for scene_list in basenames.values():
        for scene in scene_list:
            print(f"Loading: {scene}")
            scene_errors[scene] = get_median_errors(scene, experiments, calc_f_err=True, prefix='varying_focal',
                                                    **kwargs)

    print("Calculating Means")
    means = {k: get_means(scene_errors, v, experiments) for k, v in basenames.items()}

    print_monodepth_rows(0, baseline_methods, method_names_varying, means, use_focal=True,
                         cprint=cprint)
    if master:
        print_monodepth_rows(1, monodepth_methods, method_names_varying, means, use_focal=True, cprint=cprint,
                             master=True)
    else:
        for i in depth_order:
            print_monodepth_rows(i, monodepth_methods, method_names_varying, means, use_focal=True, cprint=cprint)
    cprint('\\end{tabular}}')


def typeset_latex(destination, cprint=print):
    command = 'pdflatex --output-directory=pdfs' if os.name == 'nt' else 'tectonic'
    try:
        cprint('')
        cprint('\\center')
        cprint(f'{destination.replace("_", "-")}')
        cprint('\\end{document}')

        subprocess.run([command, destination], check=True)
        print("PDF generated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during LaTeX compilation: {e}")


def type_table(table_func, prefix='', make_pdf=True, basenames=basenames, master=False, **kwargs):
    if make_pdf:
        if not os.path.exists('pdfs'):
            os.makedirs('pdfs', exist_ok=True)
        kwarg_string = '-'.join([x for x in kwargs.values() if len(x) > 0])
        kwarg_string = f'{kwarg_string}-{"-".join(basenames.keys())}'
        destination = f'pdfs/sideways-{prefix}{table_func.__name__}{kwarg_string}.tex'

        def cprint(*args):
            with open(destination, 'a') as file:
                print(*args, file=file)

        init_latex(destination)
        print(prefix, table_func.__name__)
        table_func(prefix=prefix, cprint=cprint, master=master, basenames=basenames, **kwargs)
        typeset_latex(destination, cprint=cprint)
    else:
        print(prefix, table_func.__name__)
        table_func(prefix=prefix, **kwargs)


if __name__ == '__main__':
    # just print normally
    # cprint = print

    # basenames.pop('ETH', None)
    # basenames.pop('Phototourism', None)
    # basenames.pop('ScanNet', None)
    type_table(generate_calib_table, basenames={'ETH':basenames_eth}, make_pdf=True, t='2.0t')
    type_table(generate_calib_table, basenames={'Phototourism': basenames_pt}, make_pdf=True, t='2.0t')
    type_table(generate_calib_table, basenames={'ScanNet': basenames_scannet}, make_pdf=True, t='2.0t')
    # type_table(generate_shared_table, make_pdf=True, t='2.0t', features='splg')
    # type_table(generate_varying_table, make_pdf=True, t='2.0t', features='splg')
    #
    # type_table(generate_calib_table, make_pdf=True, t='2.0t', features='roma')
    # type_table(generate_shared_table, make_pdf=True, t='2.0t', features='roma')
    # type_table(generate_varying_table, make_pdf=True, t='2.0t', features='roma')
    #
    # basenames.pop('ETH', None)
    # type_table(generate_varying_table, master=True, make_pdf=True, t='2.0t', features='mast3r')
    #
    basenames.pop('Phototourism', None)
    basenames['ETH'] = basenames_eth
    type_table(generate_shared_table, master=True, make_pdf=True, t='2.0t', features='mast3r')

    basenames.pop('Phototourism', None)
    basenames.pop('ETH', None)
    type_table(generate_calib_table, master=True, make_pdf=True, t='2.0t', features='mast3r_moge')
    type_table(generate_shared_table, master=True, make_pdf=True, t='2.0t', features='mast3r_moge')
    type_table(generate_varying_table, master=True, make_pdf=True, t='2.0t', features='mast3r_moge')

# print("GLO calib")
# generate_calib_table('GLO-')
# print("LO shared focal")
# generate_shared_table()
# print("GLO shared focal")
# generate_shared_table('GLO-')
# print("NN shared focal")
# generate_shared_table('NN-')

# print("LO varying focal")
# generate_varying_table()
# print("LO varying focal")
# generate_varying_table(lo=True)
# print("NN varying focal")
# generate_varying_table('NN-')