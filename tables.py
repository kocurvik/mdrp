import json
import os
import subprocess

# from pylatex import Document, Section, Subsection, Command, Tabular, MultiColumn, MultiRow, NoEscape

import numpy as np

from utils.data import basenames, R_err_fun, t_err_fun, err_fun_pose, \
    get_experiments, basenames_eth
from utils.tables import get_median_errors, get_means, method_names_calib, depth_names, method_opts, feature_names, \
    depth_order, method_names_shared, init_latex, method_names_varying


# method_names_calib.update({f'nLO-{k}': v for k, v in method_names_calib.items()})
# method_names_calib.update({f'GLO-{k}': v for k, v in method_names_calib.items()})
# method_names_calib.update({f'NN-{k}': v for k, v in method_names_calib.items()})
# method_names_calib = {}

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

        incdec *= len(basenames)

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
        cprint('& \\multirow{', len(methods), '}{*}{Mast3r~\\cite{leroy2024grounding}}')
    else:
        cprint('& \\multirow{', len(methods), '}{*}{', depth_names[depth], '}')
    for i, method in enumerate(methods):
        if i != 0:
            cprint('&')
        cprint(f'& {method_names[method]} & {method_opts(method)} & {"&".join(text_rows[i])} \\\\')

def generate_calib_table(cprint=print, prefix='', master=False, **kwargs):
    cprint('\\resizebox{\\linewidth}{!}{')
    num_columns = 4 + 3 * len(basenames)
    column_alignment = 'cclc' + 'c' * len(basenames) * 3
    cprint('\\begin{tabular}{' + column_alignment + '}')
    cprint('\\toprule')
    header_row = '\\multirow{2.5}{*}{{Matches}} & \\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{{Solver}} & \\multirow{2.5}{*}{{Opt}} '
    cmidrule_parts = []
    for i, part in enumerate(basenames.keys()):
        header_row += '& \\multicolumn{3}{c}{' + str(part) + '}'
        cmidrule_parts.append('\\cmidrule(rl){' + str(3 * i + 5) + '-' + str(3 * i + 7) + '}')
    column_names = '& & & & $\\epsilon(^\\circ)\\downarrow$ & mAA$\\uparrow$ & $\\tau (ms)\\downarrow$ '
    for i in range(len(basenames) - 1):
        column_names += ' & $\\epsilon(^\\circ)\\downarrow$ & mAA $\\uparrow$ & $\\tau (ms)\\downarrow$ '
    cprint(header_row + '\\\\')
    cprint(' '.join(cmidrule_parts))
    cprint(column_names + '\\\\')
    cprint('\\midrule')

    for features in ['splg', 'roma', 'mast3r']:
        master = 'mast3r' == features
        experiments = get_experiments('calib', master=master)
        experiments = [x for x in experiments if 'reproj' not in x and 'mast3r' not in x]

        # monodepth_methods = sorted(list(set([x.split('+')[0] for x in experiments]) - {'5p'}))
        monodepth_methods = list(method_names_calib.keys())[1:-1]
        baseline_methods = ['5p']

        experiments = [f'{prefix}{x}' for x in experiments]
        monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
        baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

        scene_errors = {}
        for scene_list in basenames.values():
            for scene in scene_list:
                print(f"Loading: {scene}")
                scene_errors[scene] = get_median_errors(scene, experiments, features=features, prefix='calibrated', **kwargs)

        print("Calculating Means")
        means = {k: get_means(scene_errors, v, experiments) for k, v in basenames.items()}

        if master:
            total_rows = len(baseline_methods) + len(monodepth_methods)
        else:
            total_rows = len(baseline_methods) + len(depth_order) * len(monodepth_methods)

        cprint('\\multirow{', str(total_rows), '}{*}{', feature_names[features], '}')

        print_monodepth_rows(0, baseline_methods, method_names_calib, means, cprint=cprint)
        cprint('\\cmidrule(rl){2-', str(len(basenames) * 3 + 4), '}')

        if master:
            print_monodepth_rows(1, monodepth_methods, method_names_calib, means, cprint=cprint, master=True)
        else:
            for i in depth_order:
                print_monodepth_rows(i, monodepth_methods, method_names_calib, means, cprint=cprint)
                if i!= depth_order[-1]:
                    cprint('\\cmidrule(rl){2-', str(len(basenames) * 3 + 4), '}')
        cprint('\\hline')
    cprint('\\end{tabular}}')


def generate_shared_table(cprint=print, prefix='', master=False, **kwargs):
    cprint('\\resizebox{\\linewidth}{!}{')
    num_columns = 4 + 5 * len(basenames)
    column_alignment = 'cclc' + 'c' * len(basenames) * 5
    cprint('\\begin{tabular}{' + column_alignment + '}')
    cprint('\\toprule')
    header_row = '\\multirow{2.5}{*}{{Matches}} & \\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{{Solver}} & \\multirow{2.5}{*}{{Opt}} '
    cmidrule_parts = []
    for i, part in enumerate(basenames.keys()):
        header_row += '& \\multicolumn{5}{c}{' + str(part) + '}'
        cmidrule_parts.append('\\cmidrule(rl){' + str(5 * i + 5) + '-' + str(5 * i + 9) + '}')
    column_names = '& & & & $\\epsilon(^\\circ)\\downarrow$ & $\\epsilon_f(^\\circ)\\downarrow$ & mAA$\\uparrow$ & mAA$_f\\uparrow$ & $\\tau (ms)\\downarrow$ '
    for i in range(len(basenames) - 1):
        column_names += ' & $\\epsilon(^\\circ)\\downarrow$ & $\\epsilon_f(^\\circ)\\downarrow$ & mAA$\\uparrow$ & mAA$_f\\uparrow$ & $\\tau (ms)\\downarrow$ '
    cprint(header_row + '\\\\')
    cprint(' '.join(cmidrule_parts))
    cprint(column_names + '\\\\')
    cprint('\\midrule')

    # for features in ['splg', 'roma', 'mast3r']:
    for features in ['roma', 'mast3r']:
        master = 'mast3r' == features
        experiments = get_experiments('shared', master=master)
        experiments = [x for x in experiments if 'reproj' not in x]

        # monodepth_methods = sorted(list(set([x.split('+')[0] for x in experiments]) - {'5p'}))
        if master:
            monodepth_methods = list(method_names_shared.keys())[1:]
        else:
            monodepth_methods = list(method_names_shared.keys())[1:-1]

        baseline_methods = ['6p']

        experiments = [f'{prefix}{x}' for x in experiments]
        monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
        baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

        scene_errors = {}
        for scene_list in basenames.values():
            for scene in scene_list:
                print(f"Loading: {scene}")
                scene_errors[scene] = get_median_errors(scene, experiments, features=features, prefix='shared_focal', calc_f_err=True, **kwargs)

        print("Calculating Means")
        means = {k: get_means(scene_errors, v, experiments) for k, v in basenames.items()}

        if master:
            total_rows = len(baseline_methods) + len(monodepth_methods)
        else:
            total_rows = len(baseline_methods) + len(depth_order) * len(monodepth_methods)

        cprint('\\multirow{', str(total_rows), '}{*}{', feature_names[features], '}')

        print_monodepth_rows(0, baseline_methods, method_names_shared, means, cprint=cprint, use_focal=True)
        cprint('\\cmidrule(rl){2-', str(len(basenames) * 5 + 4), '}')

        if master:
            print_monodepth_rows(1, monodepth_methods, method_names_shared, means, cprint=cprint, master=True, use_focal=True)
        else:
            for i in depth_order:
                print_monodepth_rows(i, monodepth_methods, method_names_shared, means, cprint=cprint, use_focal=True)
                if i!= depth_order[-1]:
                    cprint('\\cmidrule(rl){2-', str(len(basenames) * 5 + 4), '}')
        cprint('\\hline')
    cprint('\\end{tabular}}')

def generate_varying_table(prefix='', cprint=print, master=False, **kwargs):
    cprint('\\resizebox{\\linewidth}{!}{')
    num_columns = 4 + 5 * len(basenames)
    column_alignment = 'cclc' + 'c' * len(basenames) * 5
    cprint('\\begin{tabular}{' + column_alignment + '}')
    cprint('\\toprule')
    header_row = '\\multirow{2.5}{*}{{Matches}} & \\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{{Solver}} & \\multirow{2.5}{*}{{Opt}} '
    cmidrule_parts = []
    for i, part in enumerate(basenames.keys()):
        header_row += '& \\multicolumn{5}{c}{' + str(part) + '}'
        cmidrule_parts.append('\\cmidrule(rl){' + str(5 * i + 5) + '-' + str(5 * i + 9) + '}')
    column_names = '& & & & $\\epsilon(^\\circ)\\downarrow$ & $\\epsilon_f(^\\circ)\\downarrow$ & mAA$\\uparrow$ & mAA$_f\\uparrow$ & $\\tau (ms)\\downarrow$ '
    for i in range(len(basenames) - 1):
        column_names += ' & $\\epsilon(^\\circ)\\downarrow$ & $\\epsilon_f(^\\circ)\\downarrow$ & mAA$\\uparrow$ & mAA$_f\\uparrow$ & $\\tau (ms)\\downarrow$ '
    cprint(header_row + '\\\\')
    cprint(' '.join(cmidrule_parts))
    cprint(column_names + '\\\\')
    cprint('\\midrule')

    for features in ['splg', 'roma', 'mast3r']:
    # for features in ['roma', 'mast3r']:
        master = 'mast3r' == features
        experiments = get_experiments('varying', master=master)
        experiments = [x for x in experiments if 'reproj' not in x]

        # monodepth_methods = sorted(list(set([x.split('+')[0] for x in experiments]) - {'5p'}))
        if master:
            monodepth_methods = list(method_names_varying.keys())[1:]
        else:
            monodepth_methods = list(method_names_varying.keys())[1:-1]

        baseline_methods = ['7p']

        experiments = [f'{prefix}{x}' for x in experiments]
        monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
        baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

        scene_errors = {}
        for scene_list in basenames.values():
            for scene in scene_list:
                print(f"Loading: {scene}")
                scene_errors[scene] = get_median_errors(scene, experiments, features=features, prefix='varying_focal',
                                                        calc_f_err=True, **kwargs)

        print("Calculating Means")
        means = {k: get_means(scene_errors, v, experiments) for k, v in basenames.items()}

        if master:
            total_rows = len(baseline_methods) + len(monodepth_methods)
        else:
            total_rows = len(baseline_methods) + len(depth_order) * len(monodepth_methods)

        cprint('\\multirow{', str(total_rows), '}{*}{', feature_names[features], '}')

        print_monodepth_rows(0, baseline_methods, method_names_varying, means, cprint=cprint, use_focal=True)
        cprint('\\cmidrule(rl){2-', str(len(basenames) * 5 + 4), '}')

        if master:
            print_monodepth_rows(1, monodepth_methods, method_names_varying, means, cprint=cprint, master=True,
                                 use_focal=True)
        else:
            for i in depth_order:
                print_monodepth_rows(i, monodepth_methods, method_names_varying, means, cprint=cprint, use_focal=True)
                if i != depth_order[-1]:
                    cprint('\\cmidrule(rl){2-', str(len(basenames) * 5 + 4), '}')
        cprint('\\hline')
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


def type_table(table_func, prefix='', make_pdf=True, master=False, **kwargs):
    if make_pdf:
        if not os.path.exists('pdfs'):
            os.makedirs('pdfs', exist_ok=True)
        kwarg_string = '-'.join([x for x in kwargs.values() if len(x) > 0])
        destination = f'pdfs/{prefix}{table_func.__name__}{kwarg_string}.tex'

        def cprint(*args):
            with open(destination, 'a') as file:
                print(*args, file=file)

        init_latex(destination)
        print(prefix, table_func.__name__)
        table_func(prefix=prefix, cprint=cprint, master=master, **kwargs)
        typeset_latex(destination, cprint=cprint)
    else:
        print(prefix, table_func.__name__)
        table_func(prefix=prefix, **kwargs)


if __name__ == '__main__':
    # just print normally
    # cprint = print

    # basenames.pop('ETH', None)
    basenames.pop('Phototourism', None)
    basenames.pop('ScanNet', None)
    # type_table(generate_calib_table, make_pdf=True, t='2.0t')
    # type_table(generate_shared_table, make_pdf=True, t='2.0t')
    type_table(generate_varying_table, make_pdf=True, t='2.0t')
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