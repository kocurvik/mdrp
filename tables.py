import json
import os
import subprocess

# from pylatex import Document, Section, Subsection, Command, Tabular, MultiColumn, MultiRow, NoEscape

import numpy as np

from utils.data import basenames, R_err_fun, t_err_fun, err_fun_pose, \
    get_experiments, basenames_eth


def get_median_errors(scene, experiments, prefix='calibrated', t='', features='splg', calc_f_err=False):
    if len(t) > 0:
        t_string = f'-{t}'
    else:
        t_string = ''

    json_path = f'{prefix}-{scene}_{features}{t_string}.json'
    if 'varying' in prefix:
        graph_json_path = f'prefix-{scene}_{features}{t_string}-graph.json'
    else:
        graph_json_path = f'prefix-graph-{scene}_{features}{t_string}.json'
        
    with open(os.path.join('results', json_path), 'r') as f:
        results = json.load(f)
        
    with open(os.path.join('results_new', graph_json_path)):
        graph_results = [x for x in json.load(f) if x['info']['iterations'] == 1000]
        
    results.extend(graph_results)

    exp_results = {exp: [] for exp in experiments}
    for r in results:
        try:  # if we do not have such key in dict skip
            exp_results[r['experiment']].append(r)
        except Exception:
            ...

    out = {}

    median_samples = np.nanmedian([len(exp_results[exp]) for exp in experiments])

    for exp in experiments:
        n = len(exp_results[exp])
        if n != median_samples:
            print(f"Scene: {scene} - experiment: {exp} has only {n} samples while median is {median_samples}")
        d = {}

        R_errs = np.array([R_err_fun(x) for x in exp_results[exp]])
        R_res = np.array([np.sum(R_errs < t) / len(R_errs) for t in range(1, 11)])
        d['median_R_err'] = np.nanmedian(R_errs)
        d['mAA_R'] = np.mean(R_res)

        t_errs = np.array([t_err_fun(x) for x in exp_results[exp]])
        t_res = np.array([np.sum(t_errs < t) / len(t_errs) for t in range(1, 11)])
        d['median_t_err'] = np.nanmedian(t_errs)
        d['mAA_t'] = np.mean(t_res)

        pose_errs = np.array([err_fun_pose(x) for x in exp_results[exp]])
        pose_res = np.array([np.sum(pose_errs < t) / len(pose_errs) for t in range(1, 11)])
        d['median_pose_err'] = np.nanmedian(pose_errs)
        d['mAA_pose'] = np.mean(pose_res)

        runtimes = [x['info']['runtime'] for x in exp_results[exp]]
        d['mean_runtime'] = np.nanmean(runtimes)

        if calc_f_err:
            f_errs = np.array([x['f_err'] for x in exp_results[exp]])
            f_res = np.array([np.sum(f_errs * 100 < t) / len(f_errs) for t in range(1, 11)])
            d['median_f_err'] = np.nanmedian(f_errs)
            d['mAA_f'] = np.mean(f_res)

        out[exp] = d

    return out


def get_means(scene_errors, scenes, experiments):
    metrics = list(scene_errors[scenes[0]][experiments[0]].keys())

    means = {}

    for experiment in experiments:
        means[experiment] = {}
        for metric in metrics:
            means[experiment][metric] = np.mean([scene_errors[scene][experiment][metric] for scene in scenes])

    return means


class smart_dict(dict):
    @staticmethod
    def __missing__(key):
        if 'madpose' not in key and 'reproj' not in key and 'mast3r' not in key:
            return key.replace('_', '-') + '-sampson'
        return key.replace('_', '-')


method_names_calib = {'5p': '5PT', '3p_monodepth': '3PT$_{suv}$', '3p_reldepth': 'Rel3PT', 'p3p': 'P3P'}
method_names_calib.update({f'nLO-{k}': v for k, v in method_names_calib.items()})
method_names_calib.update({f'GLO-{k}': v for k, v in method_names_calib.items()})
method_names_calib.update({f'NN-{k}': v for k, v in method_names_calib.items()})
method_names_calib = {}
method_names_calib = smart_dict(method_names_calib)

method_names_shared = {'6p': '6PT', '3p_reldepth': '3p3d', '4p_monodepth_gb': '4PT$_{suv}f$(GB)',
                       '4p_monodepth_eigen': '4PT$_{suv}f$(Eigen)'}
method_names_shared.update({f'nLO-{k}': v for k, v in method_names_shared.items()})
method_names_shared.update({f'GLO-{k}': v for k, v in method_names_shared.items()})
method_names_shared.update({f'NN-{k}': v for k, v in method_names_shared.items()})
method_names_shared = {}
method_names_shared = smart_dict(method_names_shared)

method_names_varying = {'7p': '7PT', '4p4d': '4p4d', '4p_eigen': '4PT$_{suv}f_1f_2$(Eigen)',
                        '4p_gj': '4PT$_{suv}f_1f_2$(GJ)'}
method_names_varying.update({f'nLO-{k}': v for k, v in method_names_varying.items()})
method_names_varying.update({f'GLO-{k}': v for k, v in method_names_varying.items()})
method_names_varying.update({f'NN-{k}': v for k, v in method_names_varying.items()})
method_names_varying = {}
method_names_varying = smart_dict(method_names_varying)

depth_names = {0: '-',
               1: 'Real Depth',
               2: 'MiDas~\\cite{birkl2023midas}',
               3: 'DPT~\\cite{ranftl2021vision}',
               4: 'ZoeDepth~\\cite{bhat2023zoedepth}',
               5: 'DA V1~\\cite{yang2024depth}',
               6: 'DA V2~\\cite{yang2024depthv2}',
               7: 'Depth Pro~\\cite{bochkovskii2024depth}',
               12: 'UniDepth~\\cite{piccinelli2024unidepth}',
               8: 'Metric3d V2~\\cite{hu2024metric3d}',
               11: 'Marigold~\\cite{ke2023repurposing}',
               9: 'Marigold + FT~\\cite{martingarcia2024diffusione2eft}',
               10: 'MoGe~\\cite{wang2024moge}'}

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
        cprint('\\multirow{', len(methods), '}{*}{Mast3r}')
    else:
        cprint('\\multirow{', len(methods), '}{*}{', depth_names[depth], '}')
    for i, method in enumerate(methods):
        cprint(f'& {method_names[method]} & {"&".join(text_rows[i])} \\\\')
    cprint('\\hline')


def generate_calib_table(cprint=print, prefix='', master=False, **kwargs):
    # experiments = [f'3p_monodepth+{i}' for i in range(1, 13)]
    # experiments.extend([f'3p_reldepth+{i}' for i in range(1, 13)])
    # experiments.extend([f'p3p+{i}' for i in range(1, 13)])
    # experiments.append('5p')

    experiments = get_experiments('calib', master=master)

    monodepth_methods = sorted(list(set([x.split('+')[0] for x in experiments]) - {'5p'}))
    baseline_methods = ['5p']

    experiments = [f'{prefix}{x}' for x in experiments]
    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

    scene_errors = {}
    for scene_list in basenames.values():
        for scene in scene_list:
            print(f"Loading: {scene}")
            scene_errors[scene] = get_median_errors(scene, experiments, prefix='calibrated', **kwargs)

    print("Calculating Means")
    means = {k: get_means(scene_errors, v, experiments) for k, v in basenames.items()}

    cprint('\\resizebox{\\textwidth}{!}{')
    num_columns = 2 + 3 * len(basenames)
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
        header_row += '& \\multicolumn{3}{c}{' + str(part) + '}'
        cmidrule_parts.append('\\cmidrule(rl){' + str(3 * i + 3) + '-' + str(3 * i + 5) + '}')
    # Add the second row header for each part (the column names)
    column_names = '& & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & $\\tau (ms)\\downarrow$ '
    for i in range(len(basenames) - 1):
        column_names += ' & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & $\\tau (ms)\\downarrow$ '
    # Print the first and second header rows
    cprint(header_row + '\\\\')
    cprint(' '.join(cmidrule_parts))
    cprint(column_names + '\\\\')
    cprint('\\midrule')

    # table head
    # cprint('\\resizebox{\\textwidth}{!}{')
    # cprint('\\begin{tabular}{cccccccccccc}')
    # cprint('\\toprule')
    # cprint('\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{Method} & \\multicolumn{3}{c}{Phototourism} & \\multicolumn{3}{c}{ETH3D}  \\\\ \\cmidrule(rl){3-5} \\cmidrule(rl){6-8}')
    # cprint('& & $\\epsilon_{\\M{pose}}(^\\circ)\\downarrow$ & mAA($\\M{pose}$)$\\uparrow$& $\\tau (ms)\\downarrow$ \\  & $\\epsilon_{\\M{pose}}(^\\circ)\\downarrow$ & mAA($\\M{pose}$)$\\uparrow$& $\\tau (ms)\\downarrow$ \\ \\\\ \\midrule')

    print_monodepth_rows(0, baseline_methods, method_names_calib, means, cprint=cprint)

    if master:
        print_monodepth_rows(1, monodepth_methods, method_names_calib, means, cprint=cprint, master=True)
    else:
        for i in depth_order:
            print_monodepth_rows(i, monodepth_methods, method_names_calib, means, cprint=cprint)
    cprint('\\end{tabular}}')


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
            scene_errors[scene] = get_median_errors(scene, experiments, calc_f_err=True, prefix='shared_focal', **kwargs)

    print("Calculating Means")
    means = {k: get_means(scene_errors, v, experiments) for k, v in basenames.items()}

    cprint('\\resizebox{\\textwidth}{!}{')
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
    # cprint('\\resizebox{\\textwidth}{!}{')
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

    scene_errors = {}
    for scene_list in basenames.values():
        for scene in scene_list:
            print(f"Loading: {scene}")
            scene_errors[scene] = get_median_errors(scene, experiments, calc_f_err=True, prefix='varying_focal', **kwargs)

    print("Calculating Means")
    means = {k: get_means(scene_errors, v, experiments) for k, v in basenames.items()}

    cprint('\\resizebox{\\textwidth}{!}{')
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
    # cprint('\\resizebox{\\textwidth}{!}{')
    # cprint('\\begin{tabular}{cccccccccccccccc}')
    # cprint('\\toprule')
    # cprint('\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{Method} & \\multicolumn{5}{c}{Phototourism} & \\multicolumn{5}{c}{ETH3D}  \\\\ \\cmidrule(rl){3-7} \\cmidrule(rl){8-12}')
    # cprint('\\cmidrule(rl){3-7} \\cmidrule(rl){8-12} & & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\  & $\\epsilon_{\\M {pose}}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M {pose}$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\ \\\\ \\midrule')

    print_monodepth_rows(0, baseline_methods, method_names_varying, means, use_focal=True,
                         cprint=cprint)
    if master:
        print_monodepth_rows(1, monodepth_methods, method_names_varying, means, use_focal=True, cprint=cprint, master=True)
    else:
        for i in depth_order:
            print_monodepth_rows(i, monodepth_methods, method_names_varying, means, use_focal=True, cprint=cprint)
    cprint('\\end{tabular}}')


def init_latex(destination):
    global latex_preamble
    latex_preamble = """
    \\documentclass{article}
    % Add the required LaTeX packages
    \\usepackage{graphicx}
    \\usepackage{booktabs}
    \\usepackage{tikz}
    \\usepackage{siunitx}
    \\usepackage{verbatim}
    \\usepackage{color}
    \\usepackage{colortbl}
    \\usepackage{multirow}
    \\usepackage{makecell}
    \\usepackage{pifont}
    \\usepackage{soul}
    \\usepackage{listings}
    \\lstset{basicstyle=\\ttfamily, mathescape}
    \\usepackage{rotating}
    \\newcommand*\\rot{\\rotatebox{90}}
    \\newcommand{\\YES}{\\color{Green}{\\ding{52}}}
    \\definecolor{mygray}{gray}{.92}
    \\newsavebox\\CBox
    \\def\\textBF#1{\\sbox\\CBox{#1}\\resizebox{\\wd\\CBox}{\\ht\\CBox}{\\textbf{#1}}}
    \\newcommand*{\\VV}[1]{\\textcolor{blue}{VV: [#1]}}
    \\newcommand{\\M}[1]{\\mathbf{#1}}
    \\definecolor{wincolor}{rgb}{0.95, 0.2, 0.2}
    \\newcommand{\\win}[1]{\\textcolor{wincolor}{\\bfseries{#1}}}
    \\newcommand{\\second}[1]{\\textcolor{NavyBlue}{\\bfseries{#1}}}
    \\newcommand{\\KITTI}{\\dataset{KITTI}}
    \\newcommand{\\ETH}{\\dataset{ETH3D}}
    \\newcommand{\\PHONE}{\\dataset{PHONE}}
    \\newcommand{\\Phototourism}{\\dataset{Phototourism}}
    \\def\\gb{Gr{\"o}bner basis\\xspace}
    \\newcommand{\\dataset}[1]{{\\fontfamily{cmtt}\\selectfont #1} }
    %\\usepackage[dvipsnames]{xcolor}
    \\begin{document}
    """
    # Write the first part (preamble) to a .tex file
    with open(destination, 'w') as file:
        file.write(latex_preamble)


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
    # print("No LO calib")
    # generate_calib_table(lo=False)

    # just print normally
    # cprint = print

    type_table(generate_calib_table, make_pdf=True, t='2.0t', features='splg')
    type_table(generate_shared_table, make_pdf=True, t='2.0t', features='splg')
    type_table(generate_varying_table, make_pdf=True, t='2.0t', features='splg')

    type_table(generate_calib_table, make_pdf=True, t='2.0t', features='roma')
    type_table(generate_shared_table, make_pdf=True, t='2.0t', features='roma')
    type_table(generate_varying_table, make_pdf=True, t='2.0t', features='roma')

    basenames.pop('ETH', None)
    type_table(generate_varying_table, master=True, make_pdf=True, t='2.0t', features='mast3r')

    basenames.pop('Phototourism', None)
    basenames['ETH'] = basenames_eth
    type_table(generate_shared_table, master=True, make_pdf=True, t='2.0t', features='mast3r')



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