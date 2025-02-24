import json
import os
import subprocess
from contextlib import contextmanager

# from pylatex import Document, Section, Subsection, Command, Tabular, MultiColumn, MultiRow, NoEscape

import numpy as np

from utils.data import basenames_all, basenames_pt, basenames_eth, R_err_fun, t_err_fun


def get_median_errors(scene, experiments, prefix='calibrated', calc_f_err=False):
    json_path = f'{prefix}-{scene}.json'
    with open(os.path.join('results', json_path), 'r') as f:
        results = json.load(f)

    exp_results = {exp: [] for exp in experiments}
    for r in results:
        try: # if we do not have such key in dict skip
            exp_results[r['experiment']].append(r)
        except Exception:
            ...

    out = {}
    for exp in experiments:
        d = {}

        R_errs = np.array([R_err_fun(x) for x in exp_results[exp]])
        R_res = np.array([np.sum(R_errs < t) / len(R_errs) for t in range(1, 11)])
        d['median_R_err'] = np.nanmedian(R_errs)
        d['mAA_R'] = np.mean(R_res)

        t_errs = np.array([t_err_fun(x) for x in exp_results[exp]])
        t_res = np.array([np.sum(t_errs < t) / len(t_errs) for t in range(1, 11)])
        d['median_t_err'] = np.nanmedian(t_errs)
        d['mAA_t'] = np.mean(t_res)

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

method_names_calib = {'5p': '5PT', '3p_monodepth': '3PT$_{suv}$', '3p_reldepth': 'Rel3PT' , 'p3p': 'P3P' }
method_names_calib.update({f'nLO-{k}': v for k, v in method_names_calib.items()})
method_names_calib.update({f'GLO-{k}': v for k, v in method_names_calib.items()})
method_names_calib.update({f'NN-{k}': v for k, v in method_names_calib.items()})

method_names_shared = {'6p': '6PT', '3p_reldepth': '3p3d', '4p_monodepth_gb': '4PT$_{suv}f$(GB)' , '4p_monodepth_eigen': '4PT$_{suv}f$(Eigen)' }
method_names_shared.update({f'nLO-{k}': v for k, v in method_names_shared.items()})
method_names_shared.update({f'GLO-{k}': v for k, v in method_names_shared.items()})
method_names_shared.update({f'NN-{k}': v for k, v in method_names_shared.items()})

method_names_varying = {'7p': '7PT', '4p4d': '4p4d', '4p_eigen': '4PT$_{suv}f_1f_2$(Eigen)' , '4p_gj': '4PT$_{suv}f_1f_2$(GJ)' }
method_names_varying.update({f'nLO-{k}': v for k, v in method_names_varying.items()})
method_names_varying.update({f'GLO-{k}': v for k, v in method_names_varying.items()})
method_names_varying.update({f'NN-{k}': v for k, v in method_names_varying.items()})

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

depth_order = [1, 2, 3, 4, 5, 6, 7, 12, 8, 11, 9, 10]

def print_monodepth_rows(depth, methods, method_names, phototourism_means, eth3d_means, use_focal=False, cprint=print):
    num_rows = []
    for method in methods:
        if depth > 0:
            method = f'{method}+{depth}'
        if use_focal:
            metrics = ['median_R_err', 'median_t_err', 'median_f_err', 'mAA_R', 'mAA_t', 'mAA_f', 'mean_runtime']
            incdec = [1, 1, 1, -1, -1, -1,  1, 1, 1, 1, -1, -1, -1, 1]
        else:
            metrics = ['median_R_err', 'median_t_err', 'mAA_R', 'mAA_t', 'mean_runtime']
            incdec = [1, 1, -1, -1, 1, 1, 1, -1, -1, 1]

        pt_vals = [phototourism_means[method][x] for x in metrics]
        eth_vals = [eth3d_means[method][x] for x in metrics]

        num_rows.append(pt_vals + eth_vals)

    text_rows = [[f'{x:0.2f}' for x in y] for y in num_rows]
    arr = np.array(num_rows)
    if depth > 0:
        for j in range(len(text_rows[0])):
            idxs = np.argsort(incdec[j] * arr[:, j])
            text_rows[idxs[0]][j] = '\\textbf{' + text_rows[idxs[0]][j] + '}'
            text_rows[idxs[1]][j] = '\\underline{' + text_rows[idxs[1]][j] + '}'

    cprint('\\multirow{', len(methods), '}{*}{', depth_names[depth], '}')
    for i, method in enumerate(methods):
        cprint(f'& {method_names[method]} & {"&".join(text_rows[i])} \\\\')
    cprint('\\hline')


def generate_calib_table(prefix='', cprint=print):
    experiments = [f'3p_monodepth+{i}' for i in range(1, 13)]
    experiments.extend([f'3p_reldepth+{i}' for i in range(1, 13)])
    experiments.extend([f'p3p+{i}' for i in range(1, 13)])
    experiments.append('5p')

    monodepth_methods = ['p3p', '3p_reldepth', '3p_monodepth']
    baseline_methods = ['5p']

    experiments = [f'{prefix}{x}' for x in experiments]
    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

    scene_errors = {}
    for scene in basenames_all:
        print(f"Loading: {scene}")
        scene_errors[scene] = get_median_errors(scene, experiments, prefix='calibrated')

    print("Calculating Means")
    phototourism_means = get_means(scene_errors, basenames_pt, experiments)
    eth3d_means = get_means(scene_errors, basenames_eth, experiments)

    # table head
    cprint('\\resizebox{\\textwidth}{!}{')
    cprint('\\begin{tabular}{cccccccccccc}')
    cprint('\\toprule')
    cprint('\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{Method} & \\multicolumn{5}{c}{Phototourism} & \\multicolumn{5}{c}{ETH3D}  \\\\ \\cmidrule(rl){3-7} \\cmidrule(rl){8-12}')
    cprint('& &\\ $\\epsilon_{\\M R}(^\\circ)\\downarrow$ & $\\epsilon_{\\M t}(^\\circ)\\downarrow$ & mAA($\\M R$)$\\uparrow$ & mAA($\\M t$)$\\uparrow$& $\\tau (ms)\\downarrow$ \\  &\\ $\\epsilon_{\\M R}(^\\circ)\\downarrow$ & $\\epsilon_{\\M t}(^\\circ)\\downarrow$ & mAA($\\M R$)$\\uparrow$ & mAA($\\M t$)$\\uparrow$& $\\tau (ms)\\downarrow$ \\ \\\\ \\midrule')

    print_monodepth_rows(0, baseline_methods, method_names_calib, phototourism_means, eth3d_means, cprint=cprint)
    for i in depth_order:
        print_monodepth_rows(i, monodepth_methods, method_names_calib, phototourism_means, eth3d_means, cprint=cprint)
    cprint('\\end{tabular}}')



def generate_shared_table(prefix='', cprint=print):
    experiments = [f'4p_monodepth_eigen+{i}' for i in range(1, 13)]
    experiments.extend([f'4p_monodepth_gb+{i}' for i in range(1, 13)])
    experiments.extend([f'3p_reldepth+{i}' for i in range(1, 13)])
    experiments.append('6p')

    monodepth_methods = ['3p_reldepth', '4p_monodepth_gb', '4p_monodepth_eigen']
    baseline_methods = ['6p']

    experiments = [f'{prefix}{x}' for x in experiments]
    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]

    scene_errors = {}
    for scene in basenames_all:
        print(f"Loading: {scene}")
        scene_errors[scene] = get_median_errors(scene, experiments, prefix='shared_focal', calc_f_err=True)

    print("Calculating Means")
    phototourism_means = get_means(scene_errors, basenames_pt, experiments)
    eth3d_means = get_means(scene_errors, basenames_eth, experiments)

    # table head
    cprint('\\resizebox{\\textwidth}{!}{')
    cprint('\\begin{tabular}{cccccccccccccccc}')
    cprint('\\toprule')
    cprint('\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{Method} & \\multicolumn{7}{c}{Phototourism} & \\multicolumn{7}{c}{ETH3D}  \\\\ \\cmidrule(rl){3-8} \\cmidrule(rl){8-12}')
    cprint('\\cmidrule(rl){3-9} \\cmidrule(rl){10-16} & &\\ $\\epsilon_{\\M R}(^\\circ)\\downarrow$ & $\\epsilon_{\\M t}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M R$)$\\uparrow$ & mAA($\\M t$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\  &\\ $\\epsilon_{\\M R}(^\\circ)\\downarrow$ & $\\epsilon_{\\M t}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M R$)$\\uparrow$ & mAA($\\M t$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\ \\\\ \\midrule')


    print_monodepth_rows(0, baseline_methods, method_names_shared, phototourism_means, eth3d_means, use_focal=True, cprint=cprint)
    for i in depth_order:
        print_monodepth_rows(i, monodepth_methods, method_names_shared, phototourism_means, eth3d_means, use_focal=True, cprint=cprint)
    cprint('\\end{tabular}}')


def generate_varying_table(prefix='', cprint=print):
    experiments = [f'4p4d+{i}' for i in range(1, 13)]
    experiments.extend([f'4p_eigen+{i}' for i in range(1, 13)])
    experiments.extend([f'4p_gj+{i}' for i in range(1, 13)])
    experiments.append('7p')

    monodepth_methods = ['4p4d', '4p_eigen', '4p_gj']
    baseline_methods = ['7p']

    experiments = [f'{prefix}{x}' for x in experiments]
    monodepth_methods = [f'{prefix}{x}' for x in monodepth_methods]
    baseline_methods = [f'{prefix}{x}' for x in baseline_methods]


    scene_errors = {}
    for scene in basenames_all:
        print(f"Loading: {scene}")
        scene_errors[scene] = get_median_errors(scene, experiments, prefix='varying_focal', calc_f_err=True)

    print("Calculating Means")
    phototourism_means = get_means(scene_errors, basenames_pt, experiments)
    eth3d_means = get_means(scene_errors, basenames_eth, experiments)

    # table head
    cprint('\\resizebox{\\textwidth}{!}{')
    cprint('\\begin{tabular}{cccccccccccccccc}')
    cprint('\\toprule')
    cprint('\\multirow{2.5}{*}{{Depth}} &  \\multirow{2.5}{*}{Method} & \\multicolumn{7}{c}{Phototourism} & \\multicolumn{7}{c}{ETH3D}  \\\\ \\cmidrule(rl){3-8} \\cmidrule(rl){8-12}')
    cprint('\\cmidrule(rl){3-9} \\cmidrule(rl){10-16} & &\\ $\\epsilon_{\\M R}(^\\circ)\\downarrow$ & $\\epsilon_{\\M t}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M R$)$\\uparrow$ & mAA($\\M t$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\  &\\ $\\epsilon_{\\M R}(^\\circ)\\downarrow$ & $\\epsilon_{\\M t}(^\\circ)\\downarrow$ & $\\epsilon_{f}\\downarrow$ & mAA($\\M R$)$\\uparrow$ & mAA($\\M t$)$\\uparrow$ & mAA($f$)$\\uparrow$ & $\\tau (ms)\\downarrow$ \\ \\\\ \\midrule')


    print_monodepth_rows(0, baseline_methods, method_names_varying, phototourism_means, eth3d_means, use_focal=True, cprint=cprint)
    for i in depth_order:
        print_monodepth_rows(i, monodepth_methods, method_names_varying, phototourism_means, eth3d_means, use_focal=True, cprint=cprint)
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
    command = 'pdflatex' if os.name == 'nt' else 'tectonic'
    try:
        cprint('\\end{document}')
        subprocess.run([command, destination, '--output-directory=pdfs'], check=True)
        print("PDF generated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during LaTeX compilation: {e}")

def type_table(table_func, prefix='', make_pdf=True):
    if make_pdf:
        if not os.path.exists('pdfs'):
            os.makedirs('pdfs', exist_ok=True)
        destination = f'pdfs/{prefix}{table_func.__name__}.tex'
        def cprint(*args):
            with open(destination, 'a') as file:
                print(*args, file=file)

        init_latex(destination)
        print(prefix, table_func.__name__)
        table_func(prefix=prefix, cprint=cprint)
        typeset_latex(destination, cprint=cprint)
    else:
        print(prefix, table_func.__name__)
        table_func(prefix=prefix)

if __name__ == '__main__':
    # print("No LO calib")
    # generate_calib_table(lo=False)

    # just print normally
    # cprint = print

    type_table(generate_calib_table, make_pdf=True)
    type_table(generate_shared_table, make_pdf=True)
    type_table(generate_varying_table, make_pdf=True)


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

