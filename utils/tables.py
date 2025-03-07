import json
import os

import numpy as np

from utils.data import R_err_fun, t_err_fun, err_fun_pose

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

def get_median_errors(scene, experiments, prefix='calibrated', t='', features='splg', calc_f_err=False, **kwargs):
    if len(t) > 0:
        t_string = f'-{t}'
    else:
        t_string = ''

    json_path = f'{prefix}-{scene}_{features}{t_string}.json'

    if 'varying' in prefix:
        graph_json_path = f'{prefix}-{scene}_{features}{t_string}-graph.json'
    else:
        graph_json_path = f'{prefix}-graph-{scene}_{features}{t_string}.json'

    results = []

    try:
        with open(os.path.join('results_new', json_path), 'r') as f:
            results.extend(json.load(f))
    except Exception as e:
        print(f"{json_path} not found! not adding it!")
        print(e)

    try:
        with open(os.path.join('results_new', graph_json_path)) as f:
            graph_results = [x for x in json.load(f) if x['info'].get('iterations', 0) == 1000]
        results.extend(graph_results)
    except Exception as e:
        print(f"{graph_json_path} not found! not adding it!")
        print(e)

    if len(results) == 0:
        print("No data loaded!")

    exp_results = {exp: [] for exp in experiments}
    for r in results:
        try:  # if we do not have such key in dict skip
            exp_results[r['experiment']].append(r)
        except Exception:
            ...

    out = {}

    median_samples = np.nanmedian([len(exp_results[exp]) for exp in experiments])
    if median_samples == 0:
        print("Median samples val is 0")

    for exp in experiments:
        n = len(exp_results[exp])
        if n != median_samples:
            print(f"Scene: {scene} - experiment: {exp} has {n} samples while median is {median_samples}")
        d = {}

        R_errs = np.array([R_err_fun(x) for x in exp_results[exp]])
        R_res = np.array([np.sum(R_errs < t) / len(R_errs) for t in range(1, 11)])
        d['median_R_err'] = np.nanmedian(R_errs)
        d['mAA_R'] = 100 * np.mean(R_res)

        t_errs = np.array([t_err_fun(x) for x in exp_results[exp]])
        t_res = np.array([np.sum(t_errs < t) / len(t_errs) for t in range(1, 11)])
        d['median_t_err'] = np.nanmedian(t_errs)
        d['mAA_t'] = 100 * np.mean(t_res)

        pose_errs = np.array([err_fun_pose(x) for x in exp_results[exp]])
        pose_res = np.array([np.sum(pose_errs < t) / len(pose_errs) for t in range(1, 11)])
        d['median_pose_err'] = np.nanmedian(pose_errs)
        d['mAA_pose'] = 100 * np.mean(pose_res)

        runtimes = [x['info']['runtime'] for x in exp_results[exp]]
        d['mean_runtime'] = np.nanmean(runtimes)

        if calc_f_err:
            f_errs = np.array([x['f_err'] for x in exp_results[exp]])
            f_res = np.array([np.sum(f_errs * 100 < t) / len(f_errs) for t in range(1, 11)])
            d['median_f_err'] = np.nanmedian(f_errs)
            d['mAA_f'] = 100 * np.mean(f_res)

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
    # @staticmethod
    # def __missing__(key):
    #     if 'madpose' not in key and 'reproj' not in key and 'mast3r' not in key:
    #         return key.replace('_', '-') + '-sampson'
    #     return key.replace('_', '-')

    def __getitem__(self, key):
        try:
            if 'reproj' in key:
                key = key.replace('reproj-s', 'reproj').replace('_reproj', '')
            return dict.__getitem__(self, key)
        except Exception:
            return key



method_names_calib = {'5p': '5PT~\\cite{nister2004efficient}',
                      '3p_reldepth': 'Rel3PT~\\cite{Astermark2024}',
                      'p3p': 'P3P~\\cite{ding2023revisiting}',
                      'mad_poselib_shift_scale': '3PT$_{suv}$(M)~\\cite{yu2025relative}',
                      '3p_ours_shift_scale': '3PT$_{suv}$~(\\textbf{ours})',
                      'madpose': '3PT$_{suv}$(M)~\\cite{yu2025relative}',
                      'madpose_ours_scale_shift': '3PT$_{suv}$~(\\textbf{ours})',
                      'mast3r': '-'}

method_names_calib = smart_dict(method_names_calib)

method_names_shared = {'6p': '6PT~\\cite{larsson2017efficient}',
                       '3p_reldepth': '3p3d~\\cite{dingfundamental}',
                       'mad_poselib_shift_scale': '4PT$_{suv}f$(M)~\\cite{yu2025relative}',
                       '4p_ours_scale_shift': '4PT$_{suv}f$(\\textbf{ours})',
                       '3p_ours_scale': '3PT$_{s00}f$(\\textbf{ours})',
                       # '3p_ours': '3PT$_{100}$f(\\textbf{ours})',
                       'madpose': '4PT$_{suv}f$(M)~\\cite{yu2025relative}',
                       'madpose_ours_scale': '3PT$_{s00}f$(\\textbf{ours})',
                       'mast3r': '-'
                       }
# method_names_shared.update({f'nLO-{k}': v for k, v in method_names_shared.items()})
# method_names_shared.update({f'GLO-{k}': v for k, v in method_names_shared.items()})
# method_names_shared.update({f'NN-{k}': v for k, v in method_names_shared.items()})
method_names_shared = smart_dict(method_names_shared)


method_names_varying = {'7p': '7PT~\\cite{hartley2003multiple}',
                        '4p4d': '4p4d~\\cite{dingfundamental}',
                        'mad_poselib_shift_scale': '4PT$_{suv}f_{1,2}$(M)~\\cite{yu2025relative}',
                        '4p_ours_scale_shift': '4PT$_{suv}f_{1,2}$(\\textbf{ours})',
                        '3p_ours_scale': '3PT$_{s00}f_{1,2}$(\\textbf{ours})',
                        # '3p_ours': '3PT$_{100}$f_{1,2}(\\textbf{ours})',
                        'madpose': '4PT$_{suv}f_{1,2}$(M)~\\cite{yu2025relative}',
                        'madpose_ours_scale': '3PT$_{s00}f_{1,2}$(\\textbf{ours})',
                        'mast3r': '-'
                       }
# method_names_varying.update({f'nLO-{k}': v for k, v in method_names_varying.items()})
# method_names_varying.update({f'GLO-{k}': v for k, v in method_names_varying.items()})
# method_names_varying.update({f'NN-{k}': v for k, v in method_names_varying.items()})
# method_names_varying = {}
method_names_varying = smart_dict(method_names_varying)

depth_names = {0: '-',
               1: 'Real \\\\ Depth',
               2: 'MiDas \\\\ \\cite{birkl2023midas}',
               3: 'DPT \\\\ \\cite{ranftl2021vision}',
               4: 'ZoeDepth \\\\ \\cite{bhat2023zoedepth}',
               5: 'DA v1\\\\ \\cite{yang2024depth}',
               6: 'DA v2 \\\\ \\cite{yang2024depthv2}',
               7: 'Depth Pro \\\\ \\cite{bochkovskii2024depth}',
               12: 'UniDepth \\\\ \\cite{piccinelli2024unidepth}',
               8: 'Metric3d V2 \\\\ \\cite{hu2024metric3d}',
               11: 'Marigold \\\\ \\cite{ke2023repurposing}',
               9: 'Marigold + FT \\\\ \\cite{martingarcia2024diffusione2eft}',
               10: 'MoGe \\\\ \\cite{wang2024moge}'}

depth_names = {k: f'\\makecell{{{v}}}' for k, v in depth_names.items()}

# depth_order = [1, 2, 3, 4, 5, 6, 7, 12, 8, 11, 9, 10]
depth_order = [1, 2, 6, 10, 12]


def method_opts(method):
    if 'reproj-s' in method:
        return 'R$_s$'
    if 'reproj' in method:
        return 'R'
    if 'madpose' not in method and 'reproj' not in method and 'mast3r' not in method:
        return 'S'
    if 'madpose' in method:
        return 'H~\\cite{yu2025relative}'
    if 'mast3r' in method:
        return 'M~\\cite{leroy2024grounding}'


feature_names = {'splg': 'SP+LG~\\cite{detone2018superpoint, lindenberger2023lightglue}',
                 'roma': 'RoMA~\\cite{edstedt2024roma}',
                 'mast3r': 'Mast3r~\\cite{leroy2024grounding}'}