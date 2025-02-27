import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.data import err_fun_pose, basenames, get_experiments


def get_errors(scene, experiments, prefix='calibrated', t='', features='splg', calc_f_err=False):
    if len(t) > 0:
        json_path = f'{prefix}-{scene}_{features}-{t}.json'
    else:
        json_path = f'{prefix}-{scene}_{features}.json'

    print("Loading: ", json_path)
    with open(os.path.join('results', json_path), 'r') as f:
        results = json.load(f)

    exp_results = {exp: [] for exp in experiments}
    for r in results:
        try:  # if we do not have such key in dict skip
            exp_results[r['experiment']].append(r)
        except Exception:
            ...

    out_pose_errs = {}
    out_f_errs = {}

    # median_samples = np.nanmedian([len(exp_results[exp]) for exp in experiments])

    for exp in experiments:
        # n = len(exp_results[exp])
        # if n != median_samples:
        #     print(f"Scene: {scene} - experiment: {exp} has only {n} samples while median is {median_samples}")
        d = {}

        out_pose_errs[exp] = [err_fun_pose(x) for x in exp_results[exp]]

        runtimes = [x['info']['runtime'] for x in exp_results[exp]]
        d['mean_runtime'] = np.nanmean(runtimes)

        if calc_f_err:
            f_errs = [x['f_err'] for x in exp_results[exp]]
            out_f_errs[exp] = f_errs

    return out_pose_errs, out_f_errs


def generate_error_boxplot(experiments, error_data, title="Error Distribution Across Experiments",
                           ylabel="Error Values", ylim=None, figsize=(10, 6), save_path=None):
    """
    Generate a boxplot of errors for different experiments.

    Parameters:
    -----------
    experiments : list
        List of experiment names (strings).
    error_data : dict
        Dictionary with experiment names as keys and lists of error values as values.
    title : str, optional
        Title of the plot.
    ylabel : str, optional
        Label for the y-axis.
    ylim : tuple, optional
        Limits for the y-axis in the form (ymin, ymax). If None, min will be 0 and max will be
        20% higher than the highest whisker.
    figsize : tuple, optional
        Figure size in inches (width, height).
    save_path : str, optional
        Path to save the figure. If None, the figure will be displayed but not saved.

    Returns:
    --------
    fig, ax : tuple
        matplotlib figure and axis objects.
    """
    # Verify all experiments are in the error_data dictionary
    for exp in experiments:
        if exp not in error_data:
            raise ValueError(f"Experiment '{exp}' not found in error_data.")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data in the order of experiments list
    data_to_plot = [error_data[exp] for exp in experiments]

    # Create the boxplot
    boxplot = ax.boxplot(data_to_plot, patch_artist=True)

    # Set colors for boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    # Set the title and labels
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=10)

    # Set y-axis limits - either user-specified or automatic
    if ylim is None:
        # Get the top whisker value from the boxplot
        # In matplotlib, boxplot returns a dict with 'whiskers' key containing Line2D objects
        # Whiskers come in pairs, with even indices (0, 2, 4...) being bottom whiskers
        # and odd indices (1, 3, 5...) being top whiskers
        whiskers = boxplot['whiskers']
        top_whiskers = [whiskers[i].get_ydata()[1] for i in range(1, len(whiskers), 2)]
        max_whisker = max(top_whiskers)

        # Set y limits from 0 to 20% higher than the highest whisker
        ax.set_ylim(0, max_whisker * 1.2)
    else:
        ax.set_ylim(ylim)

    # Adjust layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def generate_dataset_boxplots(prefix, features, dataset, scenes):
    for depth in [1, 2, 6, 10, 12]:
        experiments = get_experiments(prefix, depths=[depth])

        title = f'{prefix}{dataset}-2.0t-{features}+{depth}'
        print("Loading for ", title)

        all_pose_errs = {exp: [] for exp in experiments}
        all_f_errs = {exp: [] for exp in experiments}

        for scene in scenes:
            out_pose_errs, out_f_errs = get_errors(scene, experiments, prefix=prefix, features=features, t='2.0t', calc_f_err='calibrated'!=prefix)

            for exp in experiments:
                all_pose_errs[exp].extend(out_pose_errs[exp])
                if prefix != 'calibrated':
                    all_f_errs[exp].extend(out_f_errs[exp])

        generate_error_boxplot(experiments, all_pose_errs, title=title, ylabel='Pose Error (deg)', ylim=None,
                               save_path = f'figs/boxplots/{title}-pose.png')


        if prefix != 'calibrated':
            generate_error_boxplot(experiments, all_f_errs, title=title, ylabel='Focal Error', ylim=None,
                                   save_path=f'figs/boxplots/{title}-focal.png')


def generate_boxplots():
    for dataset, scenes in basenames.items():
        for prefix in ['calibrated', 'shared_focal', 'varying_focal']:
            for features in ['splg', 'roma']:
                generate_dataset_boxplots(prefix, features, dataset, scenes)

if __name__ == '__main__':
    generate_boxplots()