import math
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm


def plot_param_diff(results: Dict[Tuple[int, int], List[List[torch.Tensor]]]):
    data = []
    for (r, t), values in results.items():
        for subj in values:
            for diff in subj:
                v = diff.abs().mean(0).detach().cpu()
                data.append([r, t, f"±{str('{:.1f}'.format(r / 2))}° - ±{str('{:.1f}'.format(t / 2))}mm",
                             v[:3].mean().item(), "Avg rotation error"])
                data.append([r, t, f"±{str('{:.1f}'.format(r / 2))}° - ±{str('{:.1f}'.format(t / 2))}mm",
                             v[:3].max().item(), "Max rotation error"])
                data.append([r, t, f"±{str('{:.1f}'.format(r / 2))}° - ±{str('{:.1f}'.format(t / 2))}mm",
                             v[3:].mean().item(), "Avg translation error"])
                data.append([r, t, f"±{str('{:.1f}'.format(r / 2))}° - ±{str('{:.1f}'.format(t / 2))}mm",
                             v[3:].max().item(), "Max translation error"])
    data = pd.DataFrame(data, columns=["r", "t", "rt", "v", "Legend"])
    plt.clf()
    plt.close("all")
    plt.figure(figsize=(28, 8))
    my_pal = {"Avg rotation error": "g", "Max rotation error": "cyan", "Avg translation error": "r",
              "Max translation error": "pink"}
    bp = sns.boxplot(data, y="v", x="rt", hue="Legend", gap=.2, width=0.8, whis=999, palette=my_pal)
    bp.axes.set_title("Absolute error distrbutions of final parameters with respect to motion-free "
                      "parameters.\nN=10 subjects, 10 repeats per subject", fontsize=20)
    bp.set_xlabel("Parameter deformation sampling ranges: Rotation (degrees) - Translation (mm)", fontsize=18)
    bp.set_ylabel("Absolute error of final parameters", fontsize=16)
    bp.set(ylim=(-.01, 1.1))
    bp.tick_params(labelsize=16)
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.setp(bp.get_legend().get_texts(), fontsize='18')  # for legend text
    plt.setp(bp.get_legend().get_title(), fontsize='0')  # for legend title
    plt.tight_layout()
    fig = bp.get_figure()
    fig.savefig("results_boxplot.png")


def plot_param_diff_before_after(results: Dict[Tuple[int, int], List[List[torch.Tensor]]]):
    data_list = []
    rand_list = []
    for (r, t), values in tqdm(results.items()):
        for subj in values:
            # Expected random value
            r_rad = r * math.pi / 180
            deform = torch.tensor([r, r, r, t, t, t]).reshape((1, 6))
            deform = deform.tile((13, 1))
            rand_diff = (torch.rand((10, *deform.shape)) * deform - deform / 2).abs()
            for v in rand_diff[:, :3].flatten():
                rand_list.append([r, t, f"±{str('{:.1f}'.format(r / 2))}° - ±{str('{:.1f}'.format(t / 2))}mm",
                                  v.detach().cpu().item(), "Starting rotation error"])
            for v in rand_diff[:, 3:].flatten():
                rand_list.append([r, t, f"±{str('{:.1f}'.format(r / 2))}° - ±{str('{:.1f}'.format(t / 2))}mm",
                                  v.detach().cpu().item(), "Starting translation error"])
    rand_data = pd.DataFrame(rand_list, columns=["r", "t", "rt", "v", "Legend"])
    for (r, t), values in tqdm(results.items()):
        for subj in values:
            for diff in subj:
                v = diff.abs().detach().cpu()
                for v_ in v[:, :3].flatten():
                    data_list.append(
                        [r, t, f"±{str('{:.1f}'.format(r / 2))}° - ±{str('{:.1f}'.format(t / 2))}mm",
                         v_.item(), "Final rotation error"])
                for v_ in v[:, 3:].flatten():
                    data_list.append(
                        [r, t, f"±{str('{:.1f}'.format(r / 2))}° - ±{str('{:.1f}'.format(t / 2))}mm",
                         v_.item(), "Final translation error"])
    subj_data = pd.DataFrame(data_list, columns=["r", "t", "rt", "v", "Legend"])
    data = pd.concat((rand_data, subj_data))
    plt.clf()
    plt.close("all")
    plt.figure(figsize=(28, 8))
    bp = sns.boxplot(data, y="v", x="rt", hue="Legend", gap=.2, width=0.8, whis=999)
    bp.axes.set_title("Absolute error distrbutions of parameters before and after optimization with respect to "
                      "motion-free parameters.\nN=10 subjects, 10 repeats per subject", fontsize=20)
    bp.set_xlabel("Parameter deformation sampling ranges: Rotation (degrees) - Translation (mm)", fontsize=18)
    bp.set_ylabel("Absolute error parameters", fontsize=18)
    bp.tick_params(labelsize=16)
    bp.set(ylim=(-.1, 25.1))
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.setp(bp.get_legend().get_texts(), fontsize='18')  # for legend text
    plt.setp(bp.get_legend().get_title(), fontsize='0')  # for legend title
    plt.tight_layout()
    fig = bp.get_figure()
    fig.savefig("results_boxplot_before-after.png")
