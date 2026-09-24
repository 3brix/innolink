"""
Distribution and separability plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config.analysis import MODELS


def plot_metric_distribution(df, metric, group="binder", bins=30, save_path=None):
    """Plot distribution of a metric by group (wide table)."""
    plt.figure(figsize=(7, 5))
    for label, group_df in df.groupby(group):
        plt.hist(group_df[metric].dropna(), bins=bins, alpha=0.5, label=str(label))
    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_metric_separability(long_df, model_groups, output_dir, colors_map, show=False):
    """
    Plot metric separability across binder classes (long table).
    For each model group: one subplot per metric, a stripplot of individual samples over a boxplot of the distribution, colored by source.
    """
    x_order = ["Binder", "Non-Binder", "Mutant+", "Mutant-", "Design"]  # 'Target Shuffle', 

    for model, metrics in model_groups.items():
        model_df = long_df[long_df["metric"].isin(metrics)].copy()
        if model_df.empty:
            continue

        valid_metrics = [
            m for m in metrics
            if not model_df[model_df["metric"] == m]["value"].isna().all()
        ]
        if not valid_metrics:
            continue

        ncols = 4
        nrows = (len(valid_metrics) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
        axes = np.atleast_1d(axes).flatten()

        last_used = -1
        for i, metric in enumerate(valid_metrics):
            ax = axes[i]
            plot_df = model_df[model_df["metric"] == metric].dropna(subset=["value"])
            if plot_df.empty:
                continue
            last_used = i

            sns.stripplot(
                data=plot_df, x="binder_type", y="value", order=x_order, hue="source",
                dodge=True, jitter=0.1, palette=colors_map, size=3, ax=ax, legend=False,
            )
            sns.boxplot(
                data=plot_df, x="binder_type", y="value", order=x_order, hue="source",
                dodge=True, palette=colors_map, width=0.7, showfliers=False, ax=ax, legend=False,
                boxprops=dict(alpha=0.5),
            )
            ax.set_title(metric, fontsize=10)
            ax.set_xlabel("Class")
            ax.set_ylabel("Score")

        for j in range(last_used + 1, len(axes)):  # remove unused axes
            fig.delaxes(axes[j])

        import matplotlib.patches as mpatches

        handles = [
            mpatches.Patch(color=color, label=label)
            for label, color in colors_map.items()
        ]

        fig.legend(
            handles=handles,
            title="Source",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
        )


        plt.suptitle(f"{model.upper()} Sample-Level Separability", fontsize=16, y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(output_dir / f"{model}_separability.png", dpi=200, bbox_inches="tight")
        plt.show() if show else plt.close()


def plot_model_agreement(long_df, output_dir, model_order=None, hue="binder_type", palette="Set2", ylabel="Score", show=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    # Hardcode your fixed order here
    fixed_hue_order = ["Binder", "Mutant+",  "Mutant-", "Non-Binder", "Design", ] #"Target Shuffle"

    for family, family_df in long_df.groupby("metric_family"):
        plot_df = family_df.dropna(subset=["value", "model"])
        if plot_df.empty:
            continue

        split = plot_df[hue].nunique() == 2  
        present = list(plot_df["model"].unique())
        order = [m for m in (model_order or MODELS) if m in present]
        if not order:
            continue

        plt.figure(figsize=(12, 5))
        sns.violinplot(
            data=plot_df, x="model", y="value", hue=hue, order=order,
            hue_order=fixed_hue_order,  # <--- Applied internally
            split=split, inner="quartile", cut=0, density_norm="width", palette=palette,
        )
        plt.title(f"{str(family).upper()} — Model Agreement on Binder Separability")
        plt.xlabel("Model")
        plt.ylabel(ylabel)
        plt.xticks(rotation=30, ha="right")
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / f"{family}_model_agreement.png", dpi=200, bbox_inches="tight")
        plt.show() if show else plt.close()