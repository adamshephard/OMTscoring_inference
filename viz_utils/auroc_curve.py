import os
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12

def plot_curve(ax, tprs, fprs, aucs, col='b', label=None, print_stds=True):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(fprs, mean_tpr)
    std_auc = np.std(aucs)
    if label == None:
        if print_stds == True:
            label = f"Average (AUROC = {mean_auc:.2f} $\pm$ {std_auc:.2f})"
        else:
            label = f"Average (AUROC = {mean_auc:.2f})"
    else:
        if print_stds == True:
            label = f"{label} (AUROC = {mean_auc:.2f} $\pm$ {std_auc:.2f})"
        else:
            label = f"{label} (AUROC = {mean_auc:.2f})"


    ax.plot(
        fprs,
        mean_tpr,
        color=col,
        label=label,
        lw=2,
        alpha=0.8,
    )
    if print_stds == True:
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            fprs,
            tprs_lower,
            tprs_upper,
            color=col, #"grey",
            alpha=0.2,
            # label=r"$\pm$ 1 std. dev.",
        )
    return ax

def auroc_curve(tprs, fprs, aucs, nr_curves=1, labels=None, print_stds=True):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'r']
    if nr_curves == 1:
        plot_curve(ax, tprs, fprs, aucs, col='b', print_stds=print_stds)
    else:
        for idx in range(0, nr_curves):
            plot_curve(ax, tprs[idx], fprs[idx], aucs[idx], colors[idx], labels[idx], print_stds=print_stds)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    return fig, ax


def plot_ensemble_curve(ax, tprs, fprs, aucs, col1='b', col2='grey', label=None):
    # Assumes format [pred1, pred2, pred3,..., ens]
    ens_tpr = tprs[-1]
    # ens_fpr = fprs[-1]
    ens_auc = aucs[-1]
    tprs = tprs[0:-1]
    # fprs = fprs[0:-1]
    aucs = aucs[0:-1]

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(fprs, mean_tpr)
    std_auc = np.std(aucs)
    if label == None:
        label1 = f"Ensemble (AUROC = {ens_auc:.2f})"
        label2 = f"Average (AUROC = {mean_auc:.2f} $\pm$ {std_auc:.2f})"
    else:
        label1 = f"{label} Ensemble (AUROC = {ens_auc:.2f})"
        label2 = f"{label} Average (AUROC = {mean_auc:.2f} $\pm$ {std_auc:.2f})"

    # ensemble curve
    ax.plot(
        fprs,
        ens_tpr,
        color=col2,
        label=label1,
        lw=2,
        alpha=0.8,
    )
    # average curve       
    ax.plot(
        fprs,
        mean_tpr,
        color=col1,
        label=label2,
        lw=2,
        alpha=0.8,
    )

    # lower and upper confidence intervals
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        fprs,
        tprs_lower,
        tprs_upper,
        color=col1, #"grey",
        alpha=0.2,
        # label=r"$\pm$ 1 std. dev.",
    )
    return ax

def ensemble_auroc_curve(tprs, fprs, aucs, nr_curves=1, labels=None):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'r']
    colors2 = ['m', 'y', 'k', 'r', 'purple']
    if nr_curves == 1:
        plot_ensemble_curve(ax, tprs, fprs, aucs, col1='b', col2='grey')
    else:
        for idx in range(0, nr_curves):
            plot_ensemble_curve(ax, tprs[idx], fprs[idx], aucs[idx], colors[idx], colors2[idx], labels[idx])

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    return fig, ax