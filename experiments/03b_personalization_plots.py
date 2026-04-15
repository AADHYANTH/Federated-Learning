import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = "data/processed/personalization_results.csv"
PLOTS_DIR = "data/processed/plots"

REQUIRED_COLUMNS = [
    "user_id",
    "global_AUC",
    "global_TPR",
    "global_FPR",
    "global_FNR",
    "cal_AUC",
    "cal_TPR",
    "cal_FPR",
    "cal_tau",
    "cal_a",
    "cal_b",
    "delta_TPR_cal",
    "delta_FPR_cal",
    "head_AUC",
    "head_TPR",
    "head_FPR",
    "head_tau",
    "delta_TPR_head",
    "delta_FPR_head",
]


def _load_results(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}. "
            "Run experiments/03_personalization.py to regenerate results."
        )
    return df


def _save_figure(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(save_path)


def chart1_auc_comparison(df, plots_dir):
    user_ids = df["user_id"].to_numpy(dtype=int)
    x = np.arange(len(user_ids))
    w = 0.27

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, df["global_AUC"].to_numpy(), width=w, color="steelblue", label="Global")
    ax.bar(x, df["cal_AUC"].to_numpy(), width=w, color="darkorange", label="Calibration")
    ax.bar(x + w, df["head_AUC"].to_numpy(), width=w, color="forestgreen", label="Head")

    ax.set_xticks(x)
    ax.set_xticklabels(user_ids)
    ax.set_xlabel("user_id")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Per-User ROC-AUC: Global vs Calibration vs Personalized Head")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    save_path = os.path.join(plots_dir, "personalization_auc_comparison.png")
    _save_figure(fig, save_path)


def chart2_tpr_improvement(df, plots_dir):
    g_tpr = df["global_TPR"].to_numpy()
    c_tpr = df["cal_TPR"].to_numpy()
    h_tpr = df["head_TPR"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    colors_cal = np.where(c_tpr >= g_tpr, "green", "red")
    colors_head = np.where(h_tpr >= g_tpr, "green", "red")

    axes[0].scatter(g_tpr, c_tpr, c=colors_cal, alpha=0.8, edgecolors="black", linewidths=0.4)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1.2)
    axes[0].set_title("Calibration")
    axes[0].set_xlabel("global_TPR")
    axes[0].set_ylabel("cal_TPR")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(g_tpr, h_tpr, c=colors_head, alpha=0.8, edgecolors="black", linewidths=0.4)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1.2)
    axes[1].set_title("Personalized Head")
    axes[1].set_xlabel("global_TPR")
    axes[1].set_ylabel("head_TPR")
    axes[1].grid(alpha=0.25)

    fig.suptitle("TPR Before vs After Personalization", y=1.02)

    save_path = os.path.join(plots_dir, "tpr_improvement.png")
    _save_figure(fig, save_path)


def _hist_delta_tpr(ax, series, title):
    vals = np.asarray(series, dtype=float)
    pos = vals[vals >= 0]
    neg = vals[vals < 0]
    if neg.size > 0:
        ax.hist(neg, bins=14, color="lightgray", alpha=0.85, label="delta < 0")
    if pos.size > 0:
        ax.hist(pos, bins=14, color="green", alpha=0.75, label="delta >= 0")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.grid(alpha=0.2)


def _hist_delta_fpr(ax, series, title):
    vals = np.asarray(series, dtype=float)
    neg = vals[vals < 0]
    non_neg = vals[vals >= 0]
    if non_neg.size > 0:
        ax.hist(non_neg, bins=14, color="lightgray", alpha=0.85, label="delta >= 0")
    if neg.size > 0:
        ax.hist(neg, bins=14, color="red", alpha=0.75, label="delta < 0")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.grid(alpha=0.2)


def chart3_delta_distributions(df, plots_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    _hist_delta_tpr(axes[0, 0], df["delta_TPR_cal"], "delta_TPR_cal")
    _hist_delta_fpr(axes[0, 1], df["delta_FPR_cal"], "delta_FPR_cal")
    _hist_delta_tpr(axes[1, 0], df["delta_TPR_head"], "delta_TPR_head")
    _hist_delta_fpr(axes[1, 1], df["delta_FPR_head"], "delta_FPR_head")

    axes[0, 0].legend(loc="upper left")
    axes[0, 1].legend(loc="upper left")
    axes[1, 0].legend(loc="upper left")
    axes[1, 1].legend(loc="upper left")

    fig.suptitle("Distribution of Per-User Improvements", y=1.02)

    save_path = os.path.join(plots_dir, "delta_distributions.png")
    _save_figure(fig, save_path)


def chart4_calibration_params(df, plots_dir):
    fig, ax = plt.subplots(figsize=(14, 6))

    sc = ax.scatter(
        df["cal_a"],
        df["cal_b"],
        c=df["cal_TPR"],
        cmap="viridis",
        s=70,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.35,
    )

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("cal_a")
    ax.set_ylabel("cal_b")
    ax.set_title("Per-User Calibration Parameters (a=scale, b=bias)")
    ax.grid(alpha=0.25)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Achieved TPR")

    far_idx = (df["cal_a"] - 1.0).abs().idxmax()
    far_user = int(df.loc[far_idx, "user_id"])
    far_a = float(df.loc[far_idx, "cal_a"])
    far_b = float(df.loc[far_idx, "cal_b"])
    ax.annotate(
        f"User {far_user}: |a-1| highest",
        xy=(far_a, far_b),
        xytext=(far_a + 0.05, far_b + 0.05),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=9,
    )
    ax.text(
        0.02,
        0.98,
        "Users with cal_a far from 1.0 needed most adjustment",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
    )

    save_path = os.path.join(plots_dir, "calibration_params.png")
    _save_figure(fig, save_path)


def chart5_full_pipeline_summary(df, plots_dir):
    centralized = {
        "ROC_AUC": 0.9990,
        "TPR": 0.9487,
        "FPR": 0.0140,
        "FNR": 1.0 - 0.9487,
    }
    federated = {
        "ROC_AUC": 0.9993,
        "TPR": 0.9136,
        "FPR": 0.0000,
        "FNR": 1.0 - 0.9136,
    }

    best_auc = np.maximum(df["cal_AUC"].to_numpy(), df["head_AUC"].to_numpy())
    best_tpr = np.maximum(df["cal_TPR"].to_numpy(), df["head_TPR"].to_numpy())
    best_fpr = np.minimum(df["cal_FPR"].to_numpy(), df["head_FPR"].to_numpy())
    cal_fnr = 1.0 - df["cal_TPR"].to_numpy()
    head_fnr = 1.0 - df["head_TPR"].to_numpy()
    best_fnr = np.minimum(cal_fnr, head_fnr)

    personalized = {
        "ROC_AUC": float(np.mean(best_auc)),
        "TPR": float(np.mean(best_tpr)),
        "FPR": float(np.mean(best_fpr)),
        "FNR": float(np.mean(best_fnr)),
    }

    metric_names = ["ROC_AUC", "TPR", "FPR", "FNR"]
    x = np.arange(len(metric_names))
    w = 0.24

    c_vals = [centralized[m] for m in metric_names]
    f_vals = [federated[m] for m in metric_names]
    p_vals = [personalized[m] for m in metric_names]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, c_vals, width=w, color="steelblue", label="Centralized")
    ax.bar(x, f_vals, width=w, color="darkorange", label="Federated")
    ax.bar(x + w, p_vals, width=w, color="forestgreen", label="Personalized")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Metric Value")
    ax.set_title("Complete Pipeline Results: Centralized -> Federated -> Personalized")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    fpr_x = x[2]
    fpr_y_top = max(c_vals[2], f_vals[2], p_vals[2]) + 0.1
    fpr_y_bottom = min(c_vals[2], f_vals[2], p_vals[2]) + 0.01
    ax.annotate(
        "lower is better",
        xy=(fpr_x, fpr_y_bottom),
        xytext=(fpr_x, fpr_y_top),
        ha="center",
        arrowprops=dict(arrowstyle="-|>", lw=1.2),
    )

    save_path = os.path.join(plots_dir, "full_pipeline_summary.png")
    _save_figure(fig, save_path)


def main(results_path=RESULTS_PATH, plots_dir=PLOTS_DIR):
    plt.style.use("seaborn-v0_8-whitegrid")
    df = _load_results(results_path)

    chart1_auc_comparison(df, plots_dir)
    chart2_tpr_improvement(df, plots_dir)
    chart3_delta_distributions(df, plots_dir)
    chart4_calibration_params(df, plots_dir)
    chart5_full_pipeline_summary(df, plots_dir)

    print("All 5 personalization charts saved to data/processed/plots/")


if __name__ == "__main__":
    main()
