import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import roc_curve, auc
from src.models.base_cnn import CNNAnomalyDetector
from src.data.preprocessing import load_user_data
from src.evaluation.metrics import evaluate_model_per_user, compute_metrics, summarize_results
from src.evaluation.plots import plot_per_user_roc, plot_metric_distributions


DATA_DIR = 'data/raw/synthetic'
N_USERS = 50
WINDOW_SIZE = 128
STRIDE = 64
PLOTS_DIR = 'data/processed/plots'
CENTRALIZED_MODEL_PATH = 'data/processed/global_model_centralized.pt'
FEDERATED_MODEL_PATH = 'data/processed/global_model_federated.pt'
CHECKPOINT_DIR = 'data/processed/checkpoints'


def load_model(path):
    model = CNNAnomalyDetector(in_channels=1, window_size=128)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def plot_training_curves(checkpoint_dir, plots_dir):
    rounds = []
    mean_val_losses = []
    mean_roc_aucs = []

    rng = np.random.default_rng(42)

    for round_idx in range(1, 21):
        checkpoint_path = os.path.join(checkpoint_dir, f'round_{round_idx:03d}.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')

        model = load_model(checkpoint_path)
        sampled_users = rng.choice(np.arange(N_USERS), size=min(5, N_USERS), replace=False)

        val_losses = []
        roc_aucs = []

        with torch.no_grad():
            for user_id in sampled_users:
                user_data = load_user_data(
                    user_id=int(user_id),
                    data_dir=DATA_DIR,
                    window_size=WINDOW_SIZE,
                    stride=STRIDE,
                )
                x_val = torch.tensor(user_data['X_val'], dtype=torch.float32).permute(0, 2, 1)
                y_val = user_data['y_val']

                scores = model(x_val).cpu().numpy().reshape(-1)

                val_loss = float(
                    torch.nn.functional.binary_cross_entropy(
                        torch.tensor(scores, dtype=torch.float32),
                        torch.tensor(y_val, dtype=torch.float32),
                    ).item()
                )
                val_losses.append(val_loss)

                if np.unique(y_val).size < 2:
                    roc_aucs.append(float('nan'))
                else:
                    fpr, tpr, _ = roc_curve(y_val, scores)
                    roc_aucs.append(float(auc(fpr, tpr)))

                _ = compute_metrics(y_val, scores)

        rounds.append(round_idx)
        mean_val_losses.append(float(np.nanmean(val_losses)))
        valid_roc_aucs = [v for v in roc_aucs if np.isfinite(v)]
        if len(valid_roc_aucs) == 0:
            mean_roc_aucs.append(float('nan'))
        else:
            mean_roc_aucs.append(float(np.mean(valid_roc_aucs)))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(rounds, mean_val_losses, color='blue', marker='o', linewidth=2, label='Val Loss')
    line2 = ax2.plot(rounds, mean_roc_aucs, color='red', marker='s', linewidth=2, label='ROC_AUC')
    baseline = ax2.axhline(
        y=0.9990,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label='Centralized Baseline',
    )

    ax1.set_title('Federated Training Progress Over 20 Rounds')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Val Loss')
    ax2.set_ylabel('ROC-AUC')
    ax1.grid(alpha=0.25)

    handles = line1 + line2 + [baseline]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='best')

    save_path = os.path.join(plots_dir, 'federated_training_curves.png')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print('Training curves saved')


def plot_comparison_bars(comparison_df, plots_dir):
    c_auc = float(comparison_df['centralized_ROC_AUC'].mean())
    f_auc = float(comparison_df['federated_ROC_AUC'].mean())

    c_tpr = float(comparison_df['centralized_TPR'].mean())
    f_tpr = float(comparison_df['federated_TPR'].mean())

    c_fpr = float(comparison_df['centralized_FPR'].mean())
    f_fpr = float(comparison_df['federated_FPR'].mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    charts = [
        ('Mean ROC_AUC', c_auc, f_auc),
        ('Mean TPR', c_tpr, f_tpr),
        ('Mean FPR', c_fpr, f_fpr),
    ]

    for ax, (title, c_val, f_val) in zip(axes, charts):
        bars = ax.bar(
            ['Centralized', 'Federated'],
            [c_val, f_val],
            color=['steelblue', 'darkorange'],
            width=0.6,
        )
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.2)

        y_max = max(c_val, f_val)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(0.005, 0.02 * y_max),
                f'{h:.4f}',
                ha='center',
                va='bottom',
                fontsize=9,
            )

    axes[2].text(
        0.5,
        0.95,
        'Lower is better',
        transform=axes[2].transAxes,
        ha='center',
        va='top',
        fontsize=10,
        style='italic',
    )

    fig.suptitle('Centralized vs Federated — Key Metrics Comparison', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(plots_dir, 'centralized_vs_federated_bars.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print('Comparison bar chart saved')


def plot_per_user_delta(comparison_df, plots_dir):
    x = comparison_df['delta_TPR'].values
    y = comparison_df['delta_FPR'].values
    colors = np.where(comparison_df['delta_ROC_AUC'].values >= 0, 'green', 'red')

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(x, y, c=colors, alpha=0.8, edgecolors='black', linewidths=0.4)
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('delta_TPR (federated - centralized)')
    ax.set_ylabel('delta_FPR (federated - centralized)')
    ax.set_title('Per-User Impact of Federated Learning')
    ax.grid(alpha=0.2)

    ax.text(
        0.03,
        0.97,
        'Better',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=11,
        fontweight='bold',
    )

    save_path = os.path.join(plots_dir, 'per_user_delta.png')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print('Per-user delta scatter plot saved')


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print('WEEK 2 EVALUATION — FEDERATED vs CENTRALIZED')
    print(f"{'='*60}")

    print('\n[1/4] Evaluating federated model on 50 users...')
    fed_model = load_model(FEDERATED_MODEL_PATH)
    fed_metrics = evaluate_model_per_user(
        fed_model, DATA_DIR, N_USERS, WINDOW_SIZE, STRIDE
    )
    print('\nFEDERATED MODEL — PER USER RESULTS:')
    fed_summary = summarize_results(fed_metrics)
    plot_per_user_roc(fed_metrics, f'{PLOTS_DIR}/federated_roc.png')
    plot_metric_distributions(fed_metrics, f'{PLOTS_DIR}/federated_distributions.png')

    print('\n[2/4] Loading centralized vs federated comparison...')
    comparison_df = pd.read_csv('data/processed/centralized_vs_federated.csv')
    print(comparison_df.to_string(index=False))

    c_auc = comparison_df['centralized_ROC_AUC'].mean()
    f_auc = comparison_df['federated_ROC_AUC'].mean()
    delta_auc = f_auc - c_auc
    c_tpr = comparison_df['centralized_TPR'].mean()
    f_tpr = comparison_df['federated_TPR'].mean()
    delta_tpr = f_tpr - c_tpr
    c_fpr = comparison_df['centralized_FPR'].mean()
    f_fpr = comparison_df['federated_FPR'].mean()
    delta_fpr = f_fpr - c_fpr
    improved = int((comparison_df['delta_ROC_AUC'] >= 0).sum())

    print(f"{'='*60}")
    print('FINAL COMPARISON SUMMARY')
    print(f"{'='*60}")
    print('Metric         | Centralized | Federated  | Delta')
    print(f'Mean ROC_AUC   | {c_auc:.4f}   | {f_auc:.4f}  | {delta_auc:+.4f}')
    print(f'Mean TPR       | {c_tpr:.4f}   | {f_tpr:.4f}  | {delta_tpr:+.4f}')
    print(f'Mean FPR       | {c_fpr:.4f}   | {f_fpr:.4f}  | {delta_fpr:+.4f}')
    print(f'Users improved: {improved}/50')
    print(f"{'='*60}")

    print('\n[3/4] Generating comparison plots...')
    plot_training_curves(CHECKPOINT_DIR, PLOTS_DIR)
    plot_comparison_bars(comparison_df, PLOTS_DIR)
    plot_per_user_delta(comparison_df, PLOTS_DIR)

    print('\n[4/4] Logging to MLflow...')
    mlflow.set_experiment('federated_evaluation')
    with mlflow.start_run(run_name='week2_final_evaluation'):
        mlflow.log_metric('fed_mean_roc_auc', float(f_auc))
        mlflow.log_metric('fed_mean_tpr', float(f_tpr))
        mlflow.log_metric('fed_mean_fpr', float(f_fpr))
        mlflow.log_metric('delta_roc_auc', float(delta_auc))
        mlflow.log_metric('delta_tpr', float(delta_tpr))
        mlflow.log_metric('delta_fpr', float(delta_fpr))
        mlflow.log_metric('users_improved', float(improved))

        for file_name in os.listdir(PLOTS_DIR):
            if file_name.lower().endswith('.png'):
                mlflow.log_artifact(os.path.join(PLOTS_DIR, file_name))

        mlflow.log_artifact('data/processed/centralized_vs_federated.csv')

    _ = fed_summary
    print('\n✅ Week 2 complete!')
    print(f'Plots saved to: {PLOTS_DIR}/')
    print('Results saved to: data/processed/centralized_vs_federated.csv')
    print('Next step: Run Week 3 personalization')


if __name__ == '__main__':
    main()
