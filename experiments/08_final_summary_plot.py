import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import onnxruntime as ort
from sklearn.metrics import roc_auc_score

from src.models.base_cnn import CNNAnomalyDetector
from src.data.preprocessing import load_user_data
from src.data.ecg_loader import load_mitbih, split_into_federated_users, normalize_ecg_users


DATA_DIR = 'data/raw/synthetic'
TRAIN_PATH = 'data/raw/ecg/mitbih_train.csv'
TEST_PATH = 'data/raw/ecg/mitbih_test.csv'

SYNTH_CKPT_DIR = 'data/processed/checkpoints'
ECG_CKPT_DIR = 'data/processed/ecg_checkpoints'

SYNTH_MODEL_PATH = 'data/processed/global_model_federated.pt'
ECG_MODEL_PATH = 'data/processed/ecg_model_federated.pt'
SYNTH_ONNX_PATH = 'data/processed/model_synth_edge.onnx'
ECG_ONNX_PATH = 'data/processed/model_ecg_edge.onnx'

ECG_RESULTS_PATH = 'data/processed/ecg_personalization_results.csv'
SYNTH_VS_ECG_PATH = 'data/processed/synthetic_vs_ecg_comparison.csv'

PLOT_PATH = 'data/processed/plots/FINAL_SUMMARY.png'

WINDOW_SIZE_SYNTH = 128
WINDOW_SIZE_ECG = 187
STRIDE = 64
N_USERS_SYNTH = 50
N_USERS_ECG = 47

SEED = 42


def _load_model(path, window_size):
    m = CNNAnomalyDetector(in_channels=1, window_size=window_size)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


def _sorted_checkpoints(folder):
    ckpts = glob.glob(os.path.join(folder, 'round_*.pt'))
    ckpts = sorted(
        ckpts,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1])
    )
    return ckpts


def _build_synth_val_cache(sample_users):
    cache = []
    for uid in sample_users:
        d = load_user_data(uid, DATA_DIR, window_size=WINDOW_SIZE_SYNTH, stride=STRIDE)
        y = d['y_val'].astype(np.int32)
        if np.unique(y).size < 2:
            continue
        x = torch.tensor(d['X_val'], dtype=torch.float32).permute(0, 2, 1)
        cache.append((x, y))
    if not cache:
        raise ValueError('No synthetic validation users with both classes found')
    return cache


def _build_ecg_val_cache():
    x_train, y_train, x_test, y_test = load_mitbih(TRAIN_PATH, TEST_PATH)
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    users = split_into_federated_users(x_all, y_all, n_users=N_USERS_ECG, seed=SEED)
    users = normalize_ecg_users(users)

    cache = []
    for user in users:
        y = user['y_val'].astype(np.int32)
        if np.unique(y).size < 2:
            continue
        x = torch.tensor(user['X_val'], dtype=torch.float32)
        cache.append((x, y))
    if not cache:
        raise ValueError('No ECG validation users with both classes found')
    return cache


def _curve_from_checkpoints(checkpoint_dir, window_size, val_cache):
    rounds = []
    aucs = []

    ckpts = _sorted_checkpoints(checkpoint_dir)
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {checkpoint_dir}')

    for ckpt in ckpts:
        round_idx = int(os.path.splitext(os.path.basename(ckpt))[0].split('_')[-1])
        model = _load_model(ckpt, window_size=window_size)

        per_user_auc = []
        with torch.no_grad():
            for x_val, y_val in val_cache:
                scores = model(x_val).cpu().numpy().reshape(-1)
                if np.unique(y_val).size > 1:
                    per_user_auc.append(float(roc_auc_score(y_val, scores)))

        rounds.append(round_idx)
        aucs.append(float(np.mean(per_user_auc)) if per_user_auc else float('nan'))

    return np.array(rounds), np.array(aucs)


def _ensure_onnx(model_path, onnx_path, window_size):
    if os.path.exists(onnx_path):
        return onnx_path

    model = _load_model(model_path, window_size=window_size)
    dummy = torch.randn(1, 1, window_size)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['signal_window'],
        output_names=['anomaly_score'],
        dynamic_axes={
            'signal_window': {0: 'batch_size'},
            'anomaly_score': {0: 'batch_size'}
        }
    )
    return onnx_path


def _latency_distribution(onnx_path, window_size, n_runs=1000):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    for _ in range(50):
        dummy = np.random.randn(1, 1, window_size).astype(np.float32)
        session.run(None, {input_name: dummy})

    lat = []
    for _ in range(n_runs):
        dummy = np.random.randn(1, 1, window_size).astype(np.float32)
        t0 = torch.cuda.Event(enable_timing=False) if torch.cuda.is_available() else None
        _ = t0
        start = torch.utils.benchmark.Timer(stmt='0').timeit(1).mean if False else None
        _ = start
        t_start = __import__('time').perf_counter()
        session.run(None, {input_name: dummy})
        t_end = __import__('time').perf_counter()
        lat.append((t_end - t_start) * 1000.0)

    return np.array(lat, dtype=np.float64)


def main():
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Subplot 1 values are fixed from final report.
    stages = ['Centralized', 'Federated', 'Pers-Synthetic', 'Pers-ECG']
    stage_auc = [0.9990, 0.9993, 0.9993, 0.9908]

    # Subplot 2 data.
    ecg_df = pd.read_csv(ECG_RESULTS_PATH).sort_values('user_id').reset_index(drop=True)
    if 'delta_TPR' not in ecg_df.columns:
        raise ValueError('ecg_personalization_results.csv must contain delta_TPR')

    # Subplot 3 data from checkpoints.
    synth_users = rng.choice(np.arange(N_USERS_SYNTH), size=min(10, N_USERS_SYNTH), replace=False)
    synth_cache = _build_synth_val_cache(synth_users)
    ecg_cache = _build_ecg_val_cache()

    synth_rounds, synth_aucs = _curve_from_checkpoints(SYNTH_CKPT_DIR, WINDOW_SIZE_SYNTH, synth_cache)
    ecg_rounds, ecg_aucs = _curve_from_checkpoints(ECG_CKPT_DIR, WINDOW_SIZE_ECG, ecg_cache)

    synth_baseline = 0.9990
    ecg_baseline = 0.9887
    if os.path.exists(SYNTH_VS_ECG_PATH):
        comp = pd.read_csv(SYNTH_VS_ECG_PATH)
        if 'real_ecg_centralized_roc_auc' in comp.columns and len(comp) > 0:
            ecg_baseline = float(comp.loc[0, 'real_ecg_centralized_roc_auc'])

    # Subplot 4 data from ONNX runtime latency.
    _ensure_onnx(SYNTH_MODEL_PATH, SYNTH_ONNX_PATH, WINDOW_SIZE_SYNTH)
    _ensure_onnx(ECG_MODEL_PATH, ECG_ONNX_PATH, WINDOW_SIZE_ECG)

    synth_lat = _latency_distribution(SYNTH_ONNX_PATH, WINDOW_SIZE_SYNTH, n_runs=1000)
    ecg_lat = _latency_distribution(ECG_ONNX_PATH, WINDOW_SIZE_ECG, n_runs=1000)

    synth_mean = float(synth_lat.mean())
    ecg_mean = float(ecg_lat.mean())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
    fig.suptitle(
        'Federated Personalization for Health Anomaly Detection\n'
        '— Complete Pipeline Results',
        fontsize=16,
        fontweight='bold'
    )

    # Subplot 1: pipeline progression ROC-AUC
    ax = axes[0, 0]
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, 4))
    bars = ax.bar(stages, stage_auc, color=colors)
    ax.set_ylim(0.96, 1.001)
    ax.set_title('ROC-AUC Across Pipeline Stages')
    ax.set_ylabel('ROC-AUC')
    ax.grid(axis='y', alpha=0.25)
    for bar, v in zip(bars, stage_auc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0004, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    # Subplot 2: per-user delta TPR on ECG
    ax = axes[0, 1]
    user_ids = ecg_df['user_id'].to_numpy(dtype=int)
    delta_tpr = ecg_df['delta_TPR'].to_numpy(dtype=float)
    bar_colors = np.where(delta_tpr > 0, '#2ca02c', '#d62728')
    ax.bar(user_ids, delta_tpr, color=bar_colors, width=0.8)
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_title('Per-User TPR Change After Personalization (ECG)')
    ax.set_xlabel('User ID')
    ax.set_ylabel('Delta TPR')
    ax.grid(axis='y', alpha=0.25)

    # Subplot 3: convergence curves
    ax = axes[1, 0]
    ax.plot(synth_rounds, synth_aucs, color='tab:blue', marker='o', linewidth=2, label='Synthetic')
    ax.plot(ecg_rounds, ecg_aucs, color='tab:orange', marker='s', linewidth=2, label='ECG')
    ax.axhline(synth_baseline, color='tab:blue', linestyle='--', linewidth=1.2, alpha=0.8, label='Synthetic baseline')
    ax.axhline(ecg_baseline, color='tab:orange', linestyle='--', linewidth=1.2, alpha=0.8, label='ECG baseline')
    ax.set_title('Federated Training Convergence')
    ax.set_xlabel('Round')
    ax.set_ylabel('ROC-AUC')
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right', fontsize=8)

    # Subplot 4: edge latency histogram
    ax = axes[1, 1]
    bins = np.linspace(0.0, 0.15, 45)
    ax.hist(synth_lat, bins=bins, alpha=0.6, color='tab:blue', label='Synthetic')
    ax.hist(ecg_lat, bins=bins, alpha=0.6, color='tab:orange', label='ECG')
    ax.axvline(50.0, color='red', linestyle='--', linewidth=1.4, label='50ms target')
    ax.axvline(synth_mean, color='tab:blue', linestyle='-', linewidth=1.5, label=f'Synthetic mean={synth_mean:.3f}ms')
    ax.axvline(ecg_mean, color='tab:orange', linestyle='-', linewidth=1.5, label=f'ECG mean={ecg_mean:.3f}ms')
    ax.set_xlim(0.0, 0.15)
    ax.set_title('Edge Inference Latency (CPU, 1000 runs)')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    print('Final summary plot saved — use this in your README and LinkedIn')


if __name__ == '__main__':
    main()
