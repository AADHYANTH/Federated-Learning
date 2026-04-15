import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocessing import load_user_data
from src.models.autoencoder import AEAnomalyDetector
from src.models.base_cnn import CNNAnomalyDetector


DATA_DIR = 'data/raw/synthetic'
N_USERS = 50
WINDOW_SIZE = 128
STRIDE = 64
BATCH_SIZE = 128
AE_EPOCHS = 10
AE_LR = 1e-3
CNN_MODEL_PATH = 'data/processed/global_model_federated.pt'
AE_MODEL_PATH = 'data/processed/autoencoder_model.pt'
OUT_PATH = 'data/processed/cnn_vs_autoencoder.csv'
SEED = 42


def collect_normal_train_windows():
    all_normal = []
    for user_id in range(N_USERS):
        d = load_user_data(user_id, DATA_DIR, window_size=WINDOW_SIZE, stride=STRIDE)
        x_train = d['X_train']
        y_train = d['y_train']
        normal_windows = x_train[y_train == 0]
        if len(normal_windows) > 0:
            x_t = torch.tensor(normal_windows, dtype=torch.float32).permute(0, 2, 1)
            all_normal.append(x_t)

    if len(all_normal) == 0:
        raise ValueError('No normal training windows found for AE training')

    return torch.cat(all_normal, dim=0)


def train_autoencoder(x_normal):
    print('\n[AE Training] Unsupervised on normal windows only')

    model = AEAnomalyDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    ds = TensorDataset(x_normal, x_normal)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(AE_EPOCHS):
        losses = []
        for xb, target in loader:
            optimizer.zero_grad()
            x_hat = model(xb)
            loss = criterion(x_hat, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f'  Epoch {epoch + 1:02d}/{AE_EPOCHS} recon_loss={avg_loss:.6f}')

    torch.save(model.state_dict(), AE_MODEL_PATH)
    print(f'AE model saved: {AE_MODEL_PATH}')
    return model


def load_cnn_model():
    model = CNNAnomalyDetector(in_channels=1, window_size=WINDOW_SIZE)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
    model.eval()
    return model


def evaluate_models(cnn_model, ae_model):
    print('\n[Evaluation] CNN vs Autoencoder on 50 users')

    cnn_model.eval()
    ae_model.eval()

    rows = []
    with torch.no_grad():
        for user_id in range(N_USERS):
            d = load_user_data(user_id, DATA_DIR, window_size=WINDOW_SIZE, stride=STRIDE)
            x_test = torch.tensor(d['X_test'], dtype=torch.float32).permute(0, 2, 1)
            y_test = d['y_test'].astype(np.int32)

            cnn_scores = cnn_model(x_test).cpu().numpy().reshape(-1)
            ae_scores = ae_model.anomaly_score(x_test).cpu().numpy().reshape(-1)

            if np.unique(y_test).size > 1:
                cnn_auc = float(roc_auc_score(y_test, cnn_scores))
                ae_auc = float(roc_auc_score(y_test, ae_scores))
            else:
                cnn_auc = float('nan')
                ae_auc = float('nan')

            rows.append({
                'user_id': user_id,
                'cnn_roc_auc': cnn_auc,
                'ae_roc_auc': ae_auc,
                'delta_cnn_minus_ae': cnn_auc - ae_auc if np.isfinite(cnn_auc) and np.isfinite(ae_auc) else float('nan')
            })

    df = pd.DataFrame(rows)
    mean_cnn = float(np.nanmean(df['cnn_roc_auc']))
    mean_ae = float(np.nanmean(df['ae_roc_auc']))

    print('=' * 64)
    print('CNN vs AUTOENCODER (ROC-AUC)')
    print('=' * 64)
    print('Model         | Mean ROC-AUC')
    print('-' * 64)
    print(f'CNN (labeled) | {mean_cnn:.4f}')
    print(f'AE (unsup.)   | {mean_ae:.4f}')
    print('-' * 64)

    if mean_cnn >= mean_ae:
        print('CNN performs better on labeled data.')
    else:
        print('AE outperformed CNN in this run.')
    print('AE remains useful when labels are unavailable.')

    summary = pd.DataFrame([{
        'user_id': 'MEAN',
        'cnn_roc_auc': mean_cnn,
        'ae_roc_auc': mean_ae,
        'delta_cnn_minus_ae': mean_cnn - mean_ae,
    }])
    out_df = pd.concat([df, summary], ignore_index=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f'Comparison saved: {OUT_PATH}')


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_normal = collect_normal_train_windows()
    ae_model = train_autoencoder(x_normal)
    cnn_model = load_cnn_model()
    evaluate_models(cnn_model, ae_model)


if __name__ == '__main__':
    main()
