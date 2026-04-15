import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.models.base_cnn import CNNAnomalyDetector


class PersonalizedHead(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.fc(features).squeeze(1)


def extract_features(encoder, X_windows, device='cpu'):
    encoder.eval()
    encoder.to(device)

    X_windows = np.asarray(X_windows, dtype=np.float32)
    if X_windows.ndim != 3:
        raise ValueError('X_windows must have shape (N, W, C)')

    if X_windows.shape[0] == 0:
        return np.empty((0, 128), dtype=np.float32)

    x_tensor = torch.tensor(X_windows, dtype=torch.float32).permute(0, 2, 1)
    all_features = []

    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], 64):
            batch = x_tensor[start:start + 64].to(device)
            feats = encoder.encode(batch)
            all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def fit_personalized_head(encoder, X_train, y_train,
                          X_val, y_val,
                          feature_dim=128,
                          epochs=100, lr=1e-3,
                          device='cpu'):
    for param in encoder.parameters():
        param.requires_grad = False

    train_features = extract_features(encoder, X_train, device=device)
    val_features = extract_features(encoder, X_val, device=device)

    X_train_feat = torch.tensor(train_features, dtype=torch.float32)
    y_train_t = torch.tensor(np.asarray(y_train).reshape(-1), dtype=torch.float32)
    X_val_feat = torch.tensor(val_features, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(np.asarray(y_val).reshape(-1), dtype=torch.float32, device=device)

    if X_train_feat.shape[0] == 0 or X_val_feat.shape[0] == 0:
        raise ValueError('Train/val features must be non-empty for personalized head training')

    pos_weight = (np.asarray(y_train) == 0).sum() / max((np.asarray(y_train) == 1).sum(), 1)

    head = PersonalizedHead(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

    batch_size = int(min(64, max(1, X_train_feat.shape[0])))
    train_loader = DataLoader(
        TensorDataset(X_train_feat, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_loss = float('inf')
    best_state = None
    val_losses = []
    patience = 12
    patience_counter = 0

    for epoch in range(int(epochs)):
        head.train()
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = head(batch_x)
            weight_tensor = torch.where(batch_y == 1, pos_weight_tensor, torch.ones_like(batch_y))
            loss = F.binary_cross_entropy(preds, batch_y, weight=weight_tensor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_preds = head(X_val_feat)
            val_loss = F.binary_cross_entropy(val_preds, y_val_t).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch > 20 and patience_counter >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    return head, val_losses
