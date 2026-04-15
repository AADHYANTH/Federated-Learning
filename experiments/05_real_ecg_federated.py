"""Real ECG federated training + personalization experiment (MIT-BIH)."""

from __future__ import annotations

import glob
import os
import shutil
import sys
from collections import OrderedDict

import flwr as fl
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


TRAIN_CSV = "data/raw/ecg/mitbih_train.csv"
TEST_CSV = "data/raw/ecg/mitbih_test.csv"
N_USERS = 47
N_ROUNDS = 20
FRACTION_FIT = 0.4
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
WINDOW_SIZE = 187
CHECKPOINT_DIR = "data/processed/checkpoints_real_ecg"
FINAL_MODEL_PATH = "data/processed/global_model_federated_real_ecg.pt"
REAL_RESULTS_PATH = "data/processed/personalization_results_real_ecg.csv"
COMPARISON_PATH = "data/processed/real_vs_synthetic_comparison.csv"
TARGET_FPR = 0.05


try:
    from src.data.datasets import get_class_weights
    from src.data.ecg_loader import load_mitbih, split_by_patient_simulated
    from src.evaluation.metrics import compute_metrics
    from src.federated.strategy import FedAvgWithLogging
    from src.models.base_cnn import CNNAnomalyDetector
    from src.personalization.calibration import ScoreCalibrator
    from src.personalization.thresholds import ThresholdSelector
except ModuleNotFoundError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.data.datasets import get_class_weights
    from src.data.ecg_loader import load_mitbih, split_by_patient_simulated
    from src.evaluation.metrics import compute_metrics
    from src.federated.strategy import FedAvgWithLogging
    from src.models.base_cnn import CNNAnomalyDetector
    from src.personalization.calibration import ScoreCalibrator
    from src.personalization.thresholds import ThresholdSelector


def _weighted_bce_loss(probs, targets, pos_weight):
    """Compute weighted BCE loss using probability outputs."""
    eps = 1e-8
    probs = torch.clamp(probs, eps, 1.0 - eps)
    loss_pos = -pos_weight * targets * torch.log(probs)
    loss_neg = -(1.0 - targets) * torch.log(1.0 - probs)
    return (loss_pos + loss_neg).mean()


class ECGFederatedClient(fl.client.NumPyClient):
    """Flower NumPyClient for one simulated ECG federated user."""

    def __init__(self, user_data, batch_size=32, local_epochs=3, device="cpu"):
        self.user_id = int(user_data["user_id"])
        self.X_train = user_data["X_train"].astype(np.float32)
        self.y_train = user_data["y_train"].astype(np.float32)
        self.X_val = user_data["X_val"].astype(np.float32)
        self.y_val = user_data["y_val"].astype(np.float32)
        self.batch_size = int(batch_size)
        self.local_epochs = int(local_epochs)
        self.device = torch.device(device)

        train_ds = TensorDataset(
            torch.as_tensor(self.X_train, dtype=torch.float32),
            torch.as_tensor(self.y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.as_tensor(self.X_val, dtype=torch.float32),
            torch.as_tensor(self.y_val, dtype=torch.float32),
        )
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        self.pos_weight = get_class_weights(self.y_train).to(self.device)
        self.model = CNNAnomalyDetector(in_channels=1, window_size=WINDOW_SIZE).to(self.device)

    def get_parameters(self, config):
        """Return current model parameters as NumPy arrays."""
        del config
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load model parameters from NumPy arrays."""
        state_dict = self.model.state_dict()
        keys = list(state_dict.keys())
        new_state = OrderedDict()
        for k, p in zip(keys, parameters):
            new_state[k] = torch.tensor(p, device=self.device)
        self.model.load_state_dict(new_state, strict=True)

    def fit(self, parameters, config):
        """Run local training and return updated parameters and train metrics."""
        self.set_parameters(parameters)
        lr = float(config.get("lr", 1e-3))
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        epoch_losses = []
        for _ in range(self.local_epochs):
            total_loss = 0.0
            total_n = 0
            for xb, yb in self.train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                probs = self.model(xb)
                loss = _weighted_bce_loss(probs, yb, self.pos_weight)
                loss.backward()
                optimizer.step()
                n = int(xb.shape[0])
                total_loss += float(loss.item()) * n
                total_n += n
            epoch_losses.append(total_loss / max(total_n, 1))

        return (
            self.get_parameters(config={}),
            int(self.X_train.shape[0]),
            {"train_loss": float(np.mean(epoch_losses))},
        )

    def evaluate(self, parameters, config):
        """Run local validation and return loss plus ROC_AUC metric."""
        del config
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        total_n = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                probs = self.model(xb)
                loss = _weighted_bce_loss(probs, yb, self.pos_weight)
                n = int(xb.shape[0])
                total_loss += float(loss.item()) * n
                total_n += n
                all_probs.append(probs.detach().cpu().numpy())
                all_labels.append(yb.detach().cpu().numpy())

        val_loss = total_loss / max(total_n, 1)
        y_scores = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
        y_true = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

        if y_true.size == 0 or np.unique(y_true).size < 2:
            roc_auc = 0.0
        else:
            roc_auc = float(roc_auc_score(y_true, y_scores))

        return float(val_loss), int(self.X_val.shape[0]), {
            "val_loss": float(val_loss),
            "roc_auc": float(roc_auc),
        }


def _make_client_fn(user_splits):
    """Create Flower client_fn bound to pre-split user datasets."""

    def client_fn(context):
        if hasattr(context, "node_config"):
            user_id = int(context.node_config.get("partition-id", 0))
        else:
            user_id = int(context)
        client = ECGFederatedClient(
            user_data=user_splits[user_id],
            batch_size=BATCH_SIZE,
            local_epochs=LOCAL_EPOCHS,
            device="cpu",
        )
        return client.to_client()

    return client_fn


def _build_initial_parameters(model):
    """Convert PyTorch model parameters to Flower Parameters."""
    ndarrays = [v.detach().cpu().numpy() for _, v in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(ndarrays)


def _predict_probs(model, X, device):
    """Predict anomaly probabilities for input windows shaped (N, 1, 187)."""
    x = torch.as_tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(x).detach().cpu().numpy().reshape(-1)
    return probs


def _load_synthetic_summary():
    """Load synthetic personalization summary from existing result CSV if found."""
    candidates = [
        "data/processed/personalization_results.csv",
        "data/processed/personalization_results_smoke.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)

            if "improved" in df.columns:
                improved_users_pct = float(100.0 * df["improved"].mean())
            elif "delta_TPR" in df.columns and "delta_FPR" in df.columns:
                improved_users_pct = float(
                    100.0 * (((df["delta_TPR"] > 0.0) & (df["delta_FPR"] <= 0.0)).mean())
                )
            else:
                improved_users_pct = float("nan")

            return {
                "dataset": "synthetic",
                "mean_delta_TPR": float(df["delta_TPR"].mean()),
                "mean_delta_FPR": float(df["delta_FPR"].mean()),
                "improved_users_pct": improved_users_pct,
                "n_users": int(df.shape[0]),
                "source": path,
            }
    return {
        "dataset": "synthetic",
        "mean_delta_TPR": float("nan"),
        "mean_delta_FPR": float("nan"),
        "improved_users_pct": float("nan"),
        "n_users": 0,
        "source": "not_found",
    }


def run_real_ecg_federated():
    """Run federated training + personalization on MIT-BIH and compare results."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)

    X_train, y_train, X_test, y_test = load_mitbih(TRAIN_CSV, TEST_CSV)
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    user_splits = split_by_patient_simulated(X_all, y_all, n_users=N_USERS, seed=42)

    base_model = CNNAnomalyDetector(in_channels=1, window_size=WINDOW_SIZE)
    initial_parameters = _build_initial_parameters(base_model)

    strategy = FedAvgWithLogging(
        initial_parameters=initial_parameters,
        n_rounds=N_ROUNDS,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    client_fn = _make_client_fn(user_splits)

    mlflow.set_experiment("real_ecg_federated")
    with mlflow.start_run(run_name="mitbih_fed_plus_personalization"):
        mlflow.log_params(
            {
                "train_csv": TRAIN_CSV,
                "test_csv": TEST_CSV,
                "n_users": N_USERS,
                "n_rounds": N_ROUNDS,
                "fraction_fit": FRACTION_FIT,
                "local_epochs": LOCAL_EPOCHS,
                "batch_size": BATCH_SIZE,
                "window_size": WINDOW_SIZE,
                "checkpoint_dir": CHECKPOINT_DIR,
                "final_model_path": FINAL_MODEL_PATH,
            }
        )

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=N_USERS,
            config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        strategy.print_final_summary()

        final_round_path = os.path.join(CHECKPOINT_DIR, f"round_{N_ROUNDS:03d}.pt")
        if os.path.exists(final_round_path):
            selected_checkpoint = final_round_path
        else:
            ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "round_*.pt"))
            if not ckpts:
                raise FileNotFoundError("No checkpoints found after real ECG training")
            selected_checkpoint = sorted(
                ckpts,
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]),
            )[-1]
        shutil.copyfile(selected_checkpoint, FINAL_MODEL_PATH)
        mlflow.log_artifact(FINAL_MODEL_PATH)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNAnomalyDetector(in_channels=1, window_size=WINDOW_SIZE).to(device)
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
        model.eval()

        selector = ThresholdSelector()
        rows = []

        for u in user_splits:
            uid = int(u["user_id"])
            X_val = u["X_val"]
            y_val = u["y_val"]
            X_test_u = u["X_test"]
            y_test_u = u["y_test"]

            val_scores = _predict_probs(model, X_val, device)
            test_scores = _predict_probs(model, X_test_u, device)

            calibrator = ScoreCalibrator()
            calibrator.fit(val_scores, y_val, lr=0.01, epochs=100)
            val_cal = calibrator.predict(val_scores)
            thr_info = selector.find_threshold(val_cal, y_val, target_fpr=TARGET_FPR)
            thr = float(thr_info["threshold"])
            val_fpr = float(thr_info["achieved_fpr"])
            val_tpr = float(thr_info["achieved_tpr"])

            global_m = compute_metrics(y_test_u, test_scores, threshold=0.5)
            test_cal = calibrator.predict(test_scores)
            pers_m = selector.evaluate_at_threshold(test_cal, y_test_u, threshold=thr)

            delta_tpr = float(pers_m["TPR"] - global_m["TPR"])
            delta_fpr = float(pers_m["FPR"] - global_m["FPR"])
            improved = int((delta_tpr > 0.0) and (delta_fpr <= 0.0))

            rows.append(
                {
                    "user_id": uid,
                    "global_TPR": float(global_m["TPR"]),
                    "global_FPR": float(global_m["FPR"]),
                    "global_FNR": float(global_m["FNR"]),
                    "global_ROC_AUC": float(global_m["ROC_AUC"]),
                    "global_PR_AUC": float(global_m["PR_AUC"]),
                    "personalized_TPR": float(pers_m["TPR"]),
                    "personalized_FPR": float(pers_m["FPR"]),
                    "personalized_FNR": float(pers_m["FNR"]),
                    "threshold": float(thr),
                    "val_target_achieved_FPR": float(val_fpr),
                    "val_target_achieved_TPR": float(val_tpr),
                    "delta_TPR": delta_tpr,
                    "delta_FPR": delta_fpr,
                    "improved": improved,
                }
            )

            mlflow.log_metric("real_delta_TPR", delta_tpr, step=uid)
            mlflow.log_metric("real_delta_FPR", delta_fpr, step=uid)

        real_df = pd.DataFrame(rows)
        real_df.to_csv(REAL_RESULTS_PATH, index=False)
        mlflow.log_artifact(REAL_RESULTS_PATH)

        real_summary = {
            "dataset": "real_ecg",
            "mean_delta_TPR": float(real_df["delta_TPR"].mean()),
            "mean_delta_FPR": float(real_df["delta_FPR"].mean()),
            "improved_users_pct": float(100.0 * real_df["improved"].mean()),
            "n_users": int(real_df.shape[0]),
            "source": REAL_RESULTS_PATH,
        }
        synthetic_summary = _load_synthetic_summary()
        comparison_df = pd.DataFrame([synthetic_summary, real_summary])
        comparison_df.to_csv(COMPARISON_PATH, index=False)
        mlflow.log_artifact(COMPARISON_PATH)

        mlflow.log_metric("real_mean_delta_TPR", real_summary["mean_delta_TPR"])
        mlflow.log_metric("real_mean_delta_FPR", real_summary["mean_delta_FPR"])
        mlflow.log_metric("real_improved_users_pct", real_summary["improved_users_pct"])

    print("Real ECG federated training + personalization complete.")
    print(f"Saved real personalization results to: {REAL_RESULTS_PATH}")
    print(f"Saved real-vs-synthetic comparison to: {COMPARISON_PATH}")


if __name__ == "__main__":
    run_real_ecg_federated()
