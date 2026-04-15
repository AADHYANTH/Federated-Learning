import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import flwr as fl
from flwr.common import Context
from src.data.preprocessing import load_user_data
from src.data.datasets import TimeSeriesDataset, create_dataloader, get_class_weights
from src.models.base_cnn import CNNAnomalyDetector


class AnomalyDetectionClient(fl.client.NumPyClient):
    def __init__(
        self,
        user_id,
        data_dir="data/raw/synthetic",
        window_size=128,
        stride=64,
        batch_size=32,
        local_epochs=3,
        device="cpu",
    ):
        user_data = load_user_data(user_id, data_dir, window_size, stride)
        X_train = user_data["X_train"]
        y_train = user_data["y_train"]
        X_val = user_data["X_val"]
        y_val = user_data["y_val"]

        self.train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
        self.val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)

        self.pos_weight = get_class_weights(y_train).float()
        self.n_train = len(X_train)
        self.n_val = len(X_val)

        self.model = CNNAnomalyDetector(in_channels=1, window_size=128)
        self.local_epochs = local_epochs
        self.device = torch.device(device)
        self.user_id = int(user_id)
        self.model.to(self.device)

        print(
            f"Client {self.user_id:03d} ready | "
            f"train={len(X_train)} val={len(X_val)} "
            f"pos_weight={float(self.pos_weight.item()):.2f}"
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: list) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = float(config.get("lr", 1e-3))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        epoch_losses = []
        device = self.device

        for _ in range(self.local_epochs):
            batch_losses = []
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = self.model(X_batch)
                weight_tensor = torch.where(
                    y_batch == 1,
                    self.pos_weight.to(device),
                    torch.ones_like(y_batch),
                )
                loss = F.binary_cross_entropy(preds, y_batch, weight=weight_tensor)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses))

        mean_train_loss = float(np.mean(epoch_losses))
        return self.get_parameters(config={}), self.n_train, {"train_loss": mean_train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        all_preds, all_labels = [], []
        device = self.device

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = self.model(X_batch)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        val_loss = float(
            F.binary_cross_entropy(
                torch.tensor(all_preds),
                torch.tensor(all_labels),
            ).item()
        )

        if len(np.unique(all_labels)) < 2:
            roc_auc = 0.0
            print(f"Warning: Client {self.user_id} val set has no positives")
        else:
            roc_auc = float(roc_auc_score(all_labels, all_preds))

        return float(val_loss), self.n_val, {"val_loss": val_loss, "roc_auc": roc_auc}


def client_fn(context: Context) -> fl.client.Client:
    user_id = int(context.node_config["partition-id"])
    return AnomalyDetectionClient(
        user_id=user_id,
        data_dir="data/raw/synthetic",
        window_size=128,
        stride=64,
        batch_size=32,
        local_epochs=3,
        device="cpu",
    ).to_client()


if __name__ == "__main__":
    c0 = AnomalyDetectionClient(
        user_id=0,
        data_dir="data/raw/synthetic",
        window_size=128,
        stride=64,
        batch_size=32,
        local_epochs=3,
        device="cpu",
    )
    c1 = AnomalyDetectionClient(
        user_id=1,
        data_dir="data/raw/synthetic",
        window_size=128,
        stride=64,
        batch_size=32,
        local_epochs=3,
        device="cpu",
    )

    params = c0.get_parameters(config={})
    print(f"Parameter arrays: {len(params)}")

    updated_params, _, fit_metrics = c0.fit(params, config={"lr": 1e-3})
    print(f"Train loss: {fit_metrics['train_loss']:.6f}")

    val_loss, _, eval_metrics = c0.evaluate(updated_params, config={})
    print(f"Val loss: {val_loss:.6f}, roc_auc: {eval_metrics['roc_auc']:.6f}")

    print("Client test passed!")
