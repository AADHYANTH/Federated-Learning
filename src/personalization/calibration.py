import numpy as np
import os


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class ScoreCalibrator:
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.loss_history = []
        self.a_i = self.a
        self.b_i = self.b

    def fit(self, base_scores, labels, lr=0.01, epochs=200):
        base_scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.float64).reshape(-1)

        if base_scores.shape[0] != labels.shape[0]:
            raise ValueError("base_scores and labels must have matching length")
        if base_scores.size == 0:
            raise ValueError("base_scores and labels must not be empty")

        self.loss_history = []
        last_epoch = -1
        loss = float("nan")

        for epoch in range(int(epochs)):
            p = sigmoid(self.a * base_scores + self.b)
            loss = bce_loss(labels, p)
            self.loss_history.append(loss)

            error = p - labels
            da = np.mean(error * base_scores)
            db = np.mean(error)

            self.a -= lr * da
            self.b -= lr * db
            self.a_i = self.a
            self.b_i = self.b

            last_epoch = epoch

            if epoch > 10 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                break

        if last_epoch == -1:
            p = sigmoid(self.a * base_scores + self.b)
            loss = bce_loss(labels, p)

        epochs_run = max(last_epoch + 1, 0)
        print(f"  Calibration done: a={self.a:.4f} b={self.b:.4f} loss={loss:.4f} epochs={epochs_run}")

    def predict(self, base_scores):
        base_scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
        return sigmoid(self.a * base_scores + self.b)

    def save(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.save(path, {"a": self.a, "b": self.b})

    def load(self, path):
        data = np.load(path, allow_pickle=True).item()
        self.a = data["a"]
        self.b = data["b"]
        self.a_i = self.a
        self.b_i = self.b
