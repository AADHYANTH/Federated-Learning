"""Autoencoder model for unsupervised time-series anomaly detection."""

import torch
import torch.nn as nn


class AEAnomalyDetector(nn.Module):
    """Conv1D autoencoder that reconstructs windows of shape (batch, 1, 128)."""

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.enc_pool1 = nn.MaxPool1d(2)
        self.enc_conv2 = nn.Conv1d(32, 16, kernel_size=5, padding=2)
        self.enc_pool2 = nn.MaxPool1d(2)
        self.enc_conv3 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.enc_gap = nn.AdaptiveAvgPool1d(4)

        self.enc_fc1 = nn.Linear(32, 16)
        self.enc_fc2 = nn.Linear(16, 8)

        # Decoder
        self.dec_fc1 = nn.Linear(8, 16)
        self.dec_fc2 = nn.Linear(16, 32)

        # 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.dec_deconv1 = nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1)
        self.dec_deconv2 = nn.ConvTranspose1d(16, 32, kernel_size=4, stride=2, padding=1)
        self.dec_deconv3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec_deconv4 = nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1)
        self.dec_deconv5 = nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def encode(self, x):
        x = self.relu(self.enc_conv1(x))
        x = self.enc_pool1(x)
        x = self.relu(self.enc_conv2(x))
        x = self.enc_pool2(x)
        x = self.relu(self.enc_conv3(x))
        x = self.enc_gap(x)

        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.enc_fc1(x))
        z = self.enc_fc2(x)
        return z

    def decode(self, z):
        x = self.relu(self.dec_fc1(z))
        x = self.dec_fc2(x)
        x = x.view(-1, 8, 4)

        x = self.relu(self.dec_deconv1(x))
        x = self.relu(self.dec_deconv2(x))
        x = self.relu(self.dec_deconv3(x))
        x = self.relu(self.dec_deconv4(x))
        x_hat = self.dec_deconv5(x)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def anomaly_score(self, x):
        x_hat = self.forward(x)
        reconstruction_error = ((x - x_hat) ** 2).mean(dim=[1, 2])
        score = torch.sigmoid(reconstruction_error * 10 - 5)
        return score
