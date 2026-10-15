#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:11:50 2026

@author: qbruant
"""

"""
LSTM on the latent sequence produced by the CNN script (CNN_from_BTV).

Input : latents.pt, a tensor (N, LATENT) saved by extract_latents() in the CNN script,
        one row per image, in temporal order.
The latent size is read from the file, so this script does not import anything from the CNN script.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ Loading
def load_latents(path="latents.pt"):
    """Loads the latent file saved by extract_latents() and checks it. Returns a (N, LATENT) float tensor."""
    z = torch.load(path, map_location="cpu")
    if not torch.is_tensor(z):
        raise TypeError(f"{path} should contain a tensor, got {type(z)}")
    if z.ndim != 2:
        raise ValueError(f"{path}: expected shape (N, LATENT), got {tuple(z.shape)}")
    if not torch.isfinite(z).all():
        raise ValueError(f"{path} contains NaN or inf: check the autoencoder training")
    z = z.float()
    print(f"Loaded {path}: {z.shape[0]} frames, latent size {z.shape[1]}")
    return z


# ------------------------------------------------------------------ Model
class LatentLSTM(nn.Module):
    def __init__(self, latent, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, latent)  # predicts the next latent vector

    def forward(self, z_seq):  # (B, T, latent)
        h, _ = self.lstm(z_seq)
        return self.out(h[:, -1])


def make_windows(z, T=10):
    """z: (N, latent) -> inputs (N-T, T, latent), targets (N-T, latent)."""
    assert len(z) > T, f"need more than T={T} frames, got {len(z)}"
    X = torch.stack([z[i:i + T] for i in range(len(z) - T)])
    y = z[T:]
    return X, y


# ------------------------------------------------------------------ Training
def train_lstm(z, T=10, epochs=300, hidden=64, lr=1e-3, save_path="lstm.pt"):
    """Trains the LSTM to predict the next latent vector from the T previous ones.
    Latents are normalized (mean/std per dimension); the stats are returned and saved
    so predictions can be converted back to the original latent scale."""
    mu, sd = z.mean(0), z.std(0) + 1e-6
    X, y = make_windows((z - mu) / sd, T)
    X, y = X.to(DEVICE), y.to(DEVICE)

    model = LatentLSTM(z.shape[1], hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Training LSTM on {DEVICE}: {len(X)} windows of {T} frames, latent size {z.shape[1]}, {epochs} epochs")
    for ep in range(1, epochs + 1):
        loss = F.mse_loss(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == 1:
            print(f"LSTM epoch {ep:3d}/{epochs} | loss {loss.item():.5f}")

    stats = (mu, sd)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "T": T, "hidden": hidden}, save_path)
    print(f"Saved {save_path}")
    return model, stats


@torch.no_grad()
def predict_next(model, stats, z_window):
    """z_window: (T, latent) in the ORIGINAL latent scale -> predicted next latent (latent,), same scale.
    The result can be given to the autoencoder's decoder to see the predicted image."""
    mu, sd = stats
    model.eval()
    x = ((z_window - mu) / sd).unsqueeze(0).to(DEVICE)
    return (model(x).cpu()[0] * sd) + mu


if __name__ == "__main__":
    z = load_latents("latents.pt")
    model, stats = train_lstm(z, T=10)