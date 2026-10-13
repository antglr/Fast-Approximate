"""
Pipeline: image series (1216 x 1936) -> one latent vector per image (2D CNN)
          -> latent sequence (N, LATENT) -> LSTM.

Step 1: train the convolutional autoencoder (no labels required).
Step 2: extract one latent vector per image.
Step 3: feed the latent sequence to an LSTM.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from helpers import *

H, W = 1216, 1936  # height, width (array shape is (N, H, W) = (100, 1216, 1936))
LATENT = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ Data
class ImageSeries(Dataset):
    """Wraps a numpy array of shape (N, H, W) = (100, 1216, 1936).
    Axis 0 must be the temporal order. Intensities are scaled to [0, 1].
    labels: optional (N, 3) array of (x, y, size), normalized to [0, 1]."""

    def __init__(self, array, labels=None):
        arr = np.asarray(array)
        assert arr.ndim == 3 and arr.shape[1:] == (H, W), f"shape {arr.shape}, expected (N, {H}, {W})"
        arr = arr.astype(np.float32)
        mx = arr.max()
        if mx > 0:
            arr /= mx  # global scaling keeps relative brightness between frames
        self.imgs = torch.from_numpy(arr)  # (N, H, W) float32, ~940 MB for 100 images
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].unsqueeze(0)  # (1, H, W)
        lab = torch.zeros(3) if self.labels is None else self.labels[i]
        return img, lab


# ------------------------------------------------------------------ Model
class AddCoords(nn.Module):
    """CoordConv: appends 2 channels (x, y) so the CNN knows the absolute position."""

    def forward(self, t):
        B, _, h, w = t.shape
        yy = torch.linspace(-1, 1, h, device=t.device).view(1, 1, h, 1).expand(B, 1, h, w)
        xx = torch.linspace(-1, 1, w, device=t.device).view(1, 1, 1, w).expand(B, 1, h, w)
        return torch.cat([t, xx, yy], dim=1)


def down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))


def up(cin, cout):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
    )


class Encoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.coords = AddCoords()
        self.conv = nn.Sequential(
            down(3, 16),    # 608 x 968   (H x W)
            down(16, 32),   # 304 x 484
            down(32, 64),   # 152 x 242
            down(64, 64),   # 76 x 121
            down(64, 64),   # 38 x 61
        )
        # No global average pooling (it loses position). Grid (height, width) = (4, 6).
        self.pool = nn.AdaptiveAvgPool2d((4, 6))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 6, 128), nn.ReLU(True), nn.Linear(128, latent))

    def forward(self, x):
        return self.fc(self.pool(self.conv(self.coords(x))))


class Decoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(True), nn.Linear(128, 64 * 4 * 6))
        self.net = nn.Sequential(up(64, 64), up(64, 32), up(32, 16), up(16, 8), up(8, 4), nn.Conv2d(4, 1, 3, padding=1))

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 4, 6)
        x = F.interpolate(x, size=(38, 61), mode="bilinear", align_corners=False)
        x = self.net(x)  # 1216 x 1952
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class AutoEncoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.encoder = Encoder(latent)
        self.decoder = Decoder(latent)
        self.aux_head = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(True), nn.Linear(32, 3), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z, self.aux_head(z)


# ------------------------------------------------------------------ Training
def weighted_mse(pred, target, k=100.0):
    """The black background dominates the image: bright pixels are up-weighted
    so the model does not learn to reconstruct an all-black image."""
    w = 1.0 + k * target
    return (w * (pred - target) ** 2).mean()


def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def train(ds, epochs=50, batch=15, lr=5e-3, aux_weight=1.0):
    """Trains the autoencoder and prints one line per epoch.
    No translation/rotation augmentation: it would move the spot and corrupt the latent."""
    supervised = ds.labels is not None
    model = AutoEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    print(f"Training autoencoder on {DEVICE}: {len(ds)} images, batch {batch}, {epochs} epochs")

    t0 = time.time()
    best, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        t_ep = time.time()
        tot = 0.0
        for img, lab in dl:
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            with torch.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
                rec, z, aux = model(img)
                loss = weighted_mse(rec.float(), img)
                if supervised:  # optional supervision (x, y, size)
                    loss = loss + aux_weight * F.mse_loss(aux.float(), lab)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is {loss.item()} at epoch {ep}: lower lr or check the data (NaN/inf).")
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(img)

        loss_ep = tot / len(ds)
        improved = loss_ep < best
        if improved:
            best = loss_ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        eta = elapsed / ep * (epochs - ep)
        print(f"epoch {ep:3d}/{epochs} | loss {loss_ep:.6f}{' *' if improved else '  '} | "
              f"{time.time() - t_ep:.1f}s | ETA {fmt_time(eta)}", flush=True)

    model.load_state_dict(best_state)  # keep the best epoch
    torch.save(model.state_dict(), "autoencoder.pt")
    print(f"Done in {fmt_time(time.time() - t0)} | best loss {best:.6f} | weights saved to autoencoder.pt")
    return model


# ------------------------------------------------------------------ Latent space tests (decoding)
def spot_stats(img):
    """img: (H, W) tensor. Returns (cx, cy, sigma) of the spot in pixels.
    The image is thresholded at half its maximum so background noise does not bias the result."""
    img = (img - 0.5 * img.max()).clamp(min=0)
    tot = img.sum() + 1e-8
    ys = torch.arange(H, dtype=img.dtype, device=img.device).view(-1, 1)
    xs = torch.arange(W, dtype=img.dtype, device=img.device).view(1, -1)
    cx = (img * xs).sum() / tot
    cy = (img * ys).sum() / tot
    var = (img * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / tot / 2
    return cx.item(), cy.item(), var.sqrt().item()


@torch.no_grad()
def reconstruct(model, imgs):
    """imgs: (B, 1, H, W) -> (reconstruction, latent)."""
    model.eval()
    rec, z, _ = model(imgs.to(DEVICE))
    return rec.cpu(), z.cpu()


@torch.no_grad()
def evaluate_reconstruction(model, ds, batch=4):
    """Encode -> decode every image and compare the original and the reconstructed spot."""
    mses, pos_err, size_err = [], [], []
    for img, _ in DataLoader(ds, batch_size=batch, shuffle=False):
        rec, _ = reconstruct(model, img)
        for a, b in zip(img, rec):
            mses.append(F.mse_loss(b, a).item())
            ax, ay, asg = spot_stats(a[0])
            bx, by, bsg = spot_stats(b[0])
            pos_err.append(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
            size_err.append(abs(asg - bsg) / (asg + 1e-8))
    print(f"MSE            mean {np.mean(mses):.6f}")
    print(f"Position error mean {np.mean(pos_err):.2f} px | max {np.max(pos_err):.2f} px")
    print(f"Size error     mean {100 * np.mean(size_err):.1f} % | max {100 * np.max(size_err):.1f} %")
    return dict(mse=mses, pos_err=pos_err, size_err=size_err)


def plot_reconstructions(model, ds, indices=(0, 25, 50, 99), path="reconstructions.png"):
    """Saves a figure: original | reconstruction | absolute difference, for the chosen frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs = torch.stack([ds[i][0] for i in indices])
    rec, _ = reconstruct(model, imgs)
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 3.2 * len(indices)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(indices):
        a, b = imgs[r, 0].numpy(), rec[r, 0].numpy()
        for c, (im, title) in enumerate([(a, f"original #{i}"), (b, "reconstruction"), (np.abs(a - b), "|difference|")]):
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(title)
            axes[r, c].axis("off")
    fig.tight_layout()
    plt.show()
    #fig.savefig(path, dpi=80)
    #plt.close(fig)
    #print("saved", path)


@torch.no_grad()
def plot_interpolation(model, ds, i=0, j=50, steps=6, path="interpolation.png"):
    """Decodes points on the straight line between the latents of frames i and j.
    A well-structured latent space makes the spot move/grow smoothly instead of fading in and out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, z = reconstruct(model, torch.stack([ds[i][0], ds[j][0]]))
    fig, axes = plt.subplots(1, steps, figsize=(3.2 * steps, 2.6))
    for k, a in enumerate(np.linspace(0, 1, steps)):
        zi = ((1 - a) * z[0] + a * z[1]).unsqueeze(0).to(DEVICE)
        dec = model.decoder(zi).cpu()[0, 0]
        cx, cy, sg = spot_stats(dec)
        axes[k].imshow(dec.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[k].set_title(f"a={a:.1f}\nx={cx:.0f} y={cy:.0f} s={sg:.1f}", fontsize=8)
        axes[k].axis("off")
    fig.tight_layout()
    plt.show()
    #fig.savefig(path, dpi=80)
    #plt.close(fig)
    #print("saved", path)


# ------------------------------------------------------------------ Latent extraction
@torch.no_grad()
def extract_latents(model, ds, batch=4):
    model.eval()
    out = []
    for img, _ in DataLoader(ds, batch_size=batch, shuffle=False):  # temporal order preserved
        out.append(model.encoder(img.to(DEVICE)).cpu())
    z = torch.cat(out)  # (N, LATENT)
    torch.save(z, "latents.pt")
    return z


# ------------------------------------------------------------------ LSTM on the latent sequence
class LatentLSTM(nn.Module):
    def __init__(self, latent=LATENT, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, latent)  # predicts the next latent vector

    def forward(self, z_seq):  # (B, T, latent)
        h, _ = self.lstm(z_seq)
        return self.out(h[:, -1])


def make_windows(z, T=10):
    """z: (N, latent) -> inputs (N-T, T, latent), targets (N-T, latent)."""
    X = torch.stack([z[i:i + T] for i in range(len(z) - T)])
    y = z[T:]
    return X, y


def train_lstm(z, T=10, epochs=300):
    mu, sd = z.mean(0), z.std(0) + 1e-6  # normalize the latents
    zn = (z - mu) / sd
    X, y = make_windows(zn, T)
    model = LatentLSTM(z.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X, y = X.to(DEVICE), y.to(DEVICE)
    for ep in range(epochs):
        loss = F.mse_loss(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 50 == 0:
            print(f"LSTM epoch {ep}  loss {loss.item():.5f}")
    return model, (mu, sd)


if __name__ == "__main__":
    folder = "../test_data/" 
    data = combiner(folder)['BTV0125'][0]['image2D']
    test_data = combiner(folder)['BTV0125'][1]['image2D']
    ds = ImageSeries(data)
    model = train(ds)
    ds_test = ImageSeries(test_data)

    evaluate_reconstruction(model, ds_test)           # decoding metrics
    plot_reconstructions(model, ds_test)              # reconstructions.png
    plot_interpolation(model, ds_test, i=0, j=50)     # interpolation.png
    # z = extract_latents(model, ds)               # (100, 16)
    # lstm, stats = train_lstm(z)