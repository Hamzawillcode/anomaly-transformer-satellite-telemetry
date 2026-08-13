import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from anomaly_transformer import AnomalyTransformer


# ─────────────────────────────────────────────────────────────────
# 1. DATASET  (with inline data fixes — no reprocessing needed)
# ─────────────────────────────────────────────────────────────────
class LazySlidingWindowDataset(Dataset):
    def __init__(self, npy_path, label_path, seq_len=128, stride=16,
                 clip_value=10.0, drop_anomalies=False):
        """
        Parameters
        ----------
        npy_path       : path to flat (T, F) numpy array
        label_path     : path to flat (T,) label array  (0=normal, 1=anomaly)
        clip_value     : clips all values to [-clip_value, +clip_value]
                         fixes the 1e9 outlier channels from the scaler bug
        drop_anomalies : if True, removes rows where label=1 before windowing
                         set True for train, False for val (keep all for loss tracking)
        """
        raw   = np.load(npy_path).astype(np.float32)
        labels = np.load(label_path).astype(np.int8)

        # ── FIX A: clip the 1e9 TC-channel outliers ───────────────────────────
        # Root cause: TC channels with no impulses in nominal training window get
        # scaled by 1/1e-9 = 1e9 in SatelliteScaler. Clipping to ±10 (≈5σ for
        # z-scored data) is the correct fix without reprocessing.
        raw = np.clip(raw, -clip_value, clip_value)

        # ── FIX B: remove anomalous timesteps from training data ──────────────
        # Anomaly Transformer assumes training data is predominantly normal.
        # Your train set has 9.96% anomalies — at that rate the model learns to
        # reconstruct anomalies as normal patterns, destroying its discriminative
        # ability. We simply drop those rows before building windows.
        if drop_anomalies:
            normal_mask = labels == 0
            raw    = raw[normal_mask]
            labels = labels[normal_mask]
            pct_kept = normal_mask.mean() * 100
            print(f"  [Dataset] Kept {pct_kept:.1f}% normal rows from {npy_path}")

        self.data      = raw
        self.seq_len   = seq_len
        self.stride    = stride
        self.n_windows = (len(self.data) - self.seq_len) // self.stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        return torch.tensor(self.data[start : start + self.seq_len, :])


# ─────────────────────────────────────────────────────────────────
# 2. KL DIVERGENCE  (Paper Eq 3 — symmetric)
# ─────────────────────────────────────────────────────────────────
def kl_loss(p, q):
    """
    Symmetric KL divergence between two row-normalised attention maps.
    p, q : (B, heads, N, N)
    Returns a scalar.
    """
    p = p + 1e-8
    q = q + 1e-8
    p_mean = p.mean(dim=1)   # (B, N, N) — average over heads
    q_mean = q.mean(dim=1)
    kl1 = (p_mean * (torch.log(p_mean) - torch.log(q_mean))).sum(dim=-1)
    kl2 = (q_mean * (torch.log(q_mean) - torch.log(p_mean))).sum(dim=-1)
    return (kl1 + kl2).mean()


# ─────────────────────────────────────────────────────────────────
# 3. TRAINING & VALIDATION LOOP
# ─────────────────────────────────────────────────────────────────
def train_model():
    data_dir     = "processed_76_channels"
    seq_len      = 128
    batch_size   = 32
    num_features = 106   # confirmed by health check

    print("[DATA] Loading datasets with fixes applied...")

    # drop_anomalies=True for train — removes the 9.96% contamination
    train_dataset = LazySlidingWindowDataset(
        npy_path       = f"{data_dir}/train_X.npy",
        label_path     = f"{data_dir}/train_y.npy",
        seq_len        = seq_len,
        stride         = 16,
        clip_value     = 10.0,
        drop_anomalies = True,      # ← KEY: train on normal data only
    )

    # drop_anomalies=False for val — we want a realistic loss estimate
    val_dataset = LazySlidingWindowDataset(
        npy_path       = f"{data_dir}/val_X.npy",
        label_path     = f"{data_dir}/val_y.npy",
        seq_len        = seq_len,
        stride         = 16,
        clip_value     = 10.0,
        drop_anomalies = False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, drop_last=True)

    print(f"  Train windows : {len(train_dataset):,}")
    print(f"  Val windows   : {len(val_dataset):,}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[SYSTEM] Training on: {device.type.upper()}")

    model = AnomalyTransformer(num_features=num_features, seq_len=seq_len).to(device)

    # Paper hyperparameters (restored now that data + backward are fixed)
    epochs = 10
    k      = 3.0    # λ in paper Eq 5
    lr     = 1e-4   # paper: Adam with lr=1e-4

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float('inf')

    print("\n[TRAIN] Starting Minimax Adversarial Training...")
    print(       "        (Paper Eq 5: minimize + maximize phases in one backward pass)\n")

    for epoch in range(epochs):

        # ── TRAINING ──────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0

        train_pbar = tqdm(train_loader,
                          desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
        for batch_x in train_pbar:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            reconstruction, series_list, prior_list = model(batch_x)
            rec_loss = criterion(reconstruction, batch_x)

            # ── Minimax loss (Paper Eq 5) ──────────────────────────────────────
            # Minimize phase: prior chases series  → series.detach() stops series grad
            # Maximize phase: series flees prior   → prior.detach() stops prior grad
            # Combined in ONE backward — detach() is the stop-gradient mechanism
            kl_min = 0.0   # prior chases series
            kl_max = 0.0   # series flees prior

            for series, prior in zip(series_list, prior_list):
                kl_min += kl_loss(prior,          series.detach())
                kl_max += kl_loss(prior.detach(), series)

            kl_min /= len(series_list)
            kl_max /= len(series_list)

            loss_min   = rec_loss + k * kl_min   # minimize discrepancy
            loss_max   = rec_loss - k * kl_max   # maximize discrepancy
            total_loss = loss_min + loss_max      # single backward pass

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += rec_loss.item()
            train_pbar.set_postfix({'rec': f"{rec_loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # ── VALIDATION ────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader,
                            desc=f"Epoch [{epoch+1}/{epochs}] Val  ", leave=False)
            for batch_x in val_pbar:
                batch_x = batch_x.to(device)
                reconstruction, _, _ = model(batch_x)
                val_loss = criterion(reconstruction, batch_x)
                total_val_loss += val_loss.item()
                val_pbar.set_postfix({'rec': f"{val_loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  ↳ Val loss improved → saving best weights...")
            torch.save(model.state_dict(), "anomaly_transformer_best.pth")

    print("\n[FINISH] Training complete. Best weights: 'anomaly_transformer_best.pth'")


if __name__ == "__main__":
    train_model()
