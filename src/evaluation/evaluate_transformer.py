"""
evaluate_transformer.py
=======================
Full evaluation pipeline for the Anomaly Transformer on ESA-ADB telemetry.
Produces all metrics, graphs, and tables from the base paper (ICLR 2022),
adapted to your dataset.

Outputs (saved to ./eval_outputs/):
  1.  metrics_table.csv          — P, R, F1 at best threshold
  2.  roc_curve.png              — ROC + AUC  (paper Fig 3)
  3.  anomaly_score_plot.png     — raw channel + anomaly score overlay (paper Fig 9)
  4.  association_discrepancy.png— AssDis for normal vs anomalous windows (paper Fig 5)
  5.  prior_vs_series_heatmap.png— prior-assoc vs series-assoc heatmap (paper Fig 1)
  6.  sigma_distribution.png     — learned σ for normal vs anomaly (paper Fig 6)
  7.  training_curve.png         — train/val loss over epochs
  8.  threshold_sensitivity.png  — F1 vs threshold r (paper Appendix A)
  9.  confusion_matrix.png       — at best F1 threshold
  10. score_distribution.png     — histogram of anomaly scores
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import math

from anomaly_transformer import AnomalyTransformer

# ─────────────────────────────────────────────────────────────────
# CONFIG — adjust if needed
# ─────────────────────────────────────────────────────────────────
DATA_DIR      = "processed_76_channels"
MODEL_PATH    = "anomaly_transformer_best.pth"
OUT_DIR       = "eval_outputs"
SEQ_LEN       = 128
STRIDE        = 16          # non-overlapping would be SEQ_LEN, overlapping gives smoother scores
NUM_FEATURES  = 106
BATCH_SIZE    = 64
CLIP_VALUE    = 10.0
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Visualisation settings
CHANNEL_TO_PLOT = 0         # which telemetry channel to show in anomaly score plot
N_CHANNELS_HEATMAP = 5      # channels shown in heatmap

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────
class InferenceDataset(Dataset):
    def __init__(self, npy_path, seq_len=128, stride=16, clip_value=10.0):
        self.data      = np.clip(np.load(npy_path).astype(np.float32),
                                 -clip_value, clip_value)
        self.seq_len   = seq_len
        self.stride    = stride
        self.n_windows = (len(self.data) - self.seq_len) // self.stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        s = idx * self.stride
        return torch.tensor(self.data[s : s + self.seq_len])

    def get_window_label(self, labels_flat, idx):
        """Window is anomalous if ANY timestep in it is anomalous."""
        s = idx * self.stride
        return int(labels_flat[s : s + self.seq_len].max())


# ─────────────────────────────────────────────────────────────────
# 2. ASSOCIATION DISCREPANCY  (Paper Eq 3)
# ─────────────────────────────────────────────────────────────────
def compute_association_discrepancy(series_list, prior_list):
    """
    Returns point-wise AssDis averaged over layers and heads.
    Shape: (B, N)
    """
    total = None
    for series, prior in zip(series_list, prior_list):
        # Average over heads → (B, N, N)
        s = series.mean(dim=1) + 1e-8
        p = prior.mean(dim=1)  + 1e-8
        kl1 = (p * (torch.log(p) - torch.log(s))).sum(dim=-1)   # (B, N)
        kl2 = (s * (torch.log(s) - torch.log(p))).sum(dim=-1)   # (B, N)
        layer_disc = kl1 + kl2
        total = layer_disc if total is None else total + layer_disc
    return total / len(series_list)   # (B, N)


# ─────────────────────────────────────────────────────────────────
# 3. ANOMALY SCORE  (Paper Eq 6)
# ─────────────────────────────────────────────────────────────────
def compute_anomaly_score(reconstruction, batch_x, ass_dis):
    """
    AnomalyScore = Softmax(-AssDis) ⊙ ||x - x̂||²   (Paper Eq 6)
    Returns per-window scalar score (mean over time and features).
    Shape: (B,)
    """
    rec_err   = ((batch_x - reconstruction) ** 2).mean(dim=-1)   # (B, N)
    norm_disc = F.softmax(-ass_dis, dim=-1)                       # (B, N)
    score     = (norm_disc * rec_err).mean(dim=-1)                # (B,)
    return score


# ─────────────────────────────────────────────────────────────────
# 4. RUN INFERENCE ON TEST SET
# ─────────────────────────────────────────────────────────────────
def run_inference(model, data_dir, device):
    print("\n[INFERENCE] Running on test set...")
    test_X_path = f"{data_dir}/test_X.npy"
    test_y_path = f"{data_dir}/test_y.npy"

    dataset     = InferenceDataset(test_X_path, SEQ_LEN, STRIDE, CLIP_VALUE)
    loader      = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    labels_flat = np.load(test_y_path).astype(np.int8)

    all_scores      = []
    all_labels      = []
    all_ass_dis     = []     # store first 200 windows for visualisation
    all_rec_errors  = []
    stored_batch    = None   # save one batch for heatmap/sigma plots

    model.eval()
    with torch.no_grad():
        for i, batch_x in enumerate(tqdm(loader, desc="  Scoring")):
            batch_x = batch_x.to(device)
            recon, series_list, prior_list = model(batch_x)

            ass_dis = compute_association_discrepancy(series_list, prior_list)  # (B, N)
            scores  = compute_anomaly_score(recon, batch_x, ass_dis)            # (B,)

            all_scores.append(scores.cpu().numpy())
            all_ass_dis.append(ass_dis.cpu().numpy())
            all_rec_errors.append(
                ((batch_x - recon) ** 2).mean(dim=-1).cpu().numpy()
            )

            # Save first batch for visualisation
            if stored_batch is None:
                stored_batch = {
                    "x"          : batch_x[:8].cpu().numpy(),
                    "recon"      : recon[:8].cpu().numpy(),
                    "series_list": [s[:8].cpu().numpy() for s in series_list],
                    "prior_list" : [p[:8].cpu().numpy() for p in prior_list],
                    "ass_dis"    : ass_dis[:8].cpu().numpy(),
                    "sigma"      : None,   # filled below
                }
                # Extract σ from first layer's attention for visualisation
                # We re-run one forward pass through just the first block to grab sigma
                x_emb = model.embedding(batch_x[:8])
                _, _, _ = model.blocks[0](x_emb)
                sigma_raw = model.blocks[0].attention.sigma_proj(x_emb)
                sigma_raw = torch.clamp(sigma_raw, 0.5, 10.0)  # (B, N, h)
                stored_batch["sigma"] = sigma_raw.cpu().numpy()

    all_scores     = np.concatenate(all_scores)
    all_ass_dis    = np.concatenate(all_ass_dis)
    all_rec_errors = np.concatenate(all_rec_errors)

    # Build window-level labels
    window_labels = np.array([
        dataset.get_window_label(labels_flat, i)
        for i in range(len(dataset))
    ])
    # Trim to match (drop_last=False but scores may be slightly longer due to padding)
    min_len       = min(len(all_scores), len(window_labels))
    all_scores    = all_scores[:min_len]
    window_labels = window_labels[:min_len]
    all_ass_dis   = all_ass_dis[:min_len]

    print(f"  Test windows   : {len(all_scores):,}")
    print(f"  Anomaly windows: {window_labels.sum():,} ({window_labels.mean()*100:.1f}%)")

    return all_scores, window_labels, all_ass_dis, all_rec_errors[:min_len], stored_batch


# ─────────────────────────────────────────────────────────────────
# 5. FIND BEST THRESHOLD  (paper: r-proportion method)
# ─────────────────────────────────────────────────────────────────
def find_best_threshold(scores, labels):
    """Sweep r (proportion flagged as anomaly) and return best F1 threshold."""
    best_f1, best_thresh, best_p, best_r = 0, 0, 0, 0
    results = []
    for r in np.linspace(0.001, 0.30, 200):
        thresh  = np.percentile(scores, (1 - r) * 100)
        preds   = (scores >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        p = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1  = f1_score(labels, preds, zero_division=0)
        results.append((r, thresh, p, rec, f1))
        if f1 > best_f1:
            best_f1, best_thresh, best_p, best_r = f1, thresh, p, rec
    return best_thresh, best_p, best_r, best_f1, results


# ─────────────────────────────────────────────────────────────────
# 6. PLOT: ROC CURVE  (paper Fig 3)
# ─────────────────────────────────────────────────────────────────
def plot_roc(scores, labels, out_dir):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E63946", lw=2,
            label=f"Anomaly Transformer (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — ESA Telemetry", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"{out_dir}/roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}  (AUC={roc_auc:.4f})")
    return roc_auc


# ─────────────────────────────────────────────────────────────────
# 7. PLOT: ANOMALY SCORE OVERLAY  (paper Fig 9)
# ─────────────────────────────────────────────────────────────────
def plot_anomaly_score_overlay(scores, labels, data_dir, out_dir,
                               max_timesteps=5000):
    test_X = np.clip(np.load(f"{data_dir}/test_X.npy").astype(np.float32),
                     -CLIP_VALUE, CLIP_VALUE)
    test_y = np.load(f"{data_dir}/test_y.npy").astype(np.int8)

    # Expand window scores back to timestep resolution
    T          = len(test_X)
    score_ts   = np.zeros(T)
    count_ts   = np.zeros(T)
    for i, sc in enumerate(scores):
        s = i * STRIDE
        e = min(s + SEQ_LEN, T)
        score_ts[s:e] += sc
        count_ts[s:e] += 1
    count_ts[count_ts == 0] = 1
    score_ts /= count_ts

    # Trim to max_timesteps for readability
    t  = np.arange(max_timesteps)
    ch = test_X[:max_timesteps, CHANNEL_TO_PLOT]
    sc = score_ts[:max_timesteps]
    lb = test_y[:max_timesteps]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Anomaly Transformer — ESA Telemetry Detection",
                 fontsize=13, fontweight="bold")

    # Row 1: raw channel
    axes[0].plot(t, ch, lw=0.8, color="#457B9D")
    axes[0].set_ylabel(f"Channel {CHANNEL_TO_PLOT}\n(normalised)", fontsize=10)
    axes[0].set_title("Input Time Series", fontsize=10)
    _shade_anomalies(axes[0], lb, t)

    # Row 2: reconstruction-only criterion
    axes[1].plot(t, sc, lw=0.8, color="#E63946")
    axes[1].set_ylabel("Anomaly Score", fontsize=10)
    axes[1].set_title("Association-based Criterion (Paper Eq 6)", fontsize=10)
    _shade_anomalies(axes[1], lb, t)

    # Row 3: ground truth labels
    axes[2].fill_between(t, lb, alpha=0.7, color="#E63946", step="mid")
    axes[2].set_ylabel("True Label", fontsize=10)
    axes[2].set_xlabel("Timestep", fontsize=10)
    axes[2].set_title("Ground Truth Anomalies", fontsize=10)
    axes[2].set_ylim(-0.05, 1.2)

    fig.tight_layout()
    path = f"{out_dir}/anomaly_score_plot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


def _shade_anomalies(ax, labels, t):
    in_anom = False
    start   = 0
    for i, lb in enumerate(labels):
        if lb == 1 and not in_anom:
            start   = t[i]
            in_anom = True
        elif lb == 0 and in_anom:
            ax.axvspan(start, t[i], alpha=0.25, color="#E63946")
            in_anom = False
    if in_anom:
        ax.axvspan(start, t[-1], alpha=0.25, color="#E63946")


# ─────────────────────────────────────────────────────────────────
# 8. PLOT: ASSOCIATION DISCREPANCY  (paper Fig 5)
# ─────────────────────────────────────────────────────────────────
def plot_association_discrepancy(all_ass_dis, window_labels, out_dir,
                                 n_examples=6):
    normal_idx = np.where(window_labels == 0)[0][:n_examples]
    anomal_idx = np.where(window_labels == 1)[0][:n_examples]

    fig, axes = plt.subplots(2, n_examples, figsize=(14, 5))
    fig.suptitle("Association Discrepancy: Normal vs Anomalous Windows\n"
                 "(lower = more anomalous, per paper observation)",
                 fontsize=11, fontweight="bold")

    for col, idx in enumerate(normal_idx):
        axes[0, col].plot(all_ass_dis[idx], lw=1.2, color="#457B9D")
        axes[0, col].set_title(f"Normal #{col+1}", fontsize=8)
        axes[0, col].set_ylim(bottom=0)
        if col == 0:
            axes[0, col].set_ylabel("AssDis", fontsize=9)

    for col, idx in enumerate(anomal_idx):
        axes[1, col].plot(all_ass_dis[idx], lw=1.2, color="#E63946")
        axes[1, col].set_title(f"Anomaly #{col+1}", fontsize=8)
        axes[1, col].set_ylim(bottom=0)
        if col == 0:
            axes[1, col].set_ylabel("AssDis", fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel("Time step", fontsize=7)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    path = f"{out_dir}/association_discrepancy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────────────────────────
# 9. PLOT: PRIOR vs SERIES ATTENTION HEATMAP  (paper Fig 1 style)
# ─────────────────────────────────────────────────────────────────
def plot_attention_heatmap(stored_batch, out_dir):
    prior  = stored_batch["prior_list"][0][0, 0]   # first layer, first head, first sample
    series = stored_batch["series_list"][0][0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Attention Maps — Layer 1, Head 1\n"
                 "Prior-Association (Gaussian) vs Series-Association (Learned)",
                 fontsize=11, fontweight="bold")

    cmap = LinearSegmentedColormap.from_list("blue_red", ["#FFFFFF", "#E63946"])

    im0 = axes[0].imshow(prior,  aspect="auto", cmap=cmap, vmin=0)
    axes[0].set_title("Prior-Association P\n(Gaussian kernel — adjacent bias)",
                       fontsize=9)
    axes[0].set_xlabel("Key timestep"); axes[0].set_ylabel("Query timestep")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(series, aspect="auto", cmap=cmap, vmin=0)
    axes[1].set_title("Series-Association S\n(Learned from raw series)",
                       fontsize=9)
    axes[1].set_xlabel("Key timestep"); axes[1].set_ylabel("Query timestep")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    path = f"{out_dir}/prior_vs_series_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────────────────────────
# 10. PLOT: σ DISTRIBUTION  (paper Fig 6)
# ─────────────────────────────────────────────────────────────────
def plot_sigma_distribution(stored_batch, window_labels, all_scores, out_dir):
    """
    Compare the mean learned σ for high-score (anomalous) vs low-score (normal) windows.
    Paper Fig 6: anomalies have smaller σ (more adjacent-concentrating prior).
    """
    sigma = stored_batch["sigma"]    # (8, N, h)
    mean_sigma_per_window = sigma.mean(axis=(1, 2))   # (8,)

    # Use the stored 8 windows — get their scores
    first8_scores = all_scores[:8]
    first8_labels = window_labels[:8]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Learned Scale Parameter σ (Prior-Association)\n"
                 "Smaller σ = more adjacent-concentrating = more anomalous",
                 fontsize=11, fontweight="bold")

    # Left: σ profile per window coloured by anomaly score
    norm = plt.Normalize(first8_scores.min(), first8_scores.max())
    cmap = plt.cm.RdYlGn_r
    for i in range(min(8, len(sigma))):
        color = cmap(norm(first8_scores[i]))
        axes[0].plot(sigma[i, :, 0], lw=1.0, color=color,
                     label=f"w{i} ({'A' if first8_labels[i] else 'N'})")
    axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("σ value")
    axes[0].set_title("σ profile (coloured by anomaly score)", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=axes[0], label="Anomaly Score")

    # Right: box plot — normal vs anomaly
    normal_sigma = sigma[first8_labels == 0].mean(axis=(1, 2)) if (first8_labels == 0).any() else np.array([0])
    anomal_sigma = sigma[first8_labels == 1].mean(axis=(1, 2)) if (first8_labels == 1).any() else np.array([0])
    axes[1].boxplot([normal_sigma, anomal_sigma],
                    labels=["Normal", "Anomaly"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#AED9E0"),
                    medianprops=dict(color="#E63946", lw=2))
    axes[1].set_ylabel("Mean σ"); axes[1].set_title("σ: Normal vs Anomaly", fontsize=9)

    fig.tight_layout()
    path = f"{out_dir}/sigma_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────────────────────────
# 11. PLOT: CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(scores, labels, threshold, out_dir):
    preds = (scores >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ["Normal", "Anomaly"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")

    thresh_cm = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh_cm else "black")
    fig.tight_layout()
    path = f"{out_dir}/confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────────────────────────
# 12. PLOT: ANOMALY SCORE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────
def plot_score_distribution(scores, labels, threshold, out_dir):
    normal_scores = scores[labels == 0]
    anomal_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_scores, bins=100, alpha=0.6, color="#457B9D",
            label="Normal", density=True)
    ax.hist(anomal_scores, bins=100, alpha=0.6, color="#E63946",
            label="Anomaly", density=True)
    ax.axvline(threshold, color="black", lw=1.5, linestyle="--",
               label=f"Best threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Anomaly Score Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"{out_dir}/score_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────────────────────────
# 13. PLOT: THRESHOLD SENSITIVITY  (paper Appendix A style)
# ─────────────────────────────────────────────────────────────────
def plot_threshold_sensitivity(threshold_results, out_dir):
    rs     = [r[0] * 100 for r in threshold_results]
    precs  = [r[2] * 100 for r in threshold_results]
    recs   = [r[3] * 100 for r in threshold_results]
    f1s    = [r[4] * 100 for r in threshold_results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rs, precs, lw=1.5, color="#457B9D",  label="Precision")
    ax.plot(rs, recs,  lw=1.5, color="#2A9D8F",  label="Recall")
    ax.plot(rs, f1s,   lw=2.0, color="#E63946",  label="F1-Score")
    best_r = threshold_results[np.argmax([r[4] for r in threshold_results])][0] * 100
    ax.axvline(best_r, color="black", lw=1.2, linestyle="--",
               label=f"Best r = {best_r:.1f}%")
    ax.set_xlabel("Anomaly proportion r (%)", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Threshold Sensitivity (P, R, F1 vs r)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"{out_dir}/threshold_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────────────────────────
# 14. SAVE METRICS TABLE  (paper Table 1 format)
# ─────────────────────────────────────────────────────────────────
def save_metrics_table(precision, recall, f1, auc_score, out_dir):
    rows = [
        ["Dataset",    "Model",               "P (%)",              "R (%)",           "F1 (%)",         "AUC"],
        ["ESA-ADB",    "Anomaly Transformer",
         f"{precision*100:.2f}",
         f"{recall*100:.2f}",
         f"{f1*100:.2f}",
         f"{auc_score:.4f}"],
    ]
    path = f"{out_dir}/metrics_table.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # Pretty print
    print("\n" + "="*55)
    print("  RESULTS TABLE  (Paper Table 1 format)")
    print("="*55)
    print(f"  {'Metric':<12} {'Value':>10}")
    print(f"  {'-'*24}")
    print(f"  {'Precision':<12} {precision*100:>9.2f}%")
    print(f"  {'Recall':<12} {recall*100:>9.2f}%")
    print(f"  {'F1-Score':<12} {f1*100:>9.2f}%")
    print(f"  {'AUC':<12} {auc_score:>10.4f}")
    print("="*55)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────────────────────────
# 15. TRAINING CURVE  (plot from log if available, else skip)
# ─────────────────────────────────────────────────────────────────
def plot_training_curve(out_dir):
    """
    Manually enter your epoch losses here from the terminal output.
    Matches paper Fig 10/11.
    """
    # ── Paste your epoch results here ──────────────────────────────
    train_losses = [0.025339, 0.019239, 0.018906, 0.018561, 0.018374]
    val_losses   = [0.091222, 0.095054, 0.096695, 0.097236, 0.096731]
    epochs       = list(range(1, len(train_losses) + 1))
    # ───────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, "o-", lw=2, color="#457B9D",
            label="Train Loss (SmoothL1)")
    ax.plot(epochs, val_losses,   "s--", lw=2, color="#E63946",
            label="Val Loss (SmoothL1)")
    ax.axvline(1, color="green", lw=1.2, linestyle=":",
               label="Best model saved (Epoch 1)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training & Validation Loss Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    fig.tight_layout()
    path = f"{out_dir}/training_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  ANOMALY TRANSFORMER — FULL EVALUATION")
    print("=" * 55)

    # Load model
    print(f"\n[MODEL] Loading from {MODEL_PATH} on {DEVICE.type.upper()}...")
    model = AnomalyTransformer(num_features=NUM_FEATURES, seq_len=SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Run inference
    scores, labels, ass_dis, rec_errors, stored_batch = run_inference(
        model, DATA_DIR, DEVICE
    )

    # Find best threshold
    print("\n[THRESHOLD] Sweeping r to find best F1...")
    best_thresh, best_p, best_r, best_f1, thresh_results = find_best_threshold(
        scores, labels
    )
    print(f"  Best threshold : {best_thresh:.6f}")
    print(f"  At r           : {(scores >= best_thresh).mean()*100:.2f}% flagged")

    # AUC
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score   = auc(fpr, tpr)

    # Generate all plots
    print("\n[PLOTS] Generating all figures...")
    plot_training_curve(OUT_DIR)
    plot_roc(scores, labels, OUT_DIR)
    plot_anomaly_score_overlay(scores, labels, DATA_DIR, OUT_DIR)
    plot_association_discrepancy(ass_dis, labels, OUT_DIR)
    plot_attention_heatmap(stored_batch, OUT_DIR)
    plot_sigma_distribution(stored_batch, labels[:8], scores[:8], OUT_DIR)
    plot_confusion_matrix(scores, labels, best_thresh, OUT_DIR)
    plot_score_distribution(scores, labels, best_thresh, OUT_DIR)
    plot_threshold_sensitivity(thresh_results, OUT_DIR)
    save_metrics_table(best_p, best_r, best_f1, auc_score, OUT_DIR)

    print(f"\n[DONE] All outputs saved to ./{OUT_DIR}/")
    print("       Use these 9 figures + metrics_table.csv directly in your PPT.")


if __name__ == "__main__":
    main()
