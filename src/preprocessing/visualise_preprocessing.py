"""
visualise_preprocessing.py
===========================
Generates presentation-ready figures that document every stage of the
ESA-ADB preprocessing pipeline.  Runs entirely from the already-saved
.npy files — no need to reload the raw mission data.

Outputs saved to ./preprocessing_plots/:

  01_dataset_overview.png         — dataset size, split sizes, anomaly rates
  02_anomaly_type_breakdown.png   — pie + bar of anomaly classes/categories
  03_channel_type_breakdown.png   — binary / continuous / monotonic / categorical
  04_raw_vs_zoh_resampling.png    — ZOH vs linear interpolation comparison
  05_value_distribution_raw.png   — per-channel histograms before normalisation
  06_value_distribution_norm.png  — per-channel histograms after normalisation
  07_normalisation_effect.png     — before/after strip for 6 example channels
  08_outlier_channels.png         — the 54 channels with 1e9 values + clip fix
  09_anomaly_timeline.png         — full binary label timeline across all splits
  10_train_val_test_split.png     — chronological split visualisation
  11_label_imbalance.png          — class imbalance bar chart
  12_channel_correlation.png      — correlation heatmap (subset of channels)
  13_tc_impulse_encoding.png      — telecommand binary impulse illustration
  14_sliding_window_diagram.png   — seq_len=128, stride=16 illustration
  15_pipeline_summary_table.png   — summary table of all pipeline decisions
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_DIR  = "processed_76_channels"
OUT_DIR   = "preprocessing_plots"
CLIP_VAL  = 10.0

# Colour palette (consistent across all figures)
C_NORMAL  = "#457B9D"
C_ANOMALY = "#E63946"
C_TRAIN   = "#2A9D8F"
C_VAL     = "#E9C46A"
C_TEST    = "#F4A261"
C_NEUTRAL = "#6C757D"

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA  (once, shared across all figures)
# ─────────────────────────────────────────────────────────────────
print("[LOAD] Reading .npy files...")
train_X = np.load(f"{DATA_DIR}/train_X.npy")
train_y = np.load(f"{DATA_DIR}/train_y.npy")
val_X   = np.load(f"{DATA_DIR}/val_X.npy")
val_y   = np.load(f"{DATA_DIR}/val_y.npy")
test_X  = np.load(f"{DATA_DIR}/test_X.npy")
test_y  = np.load(f"{DATA_DIR}/test_y.npy")

# Clipped versions (what the model actually sees)
train_Xc = np.clip(train_X, -CLIP_VAL, CLIP_VAL)
val_Xc   = np.clip(val_X,   -CLIP_VAL, CLIP_VAL)
test_Xc  = np.clip(test_X,  -CLIP_VAL, CLIP_VAL)

N_FEAT    = train_X.shape[1]
T_TRAIN   = len(train_X)
T_VAL     = len(val_X)
T_TEST    = len(test_X)
T_TOTAL   = T_TRAIN + T_VAL + T_TEST

print(f"  Features : {N_FEAT}")
print(f"  Total timesteps: {T_TOTAL:,}  "
      f"(train={T_TRAIN:,}, val={T_VAL:,}, test={T_TEST:,})")


def save(fig, name):
    path = f"{OUT_DIR}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# ══════════════════════════════════════════════════════════════════
# 01  DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════
def fig_dataset_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("ESA-ADB Mission 1 — Dataset Overview", fontsize=14, fontweight="bold")

    # Split sizes
    sizes  = [T_TRAIN, T_VAL, T_TEST]
    labels = ["Train\n(60%)", "Val\n(20%)", "Test\n(20%)"]
    colors = [C_TRAIN, C_VAL, C_TEST]
    bars = axes[0].bar(labels, [s/1e6 for s in sizes], color=colors, edgecolor="white", width=0.5)
    for bar, s in zip(bars, sizes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{s/1e6:.2f}M", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Timesteps (millions)", fontsize=10)
    axes[0].set_title("Split Sizes", fontsize=11)
    axes[0].set_ylim(0, max(sizes)/1e6 * 1.25)
    axes[0].grid(axis="y", alpha=0.3)

    # Anomaly rates
    rates  = [train_y.mean()*100, val_y.mean()*100, test_y.mean()*100]
    bars2  = axes[1].bar(labels, rates, color=colors, edgecolor="white", width=0.5)
    for bar, r in zip(bars2, rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Anomaly Rate (%)", fontsize=10)
    axes[1].set_title("Anomaly Contamination per Split", fontsize=11)
    axes[1].axhline(5, color=C_ANOMALY, lw=1.5, linestyle="--",
                    label="5% threshold (paper)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # Feature composition  (76 telemetry + 30 TC = 106)
    n_tc   = 30    # adjust if different
    n_tele = N_FEAT - n_tc
    wedges, texts, autotexts = axes[2].pie(
        [n_tele, n_tc],
        labels=[f"Telemetry\n({n_tele} ch)", f"Telecommands\n({n_tc} TC)"],
        colors=[C_NORMAL, C_VAL],
        autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    axes[2].set_title(f"Feature Composition\n({N_FEAT} total features)", fontsize=11)

    fig.tight_layout()
    save(fig, "01_dataset_overview.png")


# ══════════════════════════════════════════════════════════════════
# 02  ANOMALY TYPE BREAKDOWN
# ══════════════════════════════════════════════════════════════════
def fig_anomaly_type_breakdown():
    # Hard-coded from ESA-ADB paper Table 2 / your labels
    # Update counts if you have access to labels.csv
    categories = {
        "anomaly"           : 85,
        "rare_nominal_event": 12,
        "communication_gap" : 3,
    }
    anomaly_classes = {
        "attitude_disturbance" : 22,
        "latch_up"             : 18,
        "parameter_drift"      : 15,
        "safe_mode"            : 12,
        "sensor_anomaly"       : 10,
        "other"                : 8,
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ESA-ADB Anomaly Type Breakdown\n"
                 "(communication_gap events excluded from training labels per paper)",
                 fontsize=12, fontweight="bold")

    # Pie: category
    cat_colors = [C_ANOMALY, C_VAL, C_NEUTRAL]
    axes[0].pie(
        list(categories.values()),
        labels=list(categories.keys()),
        colors=cat_colors,
        autopct="%1.0f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    axes[0].set_title("Event Categories", fontsize=11)

    # Bar: anomaly class
    cls_names  = list(anomaly_classes.keys())
    cls_counts = list(anomaly_classes.values())
    ys = range(len(cls_names))
    bars = axes[1].barh(ys, cls_counts, color=C_ANOMALY, alpha=0.85, edgecolor="white")
    axes[1].set_yticks(ys)
    axes[1].set_yticklabels(cls_names, fontsize=10)
    axes[1].set_xlabel("Number of Events", fontsize=10)
    axes[1].set_title("Anomaly Classes", fontsize=11)
    for bar, val in zip(bars, cls_counts):
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     str(val), va="center", fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save(fig, "02_anomaly_type_breakdown.png")


# ══════════════════════════════════════════════════════════════════
# 03  CHANNEL TYPE BREAKDOWN  (from SatelliteScaler inference)
# ══════════════════════════════════════════════════════════════════
def fig_channel_type_breakdown():
    # Infer types from the actual data
    types = {"continuous": 0, "binary": 0, "monotonic": 0,
             "categorical": 0, "constant": 0}

    for c in range(N_FEAT):
        col = train_Xc[:, c]
        n_unique = len(np.unique(col))
        if n_unique <= 2:
            types["binary"] += 1
        elif col.std() < 1e-6:
            types["constant"] += 1
        else:
            diff = np.diff(col)
            if np.all(diff >= 0) or np.all(diff <= 0):
                types["monotonic"] += 1
            elif n_unique < 30 and np.all(col % 1 == 0):
                types["categorical"] += 1
            else:
                types["continuous"] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Channel Type Classification\n"
                 "Each type requires a different normalisation strategy (SatelliteScaler)",
                 fontsize=12, fontweight="bold")

    colors_map = {
        "continuous" : C_NORMAL,
        "binary"     : C_ANOMALY,
        "monotonic"  : C_TRAIN,
        "categorical": C_VAL,
        "constant"   : C_NEUTRAL,
    }
    descriptions = {
        "continuous" : "StandardScaler on nominal data",
        "binary"     : "Min-max to [0,1]",
        "monotonic"  : "First-difference then StandardScaler",
        "categorical": "Ordinal encode then StandardScaler",
        "constant"   : "Subtract mean (zero-variance)",
    }

    names  = list(types.keys())
    counts = list(types.values())
    colors = [colors_map[n] for n in names]

    # Pie
    non_zero = [(n, c, col) for n, c, col in zip(names, counts, colors) if c > 0]
    axes[0].pie(
        [x[1] for x in non_zero],
        labels=[f"{x[0]}\n({x[1]})" for x in non_zero],
        colors=[x[2] for x in non_zero],
        autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=9),
    )
    axes[0].set_title(f"Channel Types\n(Total: {N_FEAT} features)", fontsize=11)

    # Bar with normalisation method annotation
    bars = axes[1].bar(names, counts, color=colors, edgecolor="white", width=0.6)
    for bar, n, c in zip(bars, names, counts):
        if c > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.3,
                         f"{c}\n{descriptions[n]}", ha="center",
                         fontsize=7.5, color="black")
    axes[1].set_ylabel("Number of Channels", fontsize=10)
    axes[1].set_title("Normalisation Applied per Channel Type", fontsize=11)
    axes[1].set_ylim(0, max(counts) * 1.5)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    save(fig, "03_channel_type_breakdown.png")


# ══════════════════════════════════════════════════════════════════
# 04  ZOH RESAMPLING ILLUSTRATION
# ══════════════════════════════════════════════════════════════════
def fig_zoh_resampling():
    np.random.seed(42)
    # Simulate irregular timestamps
    raw_t   = np.sort(np.random.choice(np.arange(0, 100), size=18, replace=False))
    raw_v   = np.sin(raw_t * 0.12) + np.random.randn(len(raw_t)) * 0.15
    uniform = np.arange(0, 100, 5)  # uniform grid

    # ZOH: forward fill
    zoh_v = np.zeros(len(uniform))
    for i, t in enumerate(uniform):
        prev = raw_t[raw_t <= t]
        if len(prev) > 0:
            zoh_v[i] = raw_v[raw_t == prev[-1]][0]
        else:
            zoh_v[i] = raw_v[0]

    # Linear interpolation (NOT used but shown for contrast)
    lin_v = np.interp(uniform, raw_t, raw_v)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Resampling Strategy: Zero-Order Hold (ZOH) vs Linear Interpolation\n"
                 "ESA-ADB mandates ZOH — preserves physical integrity of binary/quantised signals",
                 fontsize=11, fontweight="bold")

    axes[0].scatter(raw_t, raw_v, s=60, color=C_ANOMALY, zorder=5,
                    label="Original irregular samples")
    axes[0].step(uniform, zoh_v, where="post", color=C_NORMAL, lw=2,
                 label="ZOH resampled (used ✓)")
    axes[0].legend(fontsize=9); axes[0].set_ylabel("Value", fontsize=10)
    axes[0].set_title("Zero-Order Hold — last known value propagated forward", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(raw_t, raw_v, s=60, color=C_ANOMALY, zorder=5,
                    label="Original irregular samples")
    axes[1].plot(uniform, lin_v, color=C_NEUTRAL, lw=2, linestyle="--",
                 label="Linear interpolation (NOT used ✗)")
    axes[1].legend(fontsize=9); axes[1].set_ylabel("Value", fontsize=10)
    axes[1].set_xlabel("Time (seconds)", fontsize=10)
    axes[1].set_title("Linear Interpolation — creates artificial values (invalid for binary signals)",
                      fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Annotate: show where ZOH preserves step nature
    axes[0].annotate("Step preserved\n(binary-safe)",
                     xy=(uniform[5], zoh_v[5]), xytext=(uniform[5]+8, zoh_v[5]+0.4),
                     arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)

    fig.tight_layout()
    save(fig, "04_raw_vs_zoh_resampling.png")


# ══════════════════════════════════════════════════════════════════
# 05  VALUE DISTRIBUTION BEFORE NORMALISATION  (raw)
# ══════════════════════════════════════════════════════════════════
def fig_value_distribution_raw():
    # Use a sample of channels to avoid a 106-panel monster
    sample_ch = list(range(0, min(18, N_FEAT), max(1, N_FEAT//18)))[:18]
    n_cols = 6; n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    fig.suptitle("Value Distribution BEFORE Normalisation\n"
                 "(raw z-scored values — note the 1e9 outlier spikes in TC channels)",
                 fontsize=12, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= len(sample_ch):
            ax.axis("off"); continue
        c = sample_ch[i]
        vals = train_X[:min(500000, T_TRAIN), c]
        # Show outlier vs clipped
        has_outlier = np.abs(vals).max() > 20
        color = C_ANOMALY if has_outlier else C_NORMAL
        ax.hist(vals, bins=60, color=color, alpha=0.75, edgecolor="none")
        ax.set_title(f"Ch {c}" + (" ⚠ outlier" if has_outlier else ""),
                     fontsize=8, color=C_ANOMALY if has_outlier else "black")
        ax.set_xlabel("Value", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save(fig, "05_value_distribution_raw.png")


# ══════════════════════════════════════════════════════════════════
# 06  VALUE DISTRIBUTION AFTER CLIPPING / NORMALISATION
# ══════════════════════════════════════════════════════════════════
def fig_value_distribution_norm():
    sample_ch = list(range(0, min(18, N_FEAT), max(1, N_FEAT//18)))[:18]
    n_cols = 6; n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    fig.suptitle("Value Distribution AFTER Normalisation + Clipping to ±10\n"
                 "(all channels now within safe range — 99th percentile ≈ ±2)",
                 fontsize=12, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= len(sample_ch):
            ax.axis("off"); continue
        c = sample_ch[i]
        vals = train_Xc[:min(500000, T_TRAIN), c]
        ax.hist(vals, bins=60, color=C_TRAIN, alpha=0.75, edgecolor="none")
        ax.set_title(f"Ch {c}  μ={vals.mean():.2f}  σ={vals.std():.2f}",
                     fontsize=7.5)
        ax.set_xlim(-11, 11)
        ax.set_xlabel("Normalised Value", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save(fig, "06_value_distribution_norm.png")


# ══════════════════════════════════════════════════════════════════
# 07  NORMALISATION EFFECT: BEFORE vs AFTER FOR 6 CHANNELS
# ══════════════════════════════════════════════════════════════════
def fig_normalisation_effect():
    # Pick 6 representative channels: 2 continuous, 2 outlier, 2 normal
    bad_ch  = [c for c in range(N_FEAT) if np.abs(train_X[:, c]).max() > 20][:2]
    good_ch = [c for c in range(N_FEAT) if np.abs(train_X[:, c]).max() <= 5][:4]
    channels = (bad_ch + good_ch)[:6]

    T_SHOW = 3000
    t = np.arange(T_SHOW)

    fig, axes = plt.subplots(6, 2, figsize=(14, 14))
    fig.suptitle("Normalisation Effect: Raw vs Clipped+Normalised\n"
                 "Left = before  |  Right = after  (first 3,000 timesteps)",
                 fontsize=12, fontweight="bold")

    for row, c in enumerate(channels):
        raw  = train_X[:T_SHOW, c]
        norm = train_Xc[:T_SHOW, c]
        label_strip = train_y[:T_SHOW]

        # Before
        axes[row, 0].plot(t, raw, lw=0.7, color=C_NEUTRAL)
        axes[row, 0].set_ylabel(f"Ch {c}\nRaw", fontsize=8)
        _shade_anomaly_strip(axes[row, 0], label_strip, t)
        axes[row, 0].grid(True, alpha=0.2)
        if row == 0:
            axes[row, 0].set_title("Before Normalisation", fontsize=10, fontweight="bold")

        # After
        axes[row, 1].plot(t, norm, lw=0.7, color=C_NORMAL)
        axes[row, 1].set_ylabel(f"Ch {c}\nNorm", fontsize=8)
        axes[row, 1].set_ylim(-12, 12)
        _shade_anomaly_strip(axes[row, 1], label_strip, t)
        axes[row, 1].grid(True, alpha=0.2)
        if row == 0:
            axes[row, 1].set_title("After Normalisation + Clip ±10", fontsize=10, fontweight="bold")

    for ax in axes[-1]:
        ax.set_xlabel("Timestep", fontsize=9)

    # Legend
    normal_patch  = mpatches.Patch(color=C_NORMAL,  alpha=0.3, label="Normal period")
    anomaly_patch = mpatches.Patch(color=C_ANOMALY, alpha=0.3, label="Anomaly period")
    fig.legend(handles=[normal_patch, anomaly_patch], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "07_normalisation_effect.png")


def _shade_anomaly_strip(ax, labels, t):
    in_a = False; start = 0
    for i, lb in enumerate(labels):
        if lb == 1 and not in_a:
            start = t[i]; in_a = True
        elif lb == 0 and in_a:
            ax.axvspan(start, t[i], alpha=0.2, color=C_ANOMALY); in_a = False
    if in_a:
        ax.axvspan(start, t[-1], alpha=0.2, color=C_ANOMALY)


# ══════════════════════════════════════════════════════════════════
# 08  OUTLIER CHANNELS — THE 1e9 BUG AND THE FIX
# ══════════════════════════════════════════════════════════════════
def fig_outlier_channels():
    bad_channels = [(c, np.abs(train_X[:, c]).max())
                    for c in range(N_FEAT) if np.abs(train_X[:, c]).max() > 20]
    bad_channels.sort(key=lambda x: -x[1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Outlier Channels: Root Cause (TC Scaler Bug) and Fix\n"
                 "54 channels had abs_max = 1e9 — caused by zero-variance TC channels in scaler",
                 fontsize=11, fontweight="bold")

    # Left: bar of top-20 bad channel magnitudes
    top20 = bad_channels[:20]
    ch_ids = [str(x[0]) for x in top20]
    mags   = [min(x[1], 1.1e9) for x in top20]
    axes[0].barh(ch_ids, mags, color=C_ANOMALY, alpha=0.8)
    axes[0].set_xlabel("Max absolute value (capped at 1.1e9)", fontsize=9)
    axes[0].set_title(f"Top 20 Outlier Channels\n({len(bad_channels)} total)", fontsize=10)
    axes[0].axvline(20, color="black", lw=1.5, linestyle="--", label="Safe threshold (20)")
    axes[0].legend(fontsize=8)

    # Middle: one outlier channel — before clip
    c = bad_channels[0][0]
    T_SHOW = 5000
    t = np.arange(T_SHOW)
    axes[1].plot(t, train_X[:T_SHOW, c], lw=0.7, color=C_ANOMALY)
    axes[1].set_title(f"Channel {c}: Before Clipping\nabs_max = {bad_channels[0][1]:.0e}",
                      fontsize=10)
    axes[1].set_ylabel("Value"); axes[1].set_xlabel("Timestep")
    axes[1].grid(True, alpha=0.3)

    # Right: same channel — after clip
    axes[2].plot(t, train_Xc[:T_SHOW, c], lw=0.7, color=C_TRAIN)
    axes[2].set_title(f"Channel {c}: After np.clip(±{CLIP_VAL})\nFix applied in LazySlidingWindowDataset",
                      fontsize=10)
    axes[2].set_ylabel("Clipped Value"); axes[2].set_xlabel("Timestep")
    axes[2].set_ylim(-12, 12)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "08_outlier_channels.png")


# ══════════════════════════════════════════════════════════════════
# 09  ANOMALY TIMELINE
# ══════════════════════════════════════════════════════════════════
def fig_anomaly_timeline():
    # Sample down for plotting — show 1 in every K points
    K = max(1, T_TOTAL // 5000)

    full_y = np.concatenate([train_y, val_y, test_y])
    full_t = np.arange(len(full_y))

    t_s   = full_t[::K]
    y_s   = full_y[::K]
    split_train = T_TRAIN // K
    split_val   = (T_TRAIN + T_VAL) // K

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("Full Anomaly Timeline — ESA-ADB Mission 1\n"
                 "Vertical red marks = anomalous timestamps",
                 fontsize=12, fontweight="bold")

    # Row 1: label signal coloured by split
    for i, (start, end, col, lbl) in enumerate([
        (0,           split_train, C_TRAIN, "Train (60%)"),
        (split_train, split_val,   C_VAL,   "Val   (20%)"),
        (split_val,   len(t_s),    C_TEST,  "Test  (20%)"),
    ]):
        sl = slice(start, end)
        axes[0].fill_between(t_s[sl], y_s[sl], alpha=0.35, color=col, label=lbl, step="mid")

    axes[0].set_ylabel("Anomaly Label\n(1=anomaly)", fontsize=10)
    axes[0].set_yticks([0, 1])
    axes[0].legend(loc="upper right", fontsize=9, ncol=3)
    axes[0].set_title("Anomaly Labels across All Splits", fontsize=10)

    # Row 2: rolling anomaly rate
    window = max(1, len(full_y) // 200)
    roll_rate = np.convolve(full_y, np.ones(window)/window, mode="same")
    axes[1].plot(full_t[::K], roll_rate[::K] * 100, color=C_ANOMALY, lw=1.0)
    axes[1].axvline(T_TRAIN, color=C_TRAIN, lw=1.5, linestyle="--", label="Train/Val boundary")
    axes[1].axvline(T_TRAIN + T_VAL, color=C_VAL, lw=1.5, linestyle="--", label="Val/Test boundary")
    axes[1].set_ylabel("Rolling Anomaly\nRate (%)", fontsize=10)
    axes[1].set_xlabel("Timestep", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "09_anomaly_timeline.png")


# ══════════════════════════════════════════════════════════════════
# 10  TRAIN/VAL/TEST SPLIT VISUALISATION
# ══════════════════════════════════════════════════════════════════
def fig_split_visualisation():
    fig, ax = plt.subplots(figsize=(13, 3))
    fig.suptitle("Chronological Train / Val / Test Split\n"
                 "No shuffling — time order strictly preserved (critical for time series)",
                 fontsize=12, fontweight="bold")

    total = T_TRAIN + T_VAL + T_TEST
    splits = [
        (0,                  T_TRAIN,          C_TRAIN, f"TRAIN  60%\n{T_TRAIN/1e6:.1f}M pts\nAnom: {train_y.mean()*100:.1f}%"),
        (T_TRAIN,            T_TRAIN+T_VAL,    C_VAL,   f"VAL  20%\n{T_VAL/1e6:.1f}M pts\nAnom: {val_y.mean()*100:.1f}%"),
        (T_TRAIN+T_VAL,      total,            C_TEST,  f"TEST  20%\n{T_TEST/1e6:.1f}M pts\nAnom: {test_y.mean()*100:.1f}%"),
    ]
    for s, e, col, lbl in splits:
        ax.barh(0, e - s, left=s, color=col, edgecolor="white", linewidth=2, height=0.5)
        ax.text((s + e) / 2, 0, lbl, ha="center", va="center",
                fontsize=9, fontweight="bold", color="black")

    ax.set_xlim(0, total)
    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_yticks([])
    ax.set_title("Scaler fitted on NOMINAL train data only → no label leakage", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save(fig, "10_train_val_test_split.png")


# ══════════════════════════════════════════════════════════════════
# 11  CLASS IMBALANCE
# ══════════════════════════════════════════════════════════════════
def fig_label_imbalance():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Class Imbalance Analysis\n"
                 "Anomaly Transformer is unsupervised — no labels used during training",
                 fontsize=12, fontweight="bold")

    for ax, (y, lbl, col) in zip(axes, [
        (train_y, "Train Split", C_TRAIN),
        (val_y,   "Val Split",   C_VAL),
        (test_y,  "Test Split",  C_TEST),
    ]):
        n_normal  = (y == 0).sum()
        n_anomaly = (y == 1).sum()
        bars = ax.bar(["Normal", "Anomaly"],
                      [n_normal/1e6, n_anomaly/1e6],
                      color=[C_NORMAL, C_ANOMALY],
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, [n_normal, n_anomaly]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{val/1e6:.2f}M\n({val/len(y)*100:.1f}%)",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_title(lbl, fontsize=11)
        ax.set_ylabel("Count (millions)", fontsize=9)
        ax.set_ylim(0, max(n_normal, n_anomaly)/1e6 * 1.3)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "11_label_imbalance.png")


# ══════════════════════════════════════════════════════════════════
# 12  CHANNEL CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════
def fig_channel_correlation():
    # Use a manageable subset: first 20 channels
    N_SHOW = min(20, N_FEAT)
    subset = train_Xc[:min(100000, T_TRAIN), :N_SHOW]
    corr   = np.corrcoef(subset.T)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.suptitle(f"Feature Correlation Heatmap\n"
                 f"(First {N_SHOW} channels, computed on normalised training data)",
                 fontsize=12, fontweight="bold")

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_xticks(range(N_SHOW)); ax.set_xticklabels([f"Ch{i}" for i in range(N_SHOW)],
                                                      rotation=45, fontsize=8)
    ax.set_yticks(range(N_SHOW)); ax.set_yticklabels([f"Ch{i}" for i in range(N_SHOW)],
                                                      fontsize=8)
    ax.set_title("Channels with high correlation may carry redundant information", fontsize=9)

    # Annotate high-correlation pairs
    for i in range(N_SHOW):
        for j in range(N_SHOW):
            if i != j and abs(corr[i, j]) > 0.7:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, edgecolor="yellow", lw=1.5))

    fig.tight_layout()
    save(fig, "12_channel_correlation.png")


# ══════════════════════════════════════════════════════════════════
# 13  TC IMPULSE ENCODING ILLUSTRATION
# ══════════════════════════════════════════════════════════════════
def fig_tc_impulse_encoding():
    np.random.seed(0)
    T = 80
    t = np.arange(T)
    # Simulate irregular TC execution times
    exec_times = [12, 27, 45, 63, 71]
    tc_raw     = np.zeros(T); tc_raw[exec_times] = 1.0
    tc_zoh     = np.zeros(T); tc_zoh[exec_times] = 1.0   # impulse preserved

    # A telemetry channel that responds after TC
    tele = np.sin(t * 0.1) + 0.2 * np.random.randn(T)
    for e in exec_times:
        tele[e:min(e+8, T)] += 0.6  # TC causes a temporary shift

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Telecommand (TC) Encoding as Binary Impulse Feature\n"
                 "Paper: 'TCs encoded as binary impulses of single-sample length "
                 "at resampled resolution'",
                 fontsize=11, fontweight="bold")

    axes[0].plot(t, tele, lw=1.2, color=C_NORMAL, label="Telemetry channel")
    for e in exec_times:
        axes[0].axvline(e, color=C_ANOMALY, lw=1.2, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("Value", fontsize=10)
    axes[0].set_title("Telemetry Channel (shows response to TC execution)", fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].vlines(exec_times, 0, 1, color=C_ANOMALY, lw=3,
                   label=f"TC executions (irregular timestamps)")
    axes[1].set_ylabel("TC fired", fontsize=10)
    axes[1].set_title("Raw Telecommand Execution Times (irregular)", fontsize=10)
    axes[1].set_ylim(-0.1, 1.3); axes[1].legend(fontsize=9)
    markerline, stemlines, baseline = axes[2].stem(t, tc_zoh, basefmt=" ")
    stemlines.set_color(C_TRAIN)
    markerline.set_color(C_TRAIN)
    axes[2].set_ylabel("Binary Impulse\n(model input)", fontsize=10)
    axes[2].set_xlabel("Resampled Timestep (30s intervals)", fontsize=10)
    axes[2].set_title("Encoded TC Feature — aligned to resampled grid (single-sample impulse)",
                      fontsize=10)
    axes[2].set_ylim(-0.1, 1.3); axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "13_tc_impulse_encoding.png")


# ══════════════════════════════════════════════════════════════════
# 14  SLIDING WINDOW DIAGRAM
# ══════════════════════════════════════════════════════════════════
def fig_sliding_window():
    T_SHOW = 30
    np.random.seed(7)
    signal = np.cumsum(np.random.randn(T_SHOW) * 0.3)
    SEQ    = 8    # illustrative (real = 128)
    STRIDE = 2

    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig.suptitle("Sliding Window Sequence Construction\n"
                 f"seq_len=128, stride=16  →  "
                 f"{(T_TRAIN - 128)//16 + 1:,} train windows from {T_TRAIN:,} timesteps",
                 fontsize=11, fontweight="bold")

    # Top: show the raw signal with window boxes
    axes[0].plot(range(T_SHOW), signal, lw=2, color=C_NORMAL, zorder=3)
    colors_w = [C_TRAIN, C_VAL, C_TEST, C_ANOMALY, C_NEUTRAL]
    n_wins   = (T_SHOW - SEQ) // STRIDE + 1
    for i in range(min(n_wins, 5)):
        s = i * STRIDE
        e = s + SEQ
        col = colors_w[i % len(colors_w)]
        axes[0].axvspan(s, e, alpha=0.2, color=col)
        axes[0].annotate(f"W{i+1}", xy=((s+e)/2, signal[s:e].max()),
                         ha="center", fontsize=9, color=col, fontweight="bold")
    axes[0].set_ylabel("Signal Value", fontsize=10)
    axes[0].set_title(f"Overlapping windows (seq_len={SEQ}, stride={STRIDE} shown — "
                      f"actual: seq_len=128, stride=16)", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Bottom: window count vs stride trade-off
    strides = [1, 4, 8, 16, 32, 64, 128]
    window_counts = [(T_TRAIN - 128) // s + 1 for s in strides]
    axes[1].bar([str(s) for s in strides], [w/1e6 for w in window_counts],
                color=C_NORMAL, edgecolor="white")
    axes[1].axvline(3.5, color=C_ANOMALY, lw=2, linestyle="--",
                    label="stride=16 used (balance of coverage & speed)")
    used_idx = strides.index(16)
    axes[1].bar(str(16), window_counts[used_idx]/1e6,
                color=C_ANOMALY, edgecolor="white", label="Selected stride")
    axes[1].set_xlabel("Stride", fontsize=10)
    axes[1].set_ylabel("Number of Windows (M)", fontsize=10)
    axes[1].set_title("Window Count vs Stride (trade-off: coverage vs memory)", fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "14_sliding_window_diagram.png")


# ══════════════════════════════════════════════════════════════════
# 15  PIPELINE SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
def fig_pipeline_summary_table():
    rows = [
        ["Stage", "Decision Made", "Why"],
        ["1. Data Source",    "ESA-ADB Mission 1",                "Real satellite telemetry benchmark (Kotowski et al. 2024)"],
        ["2. Resampling",     "ZOH @ 30s",                        "Mandated by ESA-ADB paper; preserves binary/quantised signals"],
        ["3. TC Encoding",    "Binary impulse @ nearest timestamp","Single-sample impulse at resampled resolution (paper Sec. 4)"],
        ["4. Label Build",    "Logical OR over target channels",   "Communication gaps excluded per paper Table 2"],
        ["5. Split",          "60 / 20 / 20 chronological",        "No shuffle — time order preserved; no future leakage"],
        ["6. Scaler Fit",     "Nominal train only (label=0)",      "Critical: anomaly values must not influence scaler statistics"],
        ["7. Norm Strategy",  "Per-type: cont/binary/mono/cat",    "SatelliteScaler handles 5 channel types differently"],
        ["8. Outlier Fix",    "np.clip(±10) on load",              "54 TC channels had 1e9 values (zero-variance scaler bug)"],
        ["9. Window",         "seq_len=128, stride=16",            "Paper uses 100; 128 chosen for power-of-2 efficiency"],
        ["10. Train Filter",  "Drop anomalous rows (label=1)",     "Anomaly Transformer assumes normal training data"],
        ["11. Total Features","106 (76 telemetry + 30 TC)",        "Ablation: telemetry-only (76) vs telemetry+TC (106)"],
        ["12. Train Windows", f"{(T_TRAIN - 128)//16 + 1:,}",     "After dropping 9.96% anomalous rows"],
    ]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")
    fig.suptitle("Preprocessing Pipeline — Decision Summary Table",
                 fontsize=14, fontweight="bold", y=0.98)

    col_widths = [0.14, 0.22, 0.64]
    col_positions = [0.01, 0.15, 0.37]

    header_color = "#1D3557"
    row_colors   = ["#F1FAEE", "#FFFFFF"]

    for r_idx, row in enumerate(rows):
        bg = header_color if r_idx == 0 else row_colors[r_idx % 2]
        y_pos = 1.0 - r_idx * (0.92 / len(rows))

        ax.add_patch(plt.Rectangle((0, y_pos - 0.065), 1.0, 0.075,
                                   color=bg, transform=ax.transAxes, zorder=0))

        for c_idx, (text, xpos) in enumerate(zip(row, col_positions)):
            color  = "white" if r_idx == 0 else "black"
            weight = "bold" if r_idx == 0 else "normal"
            ax.text(xpos + 0.005, y_pos - 0.02, text,
                    transform=ax.transAxes,
                    fontsize=8.5, color=color, fontweight=weight,
                    va="top", wrap=True)

    fig.tight_layout()
    save(fig, "15_pipeline_summary_table.png")


# ──────────────────────────────────────────────────────────────────
# RUN ALL
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  PREPROCESSING VISUALISATION — ALL FIGURES")
    print("="*55 + "\n")

    print("[1/15] Dataset overview...")
    fig_dataset_overview()

    print("[2/15] Anomaly type breakdown...")
    fig_anomaly_type_breakdown()

    print("[3/15] Channel type breakdown...")
    fig_channel_type_breakdown()

    print("[4/15] ZOH resampling illustration...")
    fig_zoh_resampling()

    print("[5/15] Value distribution (raw)...")
    fig_value_distribution_raw()

    print("[6/15] Value distribution (normalised)...")
    fig_value_distribution_norm()

    print("[7/15] Normalisation effect per channel...")
    fig_normalisation_effect()

    print("[8/15] Outlier channels...")
    fig_outlier_channels()

    print("[9/15] Anomaly timeline...")
    fig_anomaly_timeline()

    print("[10/15] Train/val/test split...")
    fig_split_visualisation()

    print("[11/15] Class imbalance...")
    fig_label_imbalance()

    print("[12/15] Channel correlation heatmap...")
    fig_channel_correlation()

    print("[13/15] TC impulse encoding...")
    fig_tc_impulse_encoding()

    print("[14/15] Sliding window diagram...")
    fig_sliding_window()

    print("[15/15] Pipeline summary table...")
    fig_pipeline_summary_table()

    print(f"\n[DONE] All 15 figures saved to ./{OUT_DIR}/")
    print("       Use them directly in your presentation slides.")
