"""
Dataset Health Check — run this BEFORE training.
Tells you exactly what (if anything) needs fixing.
"""
import numpy as np

SAVE_DIR = "processed_76_channels"

print("=" * 55)
print("DATASET HEALTH CHECK")
print("=" * 55)

# ── Load all splits ───────────────────────────────────────
train_X = np.load(f"{SAVE_DIR}/train_X.npy")
train_y = np.load(f"{SAVE_DIR}/train_y.npy")
val_X   = np.load(f"{SAVE_DIR}/val_X.npy")
val_y   = np.load(f"{SAVE_DIR}/val_y.npy")
test_X  = np.load(f"{SAVE_DIR}/test_X.npy")
test_y  = np.load(f"{SAVE_DIR}/test_y.npy")

# ── 1. Shapes ─────────────────────────────────────────────
print("\n[1] SHAPES")
print(f"  train_X : {train_X.shape}   (expected: (T, 106))")
print(f"  val_X   : {val_X.shape}")
print(f"  test_X  : {test_X.shape}")
num_features = train_X.shape[-1]
print(f"  → num_features to use in model: {num_features}")

# ── 2. Anomaly contamination ──────────────────────────────
print("\n[2] ANOMALY RATES")
train_rate = train_y.mean() * 100
val_rate   = val_y.mean()   * 100
test_rate  = test_y.mean()  * 100
print(f"  Train : {train_rate:.2f}%  {'⚠ HIGH — model sees too many anomalies' if train_rate > 5 else '✓ OK'}")
print(f"  Val   : {val_rate:.2f}%")
print(f"  Test  : {test_rate:.2f}%  (anomalies are only evaluated here)")

# ── 3. NaN / Inf check ────────────────────────────────────
print("\n[3] NaN / INF CHECK")
for name, arr in [("train_X", train_X), ("val_X", val_X), ("test_X", test_X)]:
    nans = np.isnan(arr).sum()
    infs = np.isinf(arr).sum()
    status = "✓ clean" if nans == 0 and infs == 0 else f"⚠ NaN={nans}  Inf={infs}"
    print(f"  {name}: {status}")

# ── 4. Value range (after z-score should be ~[-5, 5]) ─────
print("\n[4] VALUE RANGE (z-scored data should sit within ±5)")
for name, arr in [("train_X", train_X), ("val_X", val_X)]:
    abs_max = np.abs(arr).max()
    p99     = np.percentile(np.abs(arr), 99)
    status  = "✓ OK" if abs_max < 20 else "⚠ OUTLIERS PRESENT — will cause loss spikes"
    print(f"  {name}:  abs_max={abs_max:.2f}   99th-pct={p99:.2f}   {status}")

# ── 5. Per-channel outlier scan ───────────────────────────
print("\n[5] CHANNELS WITH EXTREME VALUES (abs_max > 20 after normalisation)")
bad_channels = []
for c in range(train_X.shape[1]):
    col_max = np.abs(train_X[:, c]).max()
    if col_max > 20:
        bad_channels.append((c, col_max))

if bad_channels:
    print(f"  ⚠  {len(bad_channels)} bad channel(s):")
    for c, v in sorted(bad_channels, key=lambda x: -x[1])[:10]:
        print(f"     channel index {c:3d}  →  abs_max = {v:.1f}")
    print("  → These need clipping in the pipeline (see fix below).")
else:
    print("  ✓ All channels within safe range.")

# ── 6. Constant / zero-variance channels ─────────────────
print("\n[6] CONSTANT CHANNELS (zero variance = useless features)")
const_channels = [c for c in range(train_X.shape[1])
                  if train_X[:, c].std() < 1e-6]
if const_channels:
    print(f"  ⚠  {len(const_channels)} constant channel(s): {const_channels[:10]}")
else:
    print("  ✓ No constant channels.")

# ── SUMMARY ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY — what you need to do:")
needs_fix = False

if train_rate > 5:
    print("  ⚠  High train anomaly rate → filter anomalous rows before training")
    needs_fix = True
if bad_channels:
    print("  ⚠  Outlier channels → add clipping to pipeline (no full rerun needed)")
    needs_fix = True
if const_channels:
    print("  ⚠  Constant channels → drop them (no full rerun needed)")
    needs_fix = True
if not needs_fix:
    print("  ✓ Dataset looks healthy. No reprocessing needed.")
    print(f"  ✓ Set num_features = {num_features} in train_transformer.py")
print("=" * 55)
