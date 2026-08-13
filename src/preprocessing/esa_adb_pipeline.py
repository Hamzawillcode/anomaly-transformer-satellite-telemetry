"""
ESA-ADB Mission 1: Complete Preprocessing Pipeline
====================================================
Tailored to the ESA Anomaly Detection Benchmark (ESA-ADB) dataset as described in:
  "European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry"
  Kotowski et al. (2024)

Supports:
  1. Transformer-based anomaly detection (sliding window sequences)
  2. LLM-based explanation generation (per-anomaly metadata)
  3. Ablation experiments: Telemetry-only vs. Telemetry + Telecommands

Dataset schema (inferred from paper Sec. 4 & Supplementary Material 2.4):
  channels/       → one .zip-compressed Pickle per channel (pandas DataFrame,
                    index=datetime timestamps, single value column)
  telecommands/   → same format; binary impulse at execution timestamps
  labels.csv      → [ID, channel_name, start_time, end_time]
  anomaly_types.csv → [ID, class, subclass, category, type]
  channels.csv    → [channel_name, subsystem, physical_unit, group, is_target]
  telecommands.csv→ [tc_name, priority]

Key satellite-telemetry constraints honoured:
  • Zero-order hold (ZOH) resampling — NOT linear interpolation
  • Point-anomaly correction after resampling
  • Per-channel standardisation on nominal training samples only (no leakage)
  • Binary and monotonic channels handled separately
  • Telecommands encoded as binary impulse features
  • Multi-segment anomaly IDs collapsed to single logical event
"""

# ===========================================================================
# 0.  IMPORTS & CONFIGURATION
# ===========================================================================

import os
import pickle
import zipfile
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# Global configuration — edit these before running
# ---------------------------------------------------------------------------
CFG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "mission_dir"      : "./ESA-Mission1",      # root folder of the mission

    # ── Resampling ─────────────────────────────────────────────────────────
    # Mission1 dominant target-channel frequency = 0.033 Hz → 1 sample / 30 s
    # Paper Sec. 4, "Target sampling frequencies"
    "resample_freq"    : "30s",                 # pandas offset alias

    # ── Train / val / test split ───────────────────────────────────────────
    # Paper: first half = train+val, second half = test
    # Last 3 months of train = validation
    "val_months"       : 3,

    # ── Sequence windows ──────────────────────────────────────────────────
    "seq_len"          : 128,                   # look-back window (samples)
    "stride"           : 1,                     # stride between windows
    "horizon"          : 1,                     # steps ahead to label

    # ── Telecommand options ────────────────────────────────────────────────
    # priority >= this threshold are included when tc_mode="filtered"
    "tc_priority_threshold": 2,

    # ── Normalisation ─────────────────────────────────────────────────────
    "n_std_binary"     : 5,                     # binary-channel clip threshold

    # ── Misc ──────────────────────────────────────────────────────────────
    "random_seed"      : 42,
    "dtype"            : np.float32,
}


# ===========================================================================
# 1.  DATA LOADING
# ===========================================================================

def _load_pickle_zip(path: Path) -> pd.DataFrame:
    """
    Load a single .zip-compressed Pickle file as used in ESA-AD.
    The file contains a pandas DataFrame with a datetime index and one column.
    Protocol: Pickle v4, zip compression (paper Supplementary Material 2.4).
    """
    with zipfile.ZipFile(path, "r") as zf:
        inner_name = zf.namelist()[0]          # single file inside the archive
        with zf.open(inner_name) as f:
            df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)} from {path}")
    df.index = pd.to_datetime(df.index)        # ensure DatetimeIndex
    df.index.name = "timestamp"
    return df


def load_channels(
    channels_dir: Path,
    channels_meta: pd.DataFrame,
    target_only: bool = False,
) -> Dict[str, pd.Series]:
    """
    Load all channel pickle files. (Full Dataset Mode)
    """
    channel_data: Dict[str, pd.Series] = {}
    files = {p.stem: p for p in channels_dir.glob("*.zip")}

    for _, row in tqdm(channels_meta.iterrows(),
                       total=len(channels_meta), desc="Loading channels"):
        name = row["channel_name"]
        
        if name not in files:
            warnings.warn(f"Channel file not found: {name}")
            continue
            
        df = _load_pickle_zip(files[name])
        series = df.iloc[:, 0].rename(name)
        channel_data[name] = series

    print(f"✓  Loaded {len(channel_data)} channels.")
    return channel_data
def load_telecommands(
    tc_dir: Path,
    tc_meta: pd.DataFrame,
    priority_threshold: int = 2,
) -> Dict[str, pd.Series]:
    """
    Load telecommand pickle files.

    Each TC is a binary signal: value=1 at execution timestamps.
    Paper: encoded as binary impulse of a single-sample length.

    Parameters
    ----------
    tc_dir             : Path to telecommands/ subfolder.
    tc_meta            : DataFrame from telecommands.csv [tc_name, priority].
    priority_threshold : Only load TCs with priority >= threshold.
                         Paper uses priority 3 for full-set experiments.
    """
    tc_data: Dict[str, pd.Series] = {}
    if not tc_dir.exists():
        print("No telecommands directory found — skipping.")
        return tc_data

    selected = tc_meta[tc_meta["priority"] >= priority_threshold]["tc_name"]
    files = {p.stem: p for p in tc_dir.glob("*.zip")}

    for name in tqdm(selected, desc="Loading telecommands"):
        if name not in files:
            continue
        df = _load_pickle_zip(files[name])
        series = df.iloc[:, 0].fillna(0).astype(np.float32).rename(name)
        tc_data[name] = series

    print(f"✓  Loaded {len(tc_data)} telecommands (priority ≥ {priority_threshold}).")
    return tc_data


def load_metadata(mission_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load channels, telecommands, and anomaly_types with robust header handling.
    """
    # 1. Force headers for channels.csv
    channels_meta = pd.read_csv(mission_dir / "channels.csv", header=None)
    channels_meta.columns = ["channel_name", "subsystem", "physical_unit", "group", "is_target"]
    if channels_meta["channel_name"].iloc[0] == "channel_name":
        channels_meta = channels_meta.iloc[1:].reset_index(drop=True)
        
    # 2. Force headers for anomaly_types.csv
    anomaly_types = pd.read_csv(mission_dir / "anomaly_types.csv", header=None,usecols=[0,1,2,3,4])
    anomaly_types.columns = ["ID", "class", "subclass", "category", "type"]
    if str(anomaly_types["ID"].iloc[0]) == "ID":
        anomaly_types = anomaly_types.iloc[1:].reset_index(drop=True)

    # 3. Force headers for telecommands.csv
    tc_meta_path = mission_dir / "telecommands.csv"
    if tc_meta_path.exists():
        tc_meta = pd.read_csv(tc_meta_path, header=None)
        tc_meta.columns = ["tc_name", "priority"]
        if tc_meta["tc_name"].iloc[0] == "tc_name":
            tc_meta = tc_meta.iloc[1:].reset_index(drop=True)
        # Convert priority to numeric so filtering works
        tc_meta["priority"] = pd.to_numeric(tc_meta["priority"], errors='coerce').fillna(0)
    else:
        tc_meta = pd.DataFrame(columns=["tc_name", "priority"])

    print(f"✓  Metadata loaded — {len(channels_meta)} channels, "
          f"{len(tc_meta)} telecommands, {len(anomaly_types)} anomaly types.")
    return channels_meta, tc_meta, anomaly_types

def load_labels(mission_dir: Path) -> pd.DataFrame:
    """
    Load labels.csv and robustly handle missing or weird headers.
    """
    # Load the file without assuming any headers exist
    labels = pd.read_csv(mission_dir / "labels.csv", header=None)
    
    # Force the 4 columns to be named exactly what the rest of the script expects
    labels.columns = ["ID", "channel_name", "start_time", "end_time"]
    
    # If the researchers actually DID include a header row, this removes that text row
    if labels["start_time"].iloc[0] == "start_time" or labels["ID"].iloc[0] == "ID":
        labels = labels.iloc[1:].reset_index(drop=True)
        
    labels["start_time"] = pd.to_datetime(labels["start_time"],utc=True).dt.tz_localize(None)
    labels["end_time"]   = pd.to_datetime(labels["end_time"],utc=True).dt.tz_localize(None)
    
    print(f"✓  Labels loaded — {len(labels)} annotated segments, "
          f"{labels['ID'].nunique()} unique event IDs.")
    return labels

# ===========================================================================
# 2.  RESAMPLING  (Zero-Order Hold — mandated by the paper)
# ===========================================================================

def _zoh_resample_series(
    series: pd.Series,
    target_freq: str,
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
) -> pd.Series:
    """
    Resample a single channel using Zero-Order Hold (ZOH) interpolation.

    ZOH = propagate the last known value forward.
    This is required by the ESA-ADB paper (Methods, "Resampling") because:
      • It does not create artificial intermediate values (safe for binary/quantised signals).
      • It does not use future samples (required for real-time/online evaluation).

    Steps mirror Algorithm in paper Sec. 4 exactly:
      1. Build uniform timestamp grid.
      2. Propagate last known value (forward-fill).
      3. Back-propagate FIRST known value for any leading NaN.
      4. Point-anomaly correction: ensure no annotated point is dropped.
         (Point anomaly correction for labels is applied separately after merge.)
    """
    # Step 1: build uniform grid from global_start to global_end
    uniform_idx = pd.date_range(start=global_start, end=global_end,
                                freq=target_freq, name="timestamp")

    # Merge original series into uniform grid, then forward-fill (ZOH)
    resampled = (
        series
        .reindex(uniform_idx.union(series.index))   # combine both indices
        .sort_index()
        .ffill()                                     # ZOH: propagate last value
        .bfill()                                     # back-fill leading NaN (Step 3)
        .reindex(uniform_idx)                        # keep only uniform grid
    )
    resampled.name = series.name
    return resampled


def resample_all_channels(
    channel_data: Dict[str, pd.Series],
    target_freq: str,
) -> pd.DataFrame:
    """
    Resample all channels to a common uniform grid and merge into a
    single wide DataFrame.  Each column = one channel.

    Alignment strategy:
      - Global start = earliest timestamp across all channels.
      - Global end   = latest timestamp across all channels.
    This mirrors the paper's approach: "Set the first/last timestamp in the
    list to the value of the earliest/latest original timestamp across all
    channels rounded down/up to the target sampling resolution."
    """
    all_times = pd.concat([s for s in channel_data.values()]).index
    global_start = all_times.min().floor(target_freq)
    global_end   = all_times.max().ceil(target_freq)

    resampled_list = []
    for name, series in tqdm(channel_data.items(), desc="Resampling channels"):
        rs = _zoh_resample_series(series, target_freq, global_start, global_end)
        resampled_list.append(rs)

    merged = pd.concat(resampled_list, axis=1)
    print(f"✓  Resampled merged DataFrame: {merged.shape}  "
          f"({global_start} → {global_end}, freq={target_freq})")
    return merged


def encode_telecommands(
    tc_data: Dict[str, pd.Series],
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align telecommand impulses to the resampled timestamp grid.

    Paper: "TCs are encoded as binary impulses of a single sample length
    according to the target resampling resolution."

    For each TC, value=1 on the resampled timestamp nearest to each
    execution, 0 everywhere else.  This preserves the impulse even if
    the original TC timestamp falls between two resampled timestamps.
    """
    if not tc_data:
        return pd.DataFrame(index=target_index)

    tc_frame = pd.DataFrame(0.0, index=target_index,
                            columns=list(tc_data.keys()))

    for name, series in tc_data.items():
        # Find the closest resampled timestamp for each TC execution
        exec_times = series[series > 0].index
        for t in exec_times:
            # nearest resampled timestamp
            idx_pos = target_index.searchsorted(t, side="left")
            idx_pos = min(idx_pos, len(target_index) - 1)
            tc_frame.at[target_index[idx_pos], name] = 1.0

    print(f"✓  Telecommands encoded: {tc_frame.shape[1]} features.")
    return tc_frame


# ===========================================================================
# 3.  LABEL CONSTRUCTION
# ===========================================================================

def build_label_matrix(
    labels: pd.DataFrame,
    anomaly_types: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    channels_meta: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Convert anomaly intervals → per-timestamp labels.

    Returns
    -------
    binary_labels  : pd.Series, shape=(T,), dtype=int8
                     1 = anomaly present in ANY target channel at that timestamp.
    channel_labels : pd.DataFrame, shape=(T, n_target_channels)
                     Per-channel binary label matrix (for channel-aware metrics).

    Notes
    -----
    • Labels operate in TIME domain, not samples domain (paper Sec. 4, Metrics).
    • Multiple segments with the same ID = single logical event (paper Sec. 2).
    • Non-target channels are NOT labelled.
    • Rare nominal events + communication gaps ARE included by default
      (paper Table 2 uses all events except comm. gaps).
      Set exclude_categories below to change this.
    """
    # Map event ID → category (anomaly | rare_nominal_event | communication_gap)
    id_to_meta = anomaly_types.set_index("ID")

    target_channels = set(
        channels_meta[channels_meta["is_target"] == True]["channel_name"]
    )

    # Binary global label (logical OR across target channels)
    binary_arr = np.zeros(len(target_index), dtype=np.int8)

    # Per-channel label matrix
    target_ch_list = sorted(target_channels)
    ch_label_df = pd.DataFrame(
        0, index=target_index, columns=target_ch_list, dtype=np.int8
    )

    # Group by event ID to handle multi-segment events
    for event_id, group in labels.groupby("ID"):
        # Skip communication gaps (paper: excluded from default metrics)
        if event_id in id_to_meta.index:
            category = id_to_meta.loc[event_id, "category"]
            if category == "communication_gap":
                continue

        for _, row in group.iterrows():
            ch = row["channel_name"]
            start, end = row["start_time"], row["end_time"]

            # Mask: timestamps within [start, end]
            mask = (target_index >= start) & (target_index <= end)
            binary_arr[mask] = 1

            if ch in target_channels:
                ch_label_df.loc[mask, ch] = 1

    binary_labels = pd.Series(binary_arr, index=target_index, name="label")

    anomaly_rate = binary_labels.mean() * 100
    print(f"✓  Labels built — anomaly rate: {anomaly_rate:.2f}% of timestamps.")
    return binary_labels, ch_label_df


def build_multiclass_labels(
    labels: pd.DataFrame,
    anomaly_types: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Multi-class label: each timestamp gets the anomaly class ID (int).
    0 = nominal.  Useful for fine-grained evaluation and LLM prompts.
    """
    id_to_class = anomaly_types.set_index("ID")["class"].to_dict()
    # Build unique class → integer mapping
    classes = sorted(set(id_to_class.values()))
    class_to_int = {c: i + 1 for i, c in enumerate(classes)}  # 0 reserved for nominal

    mc_arr = np.zeros(len(target_index), dtype=np.int16)

    for event_id, group in labels.groupby("ID"):
        if event_id in id_to_class:
            cls_int = class_to_int.get(id_to_class[event_id], 0)
            for _, row in group.iterrows():
                mask = (target_index >= row["start_time"]) & \
                       (target_index <= row["end_time"])
                mc_arr[mask] = cls_int

    mc_series = pd.Series(mc_arr, index=target_index, name="multiclass_label")
    print(f"✓  Multi-class labels: {len(classes)} distinct anomaly classes.")
    return mc_series, class_to_int


# ===========================================================================
# 4.  TRAIN / VALIDATION / TEST SPLIT
# ===========================================================================



# ===========================================================================
# 5.  CHANNEL-AWARE NORMALISATION  (no data leakage)
# ===========================================================================

class SatelliteScaler:
    """
    Per-channel normalisation honouring ESA-ADB signal-type distinctions
    (paper Sec. 4, "Standardisation"):

    Channel types and their treatment:
      1. Normal continuous  → StandardScaler on nominal train samples.
      2. Binary (only 2 unique values) → normalise to [0, 1].
      3. Constant (σ=0)    → subtract mean only.
      4. Monotonic counter  → first-difference before scaling.
      5. Categorical        → label-encode by order-of-first-occurrence, then scale.

    "Nominal samples" = train samples NOT labelled as anomalous,
    to avoid the anomaly distribution leaking into the scaler fit.
    """

    def __init__(self, channels_meta: pd.DataFrame):
        self.meta = channels_meta.set_index("channel_name")
        self.scalers: Dict[str, object] = {}
        self.channel_types: Dict[str, str] = {}
        self.cat_maps: Dict[str, Dict] = {}       # categorical → int mapping
        self.monotonic_cols: List[str] = []

    def _infer_channel_type(self, col: str, train_series: pd.Series) -> str:
        n_unique = train_series.nunique()
        if n_unique <= 2:
            return "binary"
        if train_series.std() == 0:
            return "constant"
        # Monotonic: strictly non-decreasing or non-increasing
        diff = train_series.diff().dropna()
        if (diff >= 0).all() or (diff <= 0).all():
            return "monotonic"
        # Attempt to detect categorical (integer-like with many repeated values)
        if train_series.dtype == object or (
            n_unique < 30 and (train_series % 1 == 0).all()
        ):
            return "categorical"
        return "continuous"

    def fit(self, train_data: pd.DataFrame, train_labels: pd.Series):
        """
        Fit scalers using NOMINAL samples only (label=0) from training data.
        This is critical — anomaly values must not influence the scaler.
        """
        nominal_mask = train_labels == 0
        nominal_data = train_data[nominal_mask]

        for col in train_data.columns:
            series = nominal_data[col].dropna()
            ch_type = self._infer_channel_type(col, series)
            self.channel_types[col] = ch_type

            if ch_type == "binary":
                mn, mx = series.min(), series.max()
                self.scalers[col] = {"min": mn, "range": max(mx - mn, 1e-9)}

            elif ch_type == "constant":
                self.scalers[col] = {"mean": series.mean()}

            elif ch_type == "monotonic":
                self.monotonic_cols.append(col)
                diff_series = series.diff().dropna()
                sc = StandardScaler()
                sc.fit(diff_series.values.reshape(-1, 1))
                self.scalers[col] = sc

            elif ch_type == "categorical":
                # Encode by order of first occurrence in train
                order = list(dict.fromkeys(nominal_data[col].dropna().tolist()))
                cat_map = {v: i for i, v in enumerate(order)}
                self.cat_maps[col] = cat_map
                # After encoding, standardise
                encoded = series.map(cat_map).fillna(0)
                sc = StandardScaler()
                sc.fit(encoded.values.reshape(-1, 1))
                self.scalers[col] = (cat_map, sc)

            else:  # continuous
                sc = StandardScaler()
                sc.fit(series.values.reshape(-1, 1))
                self.scalers[col] = sc

        print(f"✓  Scaler fitted — "
              f"continuous: {sum(t=='continuous' for t in self.channel_types.values())}, "
              f"binary: {sum(t=='binary' for t in self.channel_types.values())}, "
              f"monotonic: {sum(t=='monotonic' for t in self.channel_types.values())}, "
              f"categorical: {sum(t=='categorical' for t in self.channel_types.values())}, "
              f"constant: {sum(t=='constant' for t in self.channel_types.values())}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers to any split (val / test)."""
        out = data.copy().astype(np.float32)

        for col in data.columns:
            ch_type = self.channel_types.get(col, "continuous")
            sc = self.scalers.get(col)
            if sc is None:
                continue

            if ch_type == "binary":
                out[col] = (data[col] - sc["min"]) / sc["range"]

            elif ch_type == "constant":
                out[col] = data[col] - sc["mean"]

            elif ch_type == "monotonic":
                # First-difference, then scale; prepend 0 to preserve length
                diff = data[col].diff().fillna(0)
                out[col] = sc.transform(diff.values.reshape(-1, 1)).flatten()

            elif ch_type == "categorical":
                cat_map, std_sc = sc
                encoded = data[col].map(cat_map).fillna(-1)
                out[col] = std_sc.transform(
                    encoded.values.reshape(-1, 1)).flatten()

            else:  # continuous
                out[col] = sc.transform(
                    data[col].values.reshape(-1, 1)).flatten()

        return out

    def fit_transform(self, train_data: pd.DataFrame,
                      train_labels: pd.Series) -> pd.DataFrame:
        self.fit(train_data, train_labels)
        return self.transform(train_data)


# ===========================================================================
# 6.  POINT-ANOMALY CORRECTION AFTER RESAMPLING
# ===========================================================================

def apply_point_anomaly_correction(
    labels: pd.Series,
    raw_labels_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    After ZOH resampling, very short (point) anomalies may fall between two
    resampled timestamps and be silently dropped.

    Paper Sec. 4 Step 3 (resampling): "Apply a correction for missing anomalies
    to ensure that no point events are removed due to the resampling. Iterate
    through consecutive pairs of unannotated timestamps in the resampled list
    and, if there are any annotated original points in between, take the last
    annotated sample and assign its value and label to the latter timestamp
    from the pair."

    Here we ensure any resampled timestamp immediately AFTER a raw anomaly
    point gets label=1 if the raw point fell between two resampled timestamps.
    """
    # If labels already cover point anomalies (common case), this is a no-op.
    corrected = labels.copy()
    # Unannotated resampled pairs with a raw anomaly in between:
    for i in range(len(target_index) - 1):
        t0, t1 = target_index[i], target_index[i + 1]
        if labels[t0] == 0 and labels[t1] == 0:
            # Any raw label between t0 and t1?
            in_between = raw_labels_df[
                (raw_labels_df.index > t0) & (raw_labels_df.index < t1)
            ]
            if len(in_between) > 0 and (in_between > 0).any().any():
                corrected[t1] = 1   # assign to the latter timestamp

    corrected_count = (corrected - labels).sum()
    if corrected_count > 0:
        print(f"✓  Point-anomaly correction: {corrected_count} timestamps fixed.")
    return corrected


# ===========================================================================
# 7.  SLIDING-WINDOW SEQUENCE CONSTRUCTION
# ===========================================================================
def create_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    seq_len: int = 128,
    stride: int = 1,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bypassed for full dataset: Saves flat 2D arrays to SSD 
    to prevent a 344 GB memory explosion.
    """
    print(f"✓  Bypassing 3D windowing. Saving flat 2D arrays — X: {data.shape}  y: {labels.shape}")
    return data.astype(np.float32), labels.astype(np.int64)

# ===========================================================================
# 8.  PYTORCH DATASET  (lazy, memory-efficient)
# ===========================================================================

class SatelliteAnomalyDataset(Dataset):
    """
    PyTorch Dataset for satellite telemetry anomaly detection.
    (Simplified for pre-built tensors)
    """
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 128,
        stride: int = 1,
        horizon: int = 1,
        metadata: Optional[List[Dict]] = None,
    ):
        self.data     = data
        self.labels   = labels
        self.metadata = metadata 

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        # Directly grab the pre-built sliding window and label!
        x = torch.from_numpy(self.data[idx])   
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.metadata is not None:
            return x, y, self.metadata[idx]
        return x, y

def make_dataloaders(
    train_X: np.ndarray, train_y: np.ndarray,
    val_X: np.ndarray,   val_y: np.ndarray,
    test_X: np.ndarray,  test_y: np.ndarray,
    seq_len: int = 128,
    stride: int = 1,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for all three splits.
    Shuffle is True ONLY for training (never for val/test — preserves time order).
    """
    train_ds = SatelliteAnomalyDataset(train_X, train_y, seq_len, stride)
    val_ds   = SatelliteAnomalyDataset(val_X,   val_y,   seq_len, stride)
    test_ds  = SatelliteAnomalyDataset(test_X,  test_y,  seq_len, stride)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl


# ===========================================================================
# 9.  LLM EXPLANATION METADATA
# ===========================================================================

def build_llm_metadata(
    test_data: pd.DataFrame,
    test_labels: pd.Series,
    labels_df: pd.DataFrame,
    anomaly_types: pd.DataFrame,
    channels_meta: pd.DataFrame,
    seq_len: int,
) -> List[Dict]:
    """
    For each anomalous window in the test set, construct a structured dict
    suitable for feeding directly into an LLM explanation prompt.

    Structure per entry:
    {
        "window_start"       : ISO timestamp of window start,
        "window_end"         : ISO timestamp of window end,
        "anomaly_present"    : bool,
        "anomaly_ids"        : list[str] — event IDs overlapping this window,
        "anomaly_classes"    : list[str] — human-readable class names,
        "anomaly_categories" : list[str] — anomaly | rare_nominal_event,
        "anomaly_types_str"  : list[str] — point/subsequence, uni/multivariate, local/global,
        "affected_channels"  : list[str],
        "subsystems"         : list[str] — inferred from channels_meta,
        "channel_stats"      : dict — per-channel mean/std in this window (for context),
    }

    LLM PROMPT TEMPLATE (use metadata dict directly):
    --------------------------------------------------
    system: "You are a spacecraft anomaly analyst..."
    user: f"The following satellite telemetry window has been flagged as anomalous.
           Window: {meta['window_start']} to {meta['window_end']}.
           Event class: {meta['anomaly_classes']}.
           Affected subsystems: {meta['subsystems']}.
           Channel statistics: {meta['channel_stats']}.
           Explain the likely cause and recommend operator action."
    """
    id_to_meta = anomaly_types.set_index("ID")
    ch_to_subsystem = channels_meta.set_index("channel_name")["subsystem"].to_dict()

    metadata_list = []
    timestamps = test_data.index

    T = len(test_data)
    n_windows = max(0, T - seq_len)

    for i in range(n_windows):
        t_start = timestamps[i]
        t_end   = timestamps[i + seq_len - 1]
        label   = int(test_labels.iloc[i + seq_len - 1])

        if label == 0:
            metadata_list.append({"anomaly_present": False})
            continue

        # Find anomaly events overlapping this window
        window_labels = labels_df[
            (labels_df["start_time"] <= t_end) &
            (labels_df["end_time"]   >= t_start)
        ]
        event_ids = window_labels["ID"].unique().tolist()

        anomaly_classes, categories, type_strs, affected_channels = [], [], [], []
        for eid in event_ids:
            if eid in id_to_meta.index:
                row = id_to_meta.loc[eid]
                anomaly_classes.append(str(row.get("class", "unknown")))
                categories.append(str(row.get("category", "unknown")))
                type_strs.append(str(row.get("type", "unknown")))
            aff = window_labels[window_labels["ID"] == eid]["channel_name"].tolist()
            affected_channels.extend(aff)

        affected_channels = list(set(affected_channels))
        subsystems = list({ch_to_subsystem.get(ch, "unknown")
                           for ch in affected_channels})

        # Compact channel statistics (mean ± std in this window)
        window_slice = test_data.iloc[i: i + seq_len]
        ch_stats = {
            col: {
                "mean": round(float(window_slice[col].mean()), 4),
                "std" : round(float(window_slice[col].std()),  4),
            }
            for col in affected_channels if col in test_data.columns
        }

        metadata_list.append({
            "window_start"       : t_start.isoformat(),
            "window_end"         : t_end.isoformat(),
            "anomaly_present"    : True,
            "anomaly_ids"        : event_ids,
            "anomaly_classes"    : anomaly_classes,
            "anomaly_categories" : categories,
            "anomaly_types_str"  : type_strs,
            "affected_channels"  : affected_channels,
            "subsystems"         : subsystems,
            "channel_stats"      : ch_stats,
        })

    print(f"✓  LLM metadata built for {len(metadata_list)} windows "
          f"({sum(m.get('anomaly_present', False) for m in metadata_list)} anomalous).")
    return metadata_list


# ===========================================================================
# 10. EXPERIMENT CONFIGURATIONS: Telemetry-only vs. Telemetry + Telecommands
# ===========================================================================

def build_feature_matrix(
    telemetry: pd.DataFrame,
    tc_features: pd.DataFrame,
    mode: str = "telemetry_only",
) -> pd.DataFrame:
    """
    Assemble the final feature matrix for experiments.

    Parameters
    ----------
    mode :
      "telemetry_only"         → only channel measurements (target + non-target)
      "telemetry_plus_tc"      → channel measurements + telecommand binary flags

    This design mirrors the ablation in the paper: DC-VAE-ESA and Telemanom-ESA
    are trained both with and without TC features (priority-3 TCs for full sets).
    """
    if mode == "telemetry_only":
        print(f"✓  Feature matrix mode: telemetry_only — {telemetry.shape[1]} features")
        return telemetry.copy()
    elif mode == "telemetry_plus_tc":
        combined = pd.concat([telemetry, tc_features], axis=1)
        print(f"✓  Feature matrix mode: telemetry_plus_tc — "
              f"{telemetry.shape[1]} telemetry + {tc_features.shape[1]} TC = "
              f"{combined.shape[1]} total features")
        return combined
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'telemetry_only' or 'telemetry_plus_tc'.")


# ===========================================================================
# 11. MASTER PIPELINE FUNCTION
# ===========================================================================
def split_dataset(
    data: pd.DataFrame,
    labels: pd.Series,
    val_months: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Chronological split modified for PoC (60% Train, 20% Val, 20% Test).
    """
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_data = data.iloc[:train_end]
    val_data   = data.iloc[train_end:val_end]
    test_data  = data.iloc[val_end:]

    train_labels = labels.iloc[:train_end]
    val_labels   = labels.iloc[train_end:val_end]
    test_labels  = labels.iloc[val_end:]

    print(f"✓  Split — Train: {len(train_data):,}  Val: {len(val_data):,}  "
          f"Test: {len(test_data):,} timestamps")
    if len(train_labels) > 0:
        print(f"   Train anomaly rate: {train_labels.mean()*100:.2f}%  |  "
              f"Test anomaly rate: {test_labels.mean()*100:.2f}%")
    return (train_data, val_data, test_data,
            train_labels, val_labels, test_labels)
def run_pipeline(
    mission_dir: str,
    mode: str = "telemetry_only",      # "telemetry_only" | "telemetry_plus_tc"
    tc_priority: int = 2,
    seq_len: int = 128,
    stride: int = 1,
    batch_size: int = 64,
    build_llm_meta: bool = True,
) -> Dict:
    """
    End-to-end preprocessing pipeline.

    Returns a dict with keys:
      train_dl, val_dl, test_dl        : PyTorch DataLoaders
      train_X, val_X, test_X           : numpy arrays (N, seq_len, F)
      train_y, val_y, test_y           : numpy arrays (N,)
      scaler                           : fitted SatelliteScaler (for inverse transform)
      feature_names                    : list[str] — column order of features
      llm_metadata                     : list[dict] (if build_llm_meta=True)
      class_to_int                     : dict — multiclass label mapping
    """
    print("\n" + "="*60)
    print(f"ESA-ADB Preprocessing Pipeline")
    print(f"Mode: {mode}   |   seq_len: {seq_len}   |   TC priority ≥ {tc_priority}")
    print("="*60 + "\n")

    mission_dir = Path(mission_dir)

    # ── 1. Load metadata ───────────────────────────────────────────────────
    channels_meta, tc_meta, anomaly_types = load_metadata(mission_dir)
    labels_df = load_labels(mission_dir)

    # ── 2. Load raw time series ────────────────────────────────────────────
    channel_data = load_channels(
        mission_dir / "channels", channels_meta
    )
    tc_data = load_telecommands(
        mission_dir / "telecommands", tc_meta, priority_threshold=tc_priority
    ) if mode == "telemetry_plus_tc" else {}

    # ── 3. Resample all to uniform grid (ZOH) ─────────────────────────────
    telemetry_df = resample_all_channels(channel_data, CFG["resample_freq"])
    target_index = telemetry_df.index

    # ── 4. Encode telecommands onto resampled grid ─────────────────────────
    tc_df = encode_telecommands(tc_data, target_index)

    # ── 5. Build labels ────────────────────────────────────────────────────
    binary_labels, channel_label_df = build_label_matrix(
        labels_df, anomaly_types, target_index, channels_meta
    )
    mc_labels, class_to_int = build_multiclass_labels(
        labels_df, anomaly_types, target_index
    )

    # ── 6. Assemble feature matrix ─────────────────────────────────────────
    feature_df = build_feature_matrix(telemetry_df, tc_df, mode=mode)
    feature_names = list(feature_df.columns)

    # ── 7. Chronological split ─────────────────────────────────────────────
    (train_feat, val_feat, test_feat,
     train_lbl,  val_lbl,  test_lbl) = split_dataset(
        feature_df, binary_labels, val_months=CFG["val_months"]
    )

    # ── 8. Normalise (fit on nominal train only) ───────────────────────────
    scaler = SatelliteScaler(channels_meta)
    train_norm = scaler.fit_transform(train_feat, train_lbl)
    val_norm   = scaler.transform(val_feat)
    test_norm  = scaler.transform(test_feat)

    # ── 9. Sliding-window sequences ────────────────────────────────────────
    train_X, train_y = create_sequences(
        train_norm.values, train_lbl.values, seq_len, stride
    )
    val_X,   val_y   = create_sequences(
        val_norm.values,   val_lbl.values,   seq_len, stride
    )
    test_X,  test_y  = create_sequences(
        test_norm.values,  test_lbl.values,  seq_len, stride
    )

    # ── 10. PyTorch DataLoaders ────────────────────────────────────────────
    train_dl, val_dl, test_dl = make_dataloaders(
        train_X, train_y, val_X, val_y, test_X, test_y,
        seq_len=seq_len, stride=stride, batch_size=batch_size
    )

    # ── 11. LLM explanation metadata (test set only) ──────────────────────
    llm_metadata = None
    if build_llm_meta:
        llm_metadata = build_llm_metadata(
            test_feat, test_lbl, labels_df, anomaly_types,
            channels_meta, seq_len
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("PIPELINE COMPLETE — Final Data Shapes:")
    print(f"  train_X : {train_X.shape}   train_y : {train_y.shape}")
    print(f"  val_X   : {val_X.shape}     val_y   : {val_y.shape}")
    print(f"  test_X  : {test_X.shape}    test_y  : {test_y.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes : {len(class_to_int)} anomaly classes")
    print("-"*60 + "\n")

    return {
        "train_dl"      : train_dl,
        "val_dl"        : val_dl,
        "test_dl"       : test_dl,
        "train_X"       : train_X,
        "val_X"         : val_X,
        "test_X"        : test_X,
        "train_y"       : train_y,
        "val_y"         : val_y,
        "test_y"        : test_y,
        "scaler"        : scaler,
        "feature_names" : feature_names,
        "llm_metadata"  : llm_metadata,
        "class_to_int"  : class_to_int,
        "channel_label_df": channel_label_df,   # for channel-aware F-score
        "mc_labels"     : mc_labels,            # multi-class labels (full timeline)
    }


# ===========================================================================
# 12. ABLATION EXPERIMENT RUNNER
# ===========================================================================

def run_ablation_experiments(mission_dir: str) -> Dict[str, Dict]:
    """
    Run both experimental configurations required for comparison experiments.

    Config A: Telemetry-only
    Config B: Telemetry + Telecommands (priority ≥ 2)

    Returns dict with results for both configs, ready for evaluation.
    """
    results = {}

    for mode in ["telemetry_only", "telemetry_plus_tc"]:
        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT: {mode.upper()}")
        print(f"{'#'*60}")
        results[mode] = run_pipeline(
            mission_dir  = mission_dir,
            mode         = mode,
            tc_priority  = CFG["tc_priority_threshold"],
            seq_len      = CFG["seq_len"],
            stride       = CFG["stride"],
            batch_size   = 64,
            build_llm_meta = True,
        )

    # Feature count comparison
    n_tele = results["telemetry_only"]["train_X"].shape[-1]
    n_full = results["telemetry_plus_tc"]["train_X"].shape[-1]
    print(f"\nAblation summary:")
    print(f"  Telemetry-only features : {n_tele}")
    print(f"  Telemetry + TC features : {n_full} "
          f"(+{n_full - n_tele} telecommand channels)")
    # Feature count comparison
    n_tele = results["telemetry_only"]["train_X"].shape[-1]
    n_full = results["telemetry_plus_tc"]["train_X"].shape[-1]
    print(f"\nAblation summary:")
    print(f"  Telemetry-only features : {n_tele}")
    print(f"  Telemetry + TC features : {n_full} "
          f"(+{n_full - n_tele} telecommand channels)")

    # ── NEW CODE: SAVE TO SSD ─────────────────────────────────────
    import os
    import numpy as np
    
    dataset_dict = results["telemetry_plus_tc"]
    save_dir = "processed_76_channels"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n[PIPELINE] Saving full dataset tensors to ./{save_dir}/ ...")
    
    # Save Training Set
    np.save(f"{save_dir}/train_X.npy", dataset_dict["train_X"])
    np.save(f"{save_dir}/train_y.npy", dataset_dict["train_y"])
    
    # Save Validation Set
    np.save(f"{save_dir}/val_X.npy", dataset_dict["val_X"])
    np.save(f"{save_dir}/val_y.npy", dataset_dict["val_y"])
    
    # Save Test Set
    np.save(f"{save_dir}/test_X.npy", dataset_dict["test_X"])
    np.save(f"{save_dir}/test_y.npy", dataset_dict["test_y"])
    
    print("[PIPELINE] Data successfully offloaded to hard drive!")
    # ──────────────────────────────────────────────────────────────

    
    return results


# ===========================================================================
# 13. FEATURE SELECTION UTILITIES
# ===========================================================================

def suggest_feature_selection(
    channels_meta: pd.DataFrame,
    labels_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    binary_labels: pd.Series,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Recommend important channels using two complementary heuristics:

    1. Target-channel flag (from channels.csv) — mandatory in ESA-ADB.
    2. Point-biserial correlation between channel values and anomaly labels
       (computed on training portion only — no leakage).

    Returns a DataFrame ranking channels by estimated importance.
    This is a starting point; more sophisticated methods (e.g., SHAP on a
    trained model) are recommended for the final paper.
    """
    from scipy.stats import pointbiserialr

    target_set = set(
        channels_meta[channels_meta["is_target"] == True]["channel_name"]
    )

    scores = []
    for col in telemetry_df.columns:
        try:
            r, p = pointbiserialr(
                binary_labels.values,
                telemetry_df[col].fillna(0).values
            )
        except Exception:
            r, p = 0.0, 1.0

        scores.append({
            "channel"   : col,
            "is_target" : col in target_set,
            "pb_corr"   : abs(r),
            "p_value"   : p,
            "subsystem" : channels_meta.set_index("channel_name")
                          .get("subsystem", {}).get(col, "unknown"),
        })

    ranking = (
        pd.DataFrame(scores)
        .sort_values(["is_target", "pb_corr"], ascending=[False, False])
        .reset_index(drop=True)
    )

    print(f"\nTop-{top_k} channels by anomaly correlation:")
    print(ranking.head(top_k).to_string(index=False))
    return ranking


# ===========================================================================
# 14.  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    # ── Sanity check mode (runs with synthetic mini-data if real data absent) ──
    import sys

    if len(sys.argv) > 1:
        mission_dir = sys.argv[1]
        results = run_ablation_experiments(mission_dir)
    else:
        print("Usage: python esa_adb_pipeline.py /path/to/ESA-Mission1")
        print("\nRunning SYNTHETIC SANITY CHECK...")
        _run_synthetic_test()


def _run_synthetic_test():
    """
    Self-contained smoke test using synthetic data matching the ESA-ADB schema.
    Validates that every pipeline stage executes without error.
    """
    import tempfile, shutil

    print("Building synthetic ESA-ADB-shaped dataset...")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "channels").mkdir()
    (tmp / "telecommands").mkdir()

    # Build ~200 synthetic timestamps at 30-second intervals
    n_pts = 500
    base_ts = pd.date_range("2000-01-01", periods=n_pts, freq="30S")

    # Write 6 synthetic channel files (mimicking channels 41-46, lightweight subset)
    ch_names = [f"channel_{i}" for i in range(41, 47)]
    for ch in ch_names:
        vals = np.random.randn(n_pts).cumsum() * 0.1
        df = pd.DataFrame({"value": vals}, index=base_ts)
        path = tmp / "channels" / f"{ch}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f"{ch}.pkl", "w") as f:
                pickle.dump(df, f, protocol=4)

    # Write 2 synthetic telecommand files
    for tc in ["tc_001", "tc_002"]:
        exec_times = base_ts[[50, 150, 300]]
        tc_df = pd.DataFrame({"value": 1.0}, index=exec_times)
        path = tmp / "telecommands" / f"{tc}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f"{tc}.pkl", "w") as f:
                pickle.dump(tc_df, f, protocol=4)

    # Write metadata CSVs
    pd.DataFrame({
        "channel_name" : ch_names,
        "subsystem"    : ["sys_1"] * 6,
        "physical_unit": ["unit_4"] * 6,
        "group"        : [8] * 6,
        "is_target"    : [True] * 6,
    }).to_csv(tmp / "channels.csv", index=False)

    pd.DataFrame({
        "tc_name" : ["tc_001", "tc_002"],
        "priority": [3, 2],
    }).to_csv(tmp / "telecommands.csv", index=False)

    pd.DataFrame({
        "ID"          : [1, 1, 2],
        "channel_name": ["channel_41", "channel_42", "channel_44"],
        "start_time"  : [base_ts[80], base_ts[80], base_ts[200]],
        "end_time"    : [base_ts[95], base_ts[95], base_ts[220]],
    }).to_csv(tmp / "labels.csv", index=False)

    pd.DataFrame({
        "ID"      : [1, 2],
        "class"   : ["attitude_disturbance", "latch_up"],
        "subclass": ["type_a", "type_b"],
        "category": ["anomaly", "anomaly"],
        "type"    : ["multivariate_global_subsequence",
                     "univariate_local_subsequence"],
    }).to_csv(tmp / "anomaly_types.csv", index=False)

    # Run both experiment modes
    try:
        results = run_ablation_experiments(str(tmp))
        print("\n✅  SANITY CHECK PASSED")
        print(f"   train_X shape (telemetry_only): "
              f"{results['telemetry_only']['train_X'].shape}")
        print(f"   train_X shape (telemetry+tc)  : "
              f"{results['telemetry_plus_tc']['train_X'].shape}")
        print(f"   LLM metadata sample: "
              f"{results['telemetry_only']['llm_metadata'][:1]}")
    finally:
        shutil.rmtree(tmp)
