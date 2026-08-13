# Anomaly Transformer-Based Satellite Telemetry Anomaly Detection

Anomaly Transformer-based satellite telemetry anomaly detection on the ESA-ADB benchmark, with custom preprocessing, telecommand normalization, and evaluation.

## Overview

This repository contains the implementation and experimental results for my M.Tech thesis work on anomaly detection in satellite telemetry using the Anomaly Transformer.

The project adapts the Anomaly Transformer to the ESA-ADB Mission 1 satellite telemetry benchmark and develops a preprocessing pipeline for irregular multivariate telemetry data.

## Key Contributions

- Adapted the Anomaly Transformer for ESA satellite telemetry.
- Developed a Zero-Order Hold (ZOH) resampling pipeline for irregularly sampled channels.
- Designed channel-specific preprocessing and normalization.
- Investigated training instability caused by sparse telecommand channels.
- Evaluated anomaly detection using precision, recall, F1-score and anomaly scores.
- Generated attention, association discrepancy and anomaly-score visualizations.
- Prepared structured anomaly metadata for future LLM-based explanation analysis.

## Dataset

The experiments use the ESA-ADB Mission 1 satellite telemetry benchmark.

The dataset contains 106 channels consisting of telemetry and telecommand signals with highly imbalanced anomaly labels.

The raw dataset is not included in this repository. Please obtain the dataset from its original source before running the preprocessing pipeline.

## Pipeline

```text
ESA-ADB Dataset
       ↓
Data Loading
       ↓
Timestamp Alignment
       ↓
ZOH Resampling
       ↓
Channel Normalization
       ↓
Sliding-Window Dataset
       ↓
Anomaly Transformer
       ↓
Anomaly Scores
       ↓
Evaluation & Visualization
