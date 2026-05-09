#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    return path


def finite_array(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-15:
        return math.nan
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "samples": int(y_true.size),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def metrics_by_feature(y_true, y_pred, names, extra=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    for index, name in enumerate(names):
        row = dict(extra or {})
        row["feature"] = str(name)
        row["feature_index"] = int(index)
        row.update(regression_metrics(y_true[..., index].reshape(-1), y_pred[..., index].reshape(-1)))
        rows.append(row)
    return pd.DataFrame(rows)
