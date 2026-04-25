#!/usr/bin/env python3

import math

import numpy as np
import pandas as pd

try:
    from . import conf
except ImportError:
    import conf


def angle_key(angle):
    value = float(angle)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def align_feature_width(X, width):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if X.shape[1] == width:
        return X
    if X.shape[1] > width:
        return X[:, :width]
    pad = np.zeros((X.shape[0], width - X.shape[1]), dtype=float)
    return np.hstack([X, pad])


def solve_least_squares(X, Y, alpha=conf.REG_ALPHA):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[0] == 0:
        raise ValueError("cannot fit a model with zero samples")

    gram = X.T @ X
    reg = float(alpha) * np.eye(gram.shape[0])
    rhs = X.T @ Y
    return np.linalg.solve(gram + reg, rhs)


def predict_linear(X, theta):
    return align_feature_width(X, len(theta)) @ np.asarray(theta, dtype=float)


def fit_models_per_angle(train_data, angles_deg=conf.ANGLES_DEG, na=conf.NA, nb=conf.NB, alpha=conf.REG_ALPHA):
    models = {}
    for angle in angles_deg:
        item = train_data[int(angle)]
        X = item["X"]
        Y = item["Y"]
        theta = solve_least_squares(X, Y, alpha=alpha)
        dyn_size = int(na) + 2 * int(nb)
        models[int(angle)] = {
            "theta": theta,
            "a_coeffs": theta[: int(na)],
            "b_v": theta[int(na): int(na) + int(nb)],
            "b_delta": theta[int(na) + int(nb): dyn_size],
            "ramp_coeffs": theta[dyn_size:],
            "feature_width": int(len(theta)),
            "sample_count": int(X.shape[0]),
        }
    return models


def _safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-15:
        return math.nan
    return 1.0 - ss_res / ss_tot


def evaluate_split(split_data, models, angles_deg=conf.ANGLES_DEG):
    rows = []
    predictions = {}
    for angle in angles_deg:
        angle = int(angle)
        item = split_data[angle]
        model = models[angle]
        Y_true = np.asarray(item["Y"], dtype=float)
        if Y_true.size == 0:
            continue
        Y_pred = predict_linear(item["X"], model["theta"])
        err = Y_pred - Y_true
        rows.append({
            "angle": angle,
            "samples": int(Y_true.size),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "r2": float(_safe_r2(Y_true, Y_pred)),
        })
        predictions[angle] = {
            "Y_true": Y_true,
            "Y_pred": Y_pred,
            "error": err,
        }
    return pd.DataFrame(rows), predictions


def summarize_model_coefficients(models, v_obs_ref=conf.V_OBS_REF):
    global_rows = []
    ramp_rows = []

    for angle, model in models.items():
        for index, value in enumerate(model["a_coeffs"]):
            global_rows.append({
                "angle": int(angle),
                "kind": "a",
                "index": int(index),
                "value": float(value),
            })
        for index, value in enumerate(model["b_v"]):
            global_rows.append({
                "angle": int(angle),
                "kind": "b_v",
                "index": int(index),
                "value": float(value),
            })
        for index, value in enumerate(model["b_delta"]):
            global_rows.append({
                "angle": int(angle),
                "kind": "b_delta",
                "index": int(index),
                "value": float(value),
            })
        for index, value in enumerate(model["ramp_coeffs"]):
            ramp_rows.append({
                "angle": int(angle),
                "interval_id": int(index),
                "ramp_coeff": float(value),
                "gamma_interval": float(value / v_obs_ref) if abs(v_obs_ref) > 1e-12 else math.nan,
            })

    return pd.DataFrame(global_rows), pd.DataFrame(ramp_rows)


def serialize_models(models, config, metadata=None, metrics=None):
    data = {
        "schema_version": 2,
        "config": dict(config),
        "metadata": dict(metadata or {}),
        "models": {},
    }
    if metrics is not None:
        data["metrics"] = metrics

    for angle, model in models.items():
        data["models"][angle_key(angle)] = {
            "a_coeffs": np.asarray(model["a_coeffs"], dtype=float).tolist(),
            "b_v": np.asarray(model["b_v"], dtype=float).tolist(),
            "b_delta": np.asarray(model["b_delta"], dtype=float).tolist(),
            "theta": np.asarray(model["theta"], dtype=float).tolist(),
            "ramp_coeffs": np.asarray(model["ramp_coeffs"], dtype=float).tolist(),
            "sample_count": int(model.get("sample_count", 0)),
        }

    return data
