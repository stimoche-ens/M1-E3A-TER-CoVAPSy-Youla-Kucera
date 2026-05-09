#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

from build.robustctl import conf as robust_conf
from build.robustctl.kcontroller import closed_loop_right_division, is_stable
from build.robustctl.linear_model import build_lidar_system, load_model_bank
from build.validate.metrics import ensure_dir, write_json
from build.validate.plotting import angle_tick_labels, plot_heatmap, plot_lines
from conf import validation as validation_conf
from lib.artifacts import project_path


def _load_artifact(path=None):
    path = Path(path or robust_conf.CONTROLLER_ARTIFACT_PATH)
    with path.open("r") as file:
        return path, json.load(file)


def _frequency_metrics(bank, K, samples):
    omega = np.linspace(0.0, np.pi, int(samples))
    open_gain = []
    closed_gain = []
    identity = np.eye(bank.n_inputs)
    for value in omega:
        H = bank.transfer_matrix(value)
        open_gain.append(float(np.linalg.norm(H, 2)))
        try:
            closed = H @ np.linalg.inv(identity + K @ H)
            closed_gain.append(float(np.linalg.norm(closed, 2)))
        except np.linalg.LinAlgError:
            closed_gain.append(float("inf"))
    return omega / np.pi, np.asarray(open_gain), np.asarray(closed_gain)


def run(output_dir, artifact_path=None, frequency_samples=validation_conf.ROBUST_FREQUENCY_SAMPLES, plots=True):
    output_dir = ensure_dir(Path(output_dir) / "robustctl")
    artifact_path, artifact = _load_artifact(artifact_path)
    K = np.asarray(artifact["controllers"]["K0"]["K"], dtype=float)
    angles = np.asarray(artifact["model"]["angles"], dtype=float)
    params_path = project_path(artifact["source"]["linear_parameters"])
    bank = load_model_bank(params_path)
    H = build_lidar_system(bank)
    closed_loop = closed_loop_right_division(H, K)
    freq, open_gain, closed_gain = _frequency_metrics(bank, K, frequency_samples)

    figure_paths = []
    if plots:
        figure_paths.append(str(plot_heatmap(
            K,
            angle_tick_labels(angles),
            ["delta speed", "delta steering"],
            "Static robust feedback K",
            "gain",
            output_dir,
            "robustctl_K_heatmap",
            cmap="coolwarm",
        )))
        figure_paths.append(str(plot_lines(
            freq,
            {
                "open plant ||H||2": open_gain,
                "closed H(I+KH)^-1": closed_gain,
            },
            "Robust controller frequency response",
            "normalized frequency omega/pi",
            "spectral norm",
            output_dir,
            "robustctl_frequency_response",
        )))

    summary = {
        "artifact": str(artifact_path),
        "linear_parameters": str(params_path),
        "stable": bool(is_stable(closed_loop)),
        "K_shape": list(K.shape),
        "K_norm_2": float(np.linalg.norm(K, 2)),
        "open_peak": float(np.nanmax(open_gain)),
        "closed_peak": float(np.nanmax(closed_gain)),
        "figures": figure_paths,
    }
    write_json(output_dir / "robustctl_summary.json", summary)
    return summary
