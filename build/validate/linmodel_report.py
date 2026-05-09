#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build.linmodel import conf as lin_conf
from build.linmodel.data_prep import build_split_dataset, load_csv_files, split_files
from build.linmodel.regression import angle_key, evaluate_split
from build.validate.metrics import ensure_dir, write_json
from build.validate.plotting import angle_tick_labels, plot_heatmap, save_figure
from conf import validation as validation_conf


def _load_artifact(path=None):
    path = Path(path or lin_conf.CURRENT_PARAMS_PATH)
    with path.open("r") as file:
        return path, json.load(file)


def _models_from_artifact(artifact):
    models = {}
    for angle in artifact["config"]["angles"]:
        raw = artifact["models"][angle_key(angle)]
        theta = np.asarray(raw["theta"], dtype=float)
        na = int(artifact["config"]["na"])
        nb = int(artifact["config"]["nb"])
        dyn_size = na + 2 * nb
        models[int(angle)] = {
            "theta": theta,
            "a_coeffs": theta[:na],
            "b_v": theta[na:na + nb],
            "b_delta": theta[na + nb:dyn_size],
            "ramp_coeffs": theta[dyn_size:],
            "feature_width": int(theta.size),
            "sample_count": int(raw.get("sample_count", 0)),
        }
    return models


def _dataset_kwargs(artifact, source_kind):
    config = artifact["config"]
    interval_len = int(round(float(config["interval_duration"]) / float(config["Ts"])))
    return {
        "angles_deg": config["angles"],
        "na": int(config["na"]),
        "nb": int(config["nb"]),
        "v_nom": float(config["V_nom"]),
        "delta_nom": float(config["delta_nom"]),
        "d_nom": float(config["D_nom"]),
        "interval_len": interval_len,
        "ts": float(config["Ts"]),
        "source_kind": source_kind,
        "lidar_delta_mode": config.get("lidar_delta_mode", lin_conf.LIDAR_DELTA_MODE),
    }


def _evaluate_dataset(dataset_name, data_dir, artifact, models, max_files=None):
    files = load_csv_files(data_dir, pattern=lin_conf.FILE_PATTERN, max_files=max_files)
    split = split_files(files)
    kwargs = _dataset_kwargs(artifact, source_kind=dataset_name)
    split_files_by_name = {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }
    metrics_frames = []
    predictions = {}
    for split_name, paths in split_files_by_name.items():
        data = build_split_dataset(paths, **kwargs)
        metrics, preds = evaluate_split(data, models=models, angles_deg=artifact["config"]["angles"])
        metrics["dataset"] = dataset_name
        metrics["split"] = split_name
        metrics["file_count"] = len(paths)
        metrics_frames.append(metrics)
        predictions[(dataset_name, split_name)] = preds
    return pd.concat(metrics_frames, ignore_index=True), predictions


def _plot_rmse(metrics, output_dir):
    angles = sorted(metrics["angle"].unique())
    rows = []
    labels = []
    for dataset in sorted(metrics["dataset"].unique()):
        for split in ["train", "val", "test"]:
            frame = metrics[(metrics["dataset"] == dataset) & (metrics["split"] == split)]
            if frame.empty:
                continue
            rows.append([float(frame[frame["angle"] == angle]["rmse"].iloc[0]) for angle in angles])
            labels.append(f"{dataset}/{split}")
    return plot_heatmap(
        rows,
        angle_tick_labels(angles),
        labels,
        "Linear ARX RMSE by angle",
        "RMSE [m]",
        output_dir,
        "linmodel_rmse_heatmap",
        cmap="magma",
    )


def _plot_r2(metrics, output_dir):
    angles = sorted(metrics["angle"].unique())
    rows = []
    labels = []
    for dataset in sorted(metrics["dataset"].unique()):
        for split in ["train", "val", "test"]:
            frame = metrics[(metrics["dataset"] == dataset) & (metrics["split"] == split)]
            if frame.empty:
                continue
            rows.append([float(frame[frame["angle"] == angle]["r2"].iloc[0]) for angle in angles])
            labels.append(f"{dataset}/{split}")
    return plot_heatmap(
        rows,
        angle_tick_labels(angles),
        labels,
        "Linear ARX R2 by angle",
        "R2",
        output_dir,
        "linmodel_r2_heatmap",
        cmap="viridis",
    )


def _plot_predictions(predictions, angles, output_dir):
    key = next((key for key in [("real", "test"), ("sim", "test")] if key in predictions), None)
    if key is None:
        return None
    preds = predictions[key]
    selected = [angle for angle in angles if int(angle) in preds]
    if not selected:
        return None
    max_plots = min(len(selected), 12)
    selected = selected[:max_plots]
    n_cols = 3
    n_rows = int(math.ceil(max_plots / n_cols))
    from build.validate import plotting

    plotting.apply_style()
    fig, axes = plotting.plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.8 * n_rows), squeeze=False)
    max_points = validation_conf.PREDICTION_MAX_POINTS
    for ax, angle in zip(axes.ravel(), selected):
        item = preds[int(angle)]
        count = min(max_points, item["Y_true"].size)
        x = np.arange(count)
        ax.plot(x, item["Y_true"][:count], label="true", linewidth=1.2)
        ax.plot(x, item["Y_pred"][:count], label="pred", linewidth=1.2, alpha=0.8)
        ax.set_title(f"{key[0]}/{key[1]} angle {int(angle)} deg")
        ax.set_xlabel("sample")
        ax.set_ylabel("delta lidar [m]")
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    axes[0, 0].legend()
    return save_figure(fig, output_dir, "linmodel_predictions")


def _plot_coefficients(models, artifact, output_dir):
    angles = [int(angle) for angle in artifact["config"]["angles"]]
    for key, title in [
        ("a_coeffs", "AR coefficients"),
        ("b_v", "speed input coefficients"),
        ("b_delta", "steering input coefficients"),
    ]:
        matrix = [models[angle][key] for angle in angles]
        plot_heatmap(
            matrix,
            [str(i) for i in range(len(matrix[0]))],
            angle_tick_labels(angles),
            f"Linear model {title}",
            "coefficient",
            output_dir,
            f"linmodel_{key}",
            cmap="coolwarm",
        )


def run(output_dir, artifact_path=None, max_files=validation_conf.LINMODEL_MAX_FILES, plots=True):
    output_dir = ensure_dir(Path(output_dir) / "linmodel")
    artifact_path, artifact = _load_artifact(artifact_path)
    models = _models_from_artifact(artifact)

    all_metrics = []
    all_predictions = {}
    for dataset_name, data_dir in lin_conf.DATASETS.items():
        metrics, predictions = _evaluate_dataset(
            dataset_name,
            data_dir,
            artifact,
            models,
            max_files=max_files,
        )
        all_metrics.append(metrics)
        all_predictions.update(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics_path = output_dir / "linmodel_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    figure_paths = []
    if plots:
        figure_paths.extend([
            _plot_rmse(metrics, output_dir),
            _plot_r2(metrics, output_dir),
            _plot_predictions(all_predictions, artifact["config"]["angles"], output_dir),
        ])
        _plot_coefficients(models, artifact, output_dir)
        figure_paths.extend([
            output_dir / "linmodel_a_coeffs.png",
            output_dir / "linmodel_b_v.png",
            output_dir / "linmodel_b_delta.png",
        ])
        figure_paths = [str(path) for path in figure_paths if path is not None]

    summary = {
        "artifact": str(artifact_path),
        "metrics_csv": str(metrics_path),
        "figures": figure_paths,
        "mean_rmse": float(metrics["rmse"].mean()),
        "mean_r2": float(metrics["r2"].mean()),
    }
    write_json(output_dir / "linmodel_summary.json", summary)
    return summary
