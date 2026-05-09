#!/usr/bin/env python3

import glob
import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from build.validate.metrics import ensure_dir, metrics_by_feature, write_json
from build.validate.plotting import angle_tick_labels, plot_heatmap, plot_lines, save_figure
from conf import nntrain as nn_conf
from conf import validation as validation_conf


def _clean_state_dict(state_dict):
    return {
        key.replace("_orig_mod.", ""): value
        for key, value in state_dict.items()
    }


def _infer_output_width(state_dict):
    for key, value in state_dict.items():
        if key.endswith("output.weight") and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    return int(nn_conf.UQYQ_LSTM["output_width"])


def _tensor_stats(torch, width):
    return {
        "size": torch.Size([width]),
        "train_offset": torch.zeros(width, dtype=torch.float32),
        "train_scale": torch.ones(width, dtype=torch.float32),
    }


def _load_mylstm_class():
    nntrain_dir = Path(nn_conf.NNTRAIN_DIR)
    if str(nntrain_dir) not in sys.path:
        sys.path.insert(0, str(nntrain_dir))

    conf_path = nntrain_dir / "conf.py"
    old_conf = sys.modules.get("conf")
    spec = importlib.util.spec_from_file_location("conf", conf_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["conf"] = module
    spec.loader.exec_module(module)
    try:
        sys.modules.pop("pool.MyLSTM", None)
        imported = importlib.import_module("pool.MyLSTM")
        return imported.MyLSTM
    finally:
        if old_conf is not None:
            sys.modules["conf"] = old_conf
        else:
            sys.modules.pop("conf", None)


def _load_uqyq_files(pattern, max_trajectories=None):
    paths = sorted(Path(path) for path in glob.glob(str(pattern)))
    if max_trajectories is not None:
        paths = paths[: int(max_trajectories)]
    if not paths:
        raise FileNotFoundError(f"no UQYQ files matched {pattern}")
    uq_list = []
    yq_list = []
    uq_columns = None
    for path in paths:
        df = pd.read_csv(path)
        if df.shape[1] < 3:
            raise ValueError(f"{path} must contain at least one u_q column and two y_q columns")
        columns = list(df.columns[:-2])
        if uq_columns is None:
            uq_columns = columns
        elif columns != uq_columns:
            raise ValueError(f"{path} UQYQ columns differ from first file: {columns} != {uq_columns}")
        uq_list.append(df.iloc[:, :-2].to_numpy(dtype=np.float32))
        yq_list.append(df.iloc[:, -2:].to_numpy(dtype=np.float32))
    min_len = min(item.shape[0] for item in uq_list)
    uq = np.stack([item[:min_len] for item in uq_list])
    yq = np.stack([item[:min_len] for item in yq_list])
    return paths, uq_columns, uq, yq


def _window_starts(length, past_window, future_window, max_windows):
    start = int(past_window)
    stop = int(length) - int(future_window)
    if stop <= start:
        return np.zeros(0, dtype=int)
    all_starts = np.arange(start, stop + 1)
    if max_windows is not None and all_starts.size > int(max_windows):
        indices = np.linspace(0, all_starts.size - 1, int(max_windows)).round().astype(int)
        return np.unique(all_starts[indices])
    return all_starts


def _predict_windows(model, torch, uq, yq, batch_size):
    cfg = nn_conf.UQYQ_LSTM
    past_window = int(cfg["past_window"])
    future_window = int(cfg["future_window"])
    max_windows = validation_conf.NN_MAX_WINDOWS_PER_TRAJECTORY
    past_rows = []
    future_rows = []
    target_rows = []
    for traj_index in range(uq.shape[0]):
        for start in _window_starts(uq.shape[1], past_window, future_window, max_windows):
            past_rows.append(np.concatenate([yq[traj_index, start - past_window:start], uq[traj_index, start - past_window:start]], axis=1))
            future_rows.append(yq[traj_index, start:start + future_window])
            target_rows.append(uq[traj_index, start:start + future_window])

    if not past_rows:
        raise ValueError("not enough UQYQ samples to build validation windows")

    past = np.asarray(past_rows, dtype=np.float32)
    future = np.asarray(future_rows, dtype=np.float32)
    target = np.asarray(target_rows, dtype=np.float32)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, past.shape[0], int(batch_size)):
            past_batch = torch.from_numpy(past[start:start + batch_size])
            future_batch = torch.from_numpy(future[start:start + batch_size])
            preds.append(model(past_batch, future_batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0), target


def _plot_example(pred, target, feature_names, output_dir):
    from build.validate import plotting

    plotting.apply_style()
    count = min(len(feature_names), 12)
    n_cols = 3
    n_rows = int(np.ceil(count / n_cols))
    fig, axes = plotting.plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.8 * n_rows), squeeze=False)
    x = np.arange(target.shape[1])
    for ax, index in zip(axes.ravel(), range(count)):
        ax.plot(x, target[0, :, index], label="true", linewidth=1.2)
        ax.plot(x, pred[0, :, index], label="pred", linewidth=1.2, alpha=0.8)
        ax.set_title(feature_names[index])
        ax.set_xlabel("future step")
        ax.set_ylabel("u_q [m]")
    for ax in axes.ravel()[count:]:
        ax.axis("off")
    axes[0, 0].legend()
    return save_figure(fig, output_dir, "nn_example_prediction")


def run(
    output_dir,
    weights_path=None,
    uqyq_pattern=None,
    max_trajectories=validation_conf.NN_MAX_TRAJECTORIES,
    batch_size=validation_conf.NN_BATCH_SIZE,
    plots=True,
):
    output_dir = ensure_dir(Path(output_dir) / "nntrain")
    weights_path = Path(weights_path or nn_conf.MYLSTM_WEIGHTS_PATH)
    uqyq_pattern = uqyq_pattern or nn_conf.DATA_UQYQ_PATH

    import torch

    paths, uq_columns, uq, yq = _load_uqyq_files(uqyq_pattern, max_trajectories=max_trajectories)
    state_dict = _clean_state_dict(torch.load(weights_path, map_location="cpu"))
    output_width = _infer_output_width(state_dict)
    if uq.shape[2] != output_width:
        summary = {
            "status": "skipped",
            "reason": f"UQYQ width {uq.shape[2]} does not match MyLSTM output width {output_width}",
            "weights": str(weights_path),
            "uqyq_pattern": str(uqyq_pattern),
        }
        write_json(output_dir / "nntrain_summary.json", summary)
        return summary

    MyLSTM = _load_mylstm_class()
    dataset_stats = {
        nn_conf.CMD_SPEED: _tensor_stats(torch, 1),
        nn_conf.CMD_ANGLE: _tensor_stats(torch, 1),
        nn_conf.MES_LIDAR: _tensor_stats(torch, output_width),
    }
    model = MyLSTM(dataset_stats=dataset_stats, hidden_dim=int(nn_conf.UQYQ_LSTM["hidden_dim"]))
    model.load_state_dict(state_dict)

    prediction, target = _predict_windows(model, torch, uq, yq, batch_size=batch_size)
    feature_names = uq_columns or [f"u_q_{label}" for label in angle_tick_labels(range(output_width))]
    first_step_metrics = metrics_by_feature(target[:, 0, :], prediction[:, 0, :], feature_names, extra={"horizon": "first_step"})
    all_horizon_metrics = metrics_by_feature(target, prediction, feature_names, extra={"horizon": "all"})
    metrics = pd.concat([first_step_metrics, all_horizon_metrics], ignore_index=True)
    metrics_path = output_dir / "nntrain_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    horizon_rmse = np.sqrt(np.mean((prediction - target) ** 2, axis=(0, 2)))
    figure_paths = []
    if plots:
        first = first_step_metrics.sort_values("feature_index")
        figure_paths.append(str(plot_heatmap(
            [first["rmse"].to_numpy(dtype=float)],
            first["feature"].tolist(),
            ["first step"],
            "MyLSTM first-step RMSE by output",
            "RMSE [m]",
            output_dir,
            "nn_rmse_by_output",
            cmap="magma",
        )))
        figure_paths.append(str(plot_lines(
            np.arange(len(horizon_rmse)),
            {"future horizon RMSE": horizon_rmse},
            "MyLSTM prediction error over horizon",
            "future step",
            "RMSE [m]",
            output_dir,
            "nn_horizon_rmse",
        )))
        figure_paths.append(str(_plot_example(prediction, target, feature_names, output_dir)))

    summary = {
        "status": "ok",
        "weights": str(weights_path),
        "uqyq_pattern": str(uqyq_pattern),
        "uqyq_files": [str(path) for path in paths],
        "windows": int(prediction.shape[0]),
        "output_width": int(output_width),
        "first_step_rmse_mean": float(first_step_metrics["rmse"].mean()),
        "all_horizon_rmse_mean": float(all_horizon_metrics["rmse"].mean()),
        "metrics_csv": str(metrics_path),
        "figures": figure_paths,
    }
    write_json(output_dir / "nntrain_summary.json", summary)
    return summary
