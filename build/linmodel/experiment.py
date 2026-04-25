#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

try:
    from . import conf
    from .data_prep import build_split_dataset, load_csv_files, split_files
    from .regression import (
        evaluate_split,
        fit_models_per_angle,
        serialize_models,
        summarize_model_coefficients,
    )
except ImportError:
    import conf
    from data_prep import build_split_dataset, load_csv_files, split_files
    from regression import (
        evaluate_split,
        fit_models_per_angle,
        serialize_models,
        summarize_model_coefficients,
    )


def _interval_len(interval_duration, ts):
    ratio = float(interval_duration) / float(ts)
    rounded = int(round(ratio))
    if not np.isclose(ratio, rounded, atol=1e-9):
        raise ValueError(
            f"interval duration {interval_duration} is not an integer multiple of Ts={ts}"
        )
    return rounded


def _mean_metric(frame, column):
    if frame.empty or column not in frame:
        return float("nan")
    return float(frame[column].mean())


def _metrics_to_records(metrics_by_split):
    return {
        split_name: frame.to_dict(orient="records")
        for split_name, frame in metrics_by_split.items()
    }


def run_one_experiment(
    files,
    interval_duration,
    d_nom,
    angles_deg=conf.ANGLES_DEG,
    na=conf.NA,
    nb=conf.NB,
    ts=conf.TS,
    v_nom=conf.V_NOM,
    delta_nom=conf.DELTA_NOM,
    reg_alpha=conf.REG_ALPHA,
    include_ramps=True,
    source_kind="auto",
    lidar_delta_mode=conf.LIDAR_DELTA_MODE,
):
    interval_len = _interval_len(interval_duration, ts)
    split = split_files(files)

    common_kwargs = {
        "angles_deg": angles_deg,
        "na": na,
        "nb": nb,
        "v_nom": v_nom,
        "delta_nom": delta_nom,
        "d_nom": d_nom,
        "interval_len": interval_len,
        "ts": ts,
        "include_ramps": include_ramps,
        "source_kind": source_kind,
        "lidar_delta_mode": lidar_delta_mode,
    }
    train_data = build_split_dataset(split.train, **common_kwargs)
    val_data = build_split_dataset(split.val, **common_kwargs)
    test_data = build_split_dataset(split.test, **common_kwargs)

    models = fit_models_per_angle(
        train_data=train_data,
        angles_deg=angles_deg,
        na=na,
        nb=nb,
        alpha=reg_alpha,
    )
    global_coeffs, ramp_coeffs = summarize_model_coefficients(models=models)
    train_metrics, _ = evaluate_split(train_data, models=models, angles_deg=angles_deg)
    val_metrics, _ = evaluate_split(val_data, models=models, angles_deg=angles_deg)
    test_metrics, _ = evaluate_split(test_data, models=models, angles_deg=angles_deg)
    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    summary = {
        "D_NOM": float(d_nom),
        "INTERVAL_DURATION": float(interval_duration),
        "INTERVAL_LEN": int(interval_len),
        "train_rmse_mean": _mean_metric(train_metrics, "rmse"),
        "val_rmse_mean": _mean_metric(val_metrics, "rmse"),
        "test_rmse_mean": _mean_metric(test_metrics, "rmse"),
        "val_r2_mean": _mean_metric(val_metrics, "r2"),
        "test_r2_mean": _mean_metric(test_metrics, "r2"),
        "gamma_std": _mean_metric(ramp_coeffs, "gamma_interval"),
        "n_train_files": len(split.train),
        "n_val_files": len(split.val),
        "n_test_files": len(split.test),
    }
    return {
        "summary": summary,
        "models": models,
        "metrics": metrics,
        "metrics_records": _metrics_to_records(metrics),
        "global_coefficients": global_coeffs,
        "ramp_coefficients": ramp_coeffs,
        "split": split,
    }


def run_grid_search(
    files,
    interval_durations=conf.INTERVAL_DURATION_LIST,
    d_nom_values=conf.D_NOM_LIST,
    **kwargs,
):
    runs = []
    for d_nom in d_nom_values:
        for interval_duration in interval_durations:
            runs.append(
                run_one_experiment(
                    files=files,
                    d_nom=d_nom,
                    interval_duration=interval_duration,
                    **kwargs,
                )
            )

    def score(run):
        value = run["summary"]["val_rmse_mean"]
        return value if np.isfinite(value) else float("inf")

    best = min(runs, key=score)
    return best, runs


def artifact_from_run(run, angles_deg=conf.ANGLES_DEG, na=conf.NA, nb=conf.NB, **config_overrides):
    summary = run["summary"]
    config = {
        "na": int(na),
        "nb": int(nb),
        "D_nom": float(summary["D_NOM"]),
        "V_nom": float(config_overrides.get("v_nom", conf.V_NOM)),
        "delta_nom": float(config_overrides.get("delta_nom", conf.DELTA_NOM)),
        "angles": [int(angle) for angle in angles_deg],
        "interval_duration": float(summary["INTERVAL_DURATION"]),
        "Ts": float(config_overrides.get("ts", conf.TS)),
        "lidar_delta_mode": config_overrides.get("lidar_delta_mode", conf.LIDAR_DELTA_MODE),
    }
    metadata = {
        "source": "build.linmodel",
        "note": "Real lidar columns are treated as already fixed: lidar[0] is forward.",
        "lidar_delta_mode": config["lidar_delta_mode"],
    }
    return serialize_models(
        run["models"],
        config=config,
        metadata=metadata,
        metrics=run["metrics_records"],
    )


def save_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(artifact, file, indent=4)
        file.write("\n")
    return path


def identify_linear_model(
    data_dir,
    output_path=None,
    pattern=conf.FILE_PATTERN,
    max_files=None,
    interval_durations=conf.INTERVAL_DURATION_LIST,
    d_nom_values=conf.D_NOM_LIST,
    run_id=None,
    set_current=True,
    source_kind="auto",
    **kwargs,
):
    output_path = output_path or conf.default_output_path(run_id=run_id)
    files = load_csv_files(data_dir, pattern=pattern, max_files=max_files)
    best, runs = run_grid_search(
        files=files,
        interval_durations=interval_durations,
        d_nom_values=d_nom_values,
        source_kind=source_kind,
        **kwargs,
    )
    artifact = artifact_from_run(best, **kwargs)
    output_path = save_artifact(artifact, output_path)
    if set_current:
        conf.update_current_artifact(
            conf.CURRENT_PARAMS_KEY,
            output_path,
            kind="linmodel_params",
            source_kind=source_kind,
        )
    return {
        "output_path": output_path,
        "best": best,
        "runs": runs,
        "artifact": artifact,
        "files": files,
    }
