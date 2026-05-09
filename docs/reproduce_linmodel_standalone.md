# Reproduce `build/linmodel` as a Standalone Codebase

This tutorial starts from an empty directory and builds a small, self-contained
Python codebase that identifies the same kind of ARX lidar model produced by
`build/linmodel`.

It intentionally leaves out the notebook, historical generated artifacts,
`__pycache__`, and the old experiment/grid-search layer. The result performs one
configured identification run and writes one JSON artifact.

## 1. Create the Project

```bash
mkdir linmodel_standalone
cd linmodel_standalone
mkdir -p linmodel data/Lidar_data_real data/Lidar_data_sim generated
touch linmodel/__init__.py
```

Create `requirements.txt`:

```txt
numpy
pandas
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Add Configuration

Create `linmodel/config.py`:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = PROJECT_ROOT / "generated"
CURRENT_ARTIFACTS_PATH = PROJECT_ROOT / "current_artifacts.json"

DATASETS = {
    "real": DATA_DIR / "Lidar_data_real",
    "sim": DATA_DIR / "Lidar_data_sim",
}

FILE_PATTERN = "*.csv"
PARAMS_STEM = "linmodel_params"
CURRENT_PARAMS_KEY = "linmodel.params"

TIME_COL = "time_s"
SPEED_COL = "speed_km_h"
STEER_COL = "steering_angle_rad"

ANGLES_DEG = [-60, -30, 0, 30, 60]

V_NOM = 3.0
DELTA_NOM = 0.0
D_NOM = 2.0

NA = 4
NB = 4
TS = 0.032
INTERVAL_DURATION = 6.72
REG_ALPHA = 1e-3

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

V_OBS_REF = 3.0

USE_LIDAR_CLIP = False
LIDAR_MIN = 0.0
LIDAR_MAX = None

LIDAR_DELTA_MODE = "nominal_minus_lidar"
```

## 3. Add CSV Schema Normalization

Create `linmodel/signal_schema.py`:

```python
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config as cfg

_REAL_LIDAR_RE = re.compile(r"^lidar\[\s*([-+]?\d+)\s*\]$")
_INDEXED_LIDAR_RE = re.compile(r"^lidar_([-+]?\d+)$")
_PLAIN_ANGLE_RE = re.compile(r"^[-+]?\d+$")

DEFAULT_ALIASES = {
    "timestamp": cfg.TIME_COL,
    "# time": cfg.TIME_COL,
    "time": cfg.TIME_COL,
    "cmd_speed": cfg.SPEED_COL,
    "vitesse": cfg.SPEED_COL,
    "cmd_angle": cfg.STEER_COL,
    "angle": cfg.STEER_COL,
}


def validate_unique_angles(angles):
    normalized = [int(round(float(angle))) for angle in angles]
    duplicates = sorted(angle for angle, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise ValueError(
            "controller angles must be unique; duplicate angle(s): "
            + ", ".join(str(angle) for angle in duplicates)
        )
    return normalized


def canonical_lidar_column(angle):
    return f"lidar_{int(round(float(angle)))}"


def indexed_lidar_to_signed_angle(index):
    index = int(index)
    if index < 0:
        return index
    if index <= 180:
        return index
    return index - 360


def canonical_column_name(name, source_kind="auto", aliases=None):
    aliases = DEFAULT_ALIASES if aliases is None else aliases
    text = str(name).strip()
    if text in aliases:
        return aliases[text]

    match = _REAL_LIDAR_RE.match(text)
    if match:
        return canonical_lidar_column(int(match.group(1)))

    match = _INDEXED_LIDAR_RE.match(text)
    if match:
        raw_index = int(match.group(1))
        angle = raw_index if source_kind == "real" else indexed_lidar_to_signed_angle(raw_index)
        return canonical_lidar_column(angle)

    if _PLAIN_ANGLE_RE.match(text):
        return canonical_lidar_column(int(text))

    return text


def _merge_column(target, values):
    if target is None:
        return values
    return target.combine_first(values)


def normalize_columns(df, source_kind="auto", aliases=None):
    columns = {}
    for column in df.columns:
        canonical = canonical_column_name(column, source_kind=source_kind, aliases=aliases)
        columns[canonical] = _merge_column(columns.get(canonical), df[column])
    return pd.DataFrame(columns, index=df.index).copy()


def required_columns(angles):
    angles = validate_unique_angles(angles)
    return [cfg.TIME_COL, cfg.SPEED_COL, cfg.STEER_COL] + [
        canonical_lidar_column(angle) for angle in angles
    ]


def missing_required_columns(df, angles):
    required = required_columns(angles)
    return [column for column in required if column not in df.columns]


def validate_required_columns(df, angles, dataset_name="dataset"):
    missing = missing_required_columns(df, angles)
    if missing:
        raise ValueError(f"{dataset_name} is missing canonical column(s): {missing}")


def validate_csv_schema(path, angles, source_kind="auto", dataset_name=None):
    path = Path(path)
    df = normalize_columns(pd.read_csv(path, nrows=5), source_kind=source_kind)
    validate_required_columns(df, angles, dataset_name=dataset_name or str(path))
    return required_columns(angles)
```

## 4. Add Artifact Helpers

Create `linmodel/artifacts.py`:

```python
import datetime as _datetime
import json
from pathlib import Path

from . import config as cfg

RUN_ID_FORMAT = "%Y_%m_%d__%H_%M_%S"


def make_run_id(now=None):
    now = now or _datetime.datetime.now()
    return now.strftime(RUN_ID_FORMAT)


def project_str(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(cfg.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def default_output_path(run_id=None):
    run_id = run_id or make_run_id()
    return cfg.GENERATED_DIR / f"{cfg.PARAMS_STEM}_{run_id}.json"


def load_current_artifacts():
    if not cfg.CURRENT_ARTIFACTS_PATH.exists():
        return {}
    with cfg.CURRENT_ARTIFACTS_PATH.open("r") as file:
        return json.load(file)


def save_current_artifacts(data):
    cfg.CURRENT_ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with cfg.CURRENT_ARTIFACTS_PATH.open("w") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    return cfg.CURRENT_ARTIFACTS_PATH


def update_current_artifact(key, value, **metadata):
    data = load_current_artifacts()
    data[key] = project_str(value)
    if metadata:
        meta = data.setdefault("_meta", {})
        meta[key] = {
            **metadata,
            "updated_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        }
    save_current_artifacts(data)
    return cfg.CURRENT_ARTIFACTS_PATH
```

## 5. Add Data Preparation

Create `linmodel/data_prep.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .signal_schema import (
    canonical_lidar_column,
    missing_required_columns,
    normalize_columns,
    validate_unique_angles,
)


@dataclass(frozen=True)
class SplitFiles:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def load_csv_files(folder, pattern=cfg.FILE_PATTERN, max_files=None):
    files = sorted(Path(folder).glob(pattern))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"no CSV files matched {Path(folder) / pattern}")
    return files


def split_files(files, train_ratio=cfg.TRAIN_RATIO, val_ratio=cfg.VAL_RATIO, test_ratio=cfg.TEST_RATIO):
    files = list(files)
    n_files = len(files)
    if n_files == 1:
        return SplitFiles(train=files, val=files, test=files)

    train_end = max(1, int(round(n_files * train_ratio)))
    val_count = int(round(n_files * val_ratio))
    if train_end + val_count >= n_files:
        val_count = max(0, n_files - train_end - 1)
    val_end = train_end + val_count
    return SplitFiles(
        train=files[:train_end],
        val=files[train_end:val_end] or files[:train_end],
        test=files[val_end:] or files[train_end:val_end] or files[:train_end],
    )


def _numeric_frame(df, columns):
    duplicate_columns = sorted({column for column in columns if columns.count(column) > 1})
    if duplicate_columns:
        raise ValueError(
            "duplicate canonical column request(s): "
            + ", ".join(duplicate_columns)
            + ". Check ANGLES_DEG for repeated angles."
        )
    out = df.loc[:, columns].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.interpolate(limit_direction="both").ffill().bfill()


def _maybe_convert_lidar_to_meters(values):
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmedian(np.abs(finite)) > 50.0:
        return values / 1000.0
    return values


def _normalize_time_origin(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size:
        values = values - finite[0]
    return values


def lidar_variation(distance_m, d_nom, mode=cfg.LIDAR_DELTA_MODE):
    if mode == "nominal_minus_lidar":
        return d_nom - distance_m
    if mode == "lidar_minus_nominal":
        return distance_m - d_nom
    raise ValueError(f"unsupported lidar delta mode: {mode}")


def preprocess_dataframe(
    df,
    angles_deg=cfg.ANGLES_DEG,
    speed_col=cfg.SPEED_COL,
    steer_col=cfg.STEER_COL,
    v_nom=cfg.V_NOM,
    delta_nom=cfg.DELTA_NOM,
    d_nom=cfg.D_NOM,
    use_lidar_clip=cfg.USE_LIDAR_CLIP,
    lidar_min=cfg.LIDAR_MIN,
    lidar_max=cfg.LIDAR_MAX,
    source_kind="auto",
    lidar_delta_mode=cfg.LIDAR_DELTA_MODE,
):
    angles_deg = validate_unique_angles(angles_deg)
    df = normalize_columns(df, source_kind=source_kind)
    missing = missing_required_columns(df, angles_deg)
    if missing:
        raise ValueError(f"missing required columns after normalization: {missing}")

    numeric_columns = [cfg.TIME_COL, speed_col, steer_col] + [
        canonical_lidar_column(angle) for angle in angles_deg
    ]
    df = _numeric_frame(df, numeric_columns)

    out = pd.DataFrame(index=df.index)
    out[cfg.TIME_COL] = _normalize_time_origin(df[cfg.TIME_COL].to_numpy(dtype=float))
    out["delta_v"] = df[speed_col].to_numpy(dtype=float) - float(v_nom)
    out["delta_delta"] = df[steer_col].to_numpy(dtype=float) - float(delta_nom)

    for angle in angles_deg:
        column = canonical_lidar_column(angle)
        lidar_m = _maybe_convert_lidar_to_meters(df[column].to_numpy(dtype=float))
        if use_lidar_clip:
            lidar_m = np.clip(lidar_m, lidar_min, lidar_max)
        out[column] = lidar_m
        out[f"delta_y_{int(angle)}"] = lidar_variation(
            lidar_m,
            d_nom=float(d_nom),
            mode=lidar_delta_mode,
        )

    return out.dropna().reset_index(drop=True)


def assign_interval_ids_and_local_ramps(df, interval_len, ts):
    out = df.copy()
    n_rows = len(out)
    interval_ids = np.arange(n_rows) // int(interval_len)
    local_idx = np.arange(n_rows) % int(interval_len)
    out["interval_id"] = interval_ids.astype(int)
    out["interval_local_idx"] = local_idx.astype(float)
    out["interval_local_time"] = local_idx.astype(float) * float(ts)
    return out


def trim_to_full_intervals(df, interval_len, min_rows):
    n_full = (len(df) // int(interval_len)) * int(interval_len)
    if n_full <= min_rows:
        return df.iloc[0:0].copy()
    return df.iloc[:n_full].copy()


def build_file_dataset_for_angle(df, angle, na, nb, include_ramps=True):
    y = df[f"delta_y_{int(angle)}"].to_numpy(dtype=float)
    delta_v = df["delta_v"].to_numpy(dtype=float)
    delta_delta = df["delta_delta"].to_numpy(dtype=float)
    interval_ids = df["interval_id"].to_numpy(dtype=int)
    local_ramp = df["interval_local_time"].to_numpy(dtype=float)

    n_intervals = int(interval_ids.max()) + 1 if len(interval_ids) else 0
    max_lag = max(int(na), int(nb))
    rows = []
    targets = []
    meta = []

    for k in range(max_lag - 1, len(df) - 1):
        row = []
        row.extend(y[k - lag] for lag in range(int(na)))
        row.extend(delta_v[k - lag] for lag in range(int(nb)))
        row.extend(delta_delta[k - lag] for lag in range(int(nb)))

        if include_ramps:
            ramps = np.zeros(n_intervals, dtype=float)
            ramps[interval_ids[k]] = local_ramp[k]
            row.extend(ramps)

        rows.append(row)
        targets.append(y[k + 1])
        meta.append({"k": int(k), "interval_id": int(interval_ids[k])})

    if not rows:
        return {
            "X": np.zeros((0, int(na) + 2 * int(nb))),
            "Y": np.zeros(0),
            "meta": [],
            "n_intervals": n_intervals,
        }

    return {
        "X": np.asarray(rows, dtype=float),
        "Y": np.asarray(targets, dtype=float),
        "meta": meta,
        "n_intervals": n_intervals,
    }


def _pad_features(matrix, width):
    if matrix.shape[1] == width:
        return matrix
    if matrix.shape[1] > width:
        return matrix[:, :width]
    pad = np.zeros((matrix.shape[0], width - matrix.shape[1]), dtype=float)
    return np.hstack([matrix, pad])


def build_split_dataset(
    file_list,
    angles_deg=cfg.ANGLES_DEG,
    na=cfg.NA,
    nb=cfg.NB,
    v_nom=cfg.V_NOM,
    delta_nom=cfg.DELTA_NOM,
    d_nom=cfg.D_NOM,
    interval_len=None,
    ts=cfg.TS,
    include_ramps=True,
    source_kind="auto",
    lidar_delta_mode=cfg.LIDAR_DELTA_MODE,
):
    angles_deg = validate_unique_angles(angles_deg)
    interval_len = int(interval_len or round(cfg.INTERVAL_DURATION / ts))
    min_rows = max(int(na), int(nb)) + 1
    by_angle = {int(angle): {"files": []} for angle in angles_deg}
    max_width = int(na) + 2 * int(nb)

    for path in file_list:
        raw = pd.read_csv(path)
        frame = preprocess_dataframe(
            raw,
            angles_deg=angles_deg,
            v_nom=v_nom,
            delta_nom=delta_nom,
            d_nom=d_nom,
            source_kind=source_kind,
            lidar_delta_mode=lidar_delta_mode,
        )
        frame = trim_to_full_intervals(frame, interval_len, min_rows)
        if frame.empty:
            continue

        frame = assign_interval_ids_and_local_ramps(frame, interval_len=interval_len, ts=ts)
        for angle in angles_deg:
            item = build_file_dataset_for_angle(
                frame,
                angle=angle,
                na=na,
                nb=nb,
                include_ramps=include_ramps,
            )
            item["path"] = str(path)
            by_angle[int(angle)]["files"].append(item)
            max_width = max(max_width, item["X"].shape[1])

    for angle in angles_deg:
        bucket = by_angle[int(angle)]["files"]
        if bucket:
            by_angle[int(angle)]["X"] = np.vstack([
                _pad_features(item["X"], max_width) for item in bucket
            ])
            by_angle[int(angle)]["Y"] = np.concatenate([item["Y"] for item in bucket])
        else:
            by_angle[int(angle)]["X"] = np.zeros((0, max_width), dtype=float)
            by_angle[int(angle)]["Y"] = np.zeros(0, dtype=float)
        by_angle[int(angle)]["feature_width"] = max_width

    return by_angle
```

## 6. Add Regression and Serialization

Create `linmodel/regression.py`:

```python
import math

import numpy as np
import pandas as pd

from . import config as cfg


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


def solve_least_squares(X, Y, alpha=cfg.REG_ALPHA):
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


def fit_models_per_angle(train_data, angles_deg=cfg.ANGLES_DEG, na=cfg.NA, nb=cfg.NB, alpha=cfg.REG_ALPHA):
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


def evaluate_split(split_data, models, angles_deg=cfg.ANGLES_DEG):
    rows = []
    predictions = {}
    for angle in angles_deg:
        angle = int(angle)
        item = split_data[angle]
        model = models[angle]
        y_true = np.asarray(item["Y"], dtype=float)
        if y_true.size == 0:
            continue
        y_pred = predict_linear(item["X"], model["theta"])
        err = y_pred - y_true
        rows.append({
            "angle": angle,
            "samples": int(y_true.size),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "r2": float(_safe_r2(y_true, y_pred)),
        })
        predictions[angle] = {
            "Y_true": y_true,
            "Y_pred": y_pred,
            "error": err,
        }
    return pd.DataFrame(rows), predictions


def summarize_model_coefficients(models, v_obs_ref=cfg.V_OBS_REF):
    global_rows = []
    ramp_rows = []

    for angle, model in models.items():
        for index, value in enumerate(model["a_coeffs"]):
            global_rows.append({"angle": int(angle), "kind": "a", "index": int(index), "value": float(value)})
        for index, value in enumerate(model["b_v"]):
            global_rows.append({"angle": int(angle), "kind": "b_v", "index": int(index), "value": float(value)})
        for index, value in enumerate(model["b_delta"]):
            global_rows.append({"angle": int(angle), "kind": "b_delta", "index": int(index), "value": float(value)})
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
```

## 7. Add the CLI

Create `linmodel/identify.py`:

```python
import argparse
import json
from pathlib import Path

import numpy as np

from . import config as cfg
from .artifacts import default_output_path, update_current_artifact
from .data_prep import build_split_dataset, load_csv_files, split_files
from .regression import (
    evaluate_split,
    fit_models_per_angle,
    serialize_models,
    summarize_model_coefficients,
)


def _int_list(value, default):
    if value is None:
        return list(default)
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _interval_len(interval_duration, ts):
    ratio = float(interval_duration) / float(ts)
    rounded = int(round(ratio))
    if not np.isclose(ratio, rounded, atol=1e-9):
        raise ValueError(f"interval duration {interval_duration} is not an integer multiple of Ts={ts}")
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
    pattern=cfg.FILE_PATTERN,
    max_files=None,
    angles_deg=cfg.ANGLES_DEG,
    na=cfg.NA,
    nb=cfg.NB,
    ts=cfg.TS,
    v_nom=cfg.V_NOM,
    delta_nom=cfg.DELTA_NOM,
    d_nom=cfg.D_NOM,
    interval_duration=cfg.INTERVAL_DURATION,
    reg_alpha=cfg.REG_ALPHA,
    include_ramps=True,
    source_kind="auto",
    lidar_delta_mode=cfg.LIDAR_DELTA_MODE,
    run_id=None,
    set_current=True,
):
    output_path = output_path or default_output_path(run_id=run_id)
    interval_len = _interval_len(interval_duration, ts)
    files = load_csv_files(data_dir, pattern=pattern, max_files=max_files)
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
    train_metrics, _ = evaluate_split(train_data, models=models, angles_deg=angles_deg)
    val_metrics, _ = evaluate_split(val_data, models=models, angles_deg=angles_deg)
    test_metrics, _ = evaluate_split(test_data, models=models, angles_deg=angles_deg)
    _, ramp_coeffs = summarize_model_coefficients(models=models)

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

    artifact = serialize_models(
        models,
        config={
            "na": int(na),
            "nb": int(nb),
            "D_nom": float(d_nom),
            "V_nom": float(v_nom),
            "delta_nom": float(delta_nom),
            "angles": [int(angle) for angle in angles_deg],
            "interval_duration": float(interval_duration),
            "Ts": float(ts),
            "lidar_delta_mode": lidar_delta_mode,
        },
        metadata={
            "source": "standalone.linmodel",
            "note": "Real lidar columns are treated as already fixed: lidar[0] is forward.",
            "lidar_delta_mode": lidar_delta_mode,
            "summary": summary,
        },
        metrics=_metrics_to_records(metrics),
    )
    output_path = save_artifact(artifact, output_path)
    if set_current:
        update_current_artifact(
            cfg.CURRENT_PARAMS_KEY,
            output_path,
            kind="linmodel_params",
            source_kind=source_kind,
        )
    return {"output_path": output_path, "summary": summary, "artifact": artifact, "files": files}


def parse_args():
    parser = argparse.ArgumentParser(description="Identify ARX lidar models from CSV files.")
    parser.add_argument("--dataset", choices=sorted(cfg.DATASETS), default="real")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--pattern", default=cfg.FILE_PATTERN)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-set-current", action="store_true")
    parser.add_argument("--source-kind", choices=["auto", "real", "sim"], default="auto")
    parser.add_argument("--interval-duration", type=float, default=cfg.INTERVAL_DURATION)
    parser.add_argument("--d-nom", type=float, default=cfg.D_NOM)
    parser.add_argument("--angles", default=None, help="Comma-separated signed angles, e.g. -30,0,30")
    parser.add_argument("--no-ramps", action="store_true")
    parser.add_argument("--reg-alpha", type=float, default=cfg.REG_ALPHA)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir or cfg.DATASETS[args.dataset]
    angles = _int_list(args.angles, cfg.ANGLES_DEG)
    result = identify_linear_model(
        data_dir=data_dir,
        output_path=args.output,
        pattern=args.pattern,
        max_files=args.max_files,
        angles_deg=angles,
        d_nom=args.d_nom,
        interval_duration=args.interval_duration,
        include_ramps=not args.no_ramps,
        reg_alpha=args.reg_alpha,
        run_id=args.run_id,
        set_current=not args.no_set_current,
        source_kind=args.source_kind,
    )
    summary = result["summary"]
    print(result["output_path"])
    print(
        "fit: "
        f"D_nom={summary['D_NOM']} "
        f"interval={summary['INTERVAL_DURATION']} "
        f"val_rmse={summary['val_rmse_mean']:.6g} "
        f"test_rmse={summary['test_rmse_mean']:.6g}"
    )


if __name__ == "__main__":
    main()
```

## 8. Add Data

The code expects CSV files under one of these directories:

```txt
data/Lidar_data_real/*.csv
data/Lidar_data_sim/*.csv
```

After normalization, every CSV must expose:

```txt
time_s, speed_km_h, steering_angle_rad,
lidar_-60, lidar_-30, lidar_0, lidar_30, lidar_60
```

Accepted aliases include:

- Time: `time_s`, `timestamp`, `# time`, `time`
- Speed: `speed_km_h`, `cmd_speed`, `vitesse`
- Steering: `steering_angle_rad`, `cmd_angle`, `angle`
- Real lidar: `lidar[-60]`, `lidar[-30]`, `lidar[0]`, `lidar[30]`, `lidar[60]`
- Simulation lidar: `lidar_300`, `lidar_330`, `lidar_0`, `lidar_30`, `lidar_60`

To reproduce the current repository run, copy the real input CSVs:

```bash
cp /path/to/original/repo/data/Lidar_data_real/*.csv data/Lidar_data_real/
```

## 9. Run Identification

Run the standalone code:

```bash
python -m linmodel.identify \
  --dataset real \
  --source-kind real \
  --interval-duration 6.72 \
  --d-nom 2.0 \
  --run-id rebuild
```

This writes:

```txt
generated/linmodel_params_rebuild.json
current_artifacts.json
```

For a rehearsal that does not update `current_artifacts.json`:

```bash
python -m linmodel.identify \
  --dataset real \
  --source-kind real \
  --interval-duration 6.72 \
  --d-nom 2.0 \
  --run-id smoke \
  --no-set-current
```

## 10. Inspect the Artifact

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("generated/linmodel_params_rebuild.json")
data = json.loads(path.read_text())
print(data["schema_version"])
print(data["config"])
print(data["models"].keys())
for angle, model in data["models"].items():
    print(angle, len(model["theta"]), model["sample_count"])
PY
```

Expected structure:

```txt
schema_version: 2
config: na, nb, D_nom, V_nom, delta_nom, angles, interval_duration, Ts
models: one model per configured angle
each model: a_coeffs, b_v, b_delta, theta, ramp_coeffs, sample_count
metrics: train, val, test
```

## 11. What This Reproduces

This standalone code reproduces the production part of `build/linmodel`:

- CSV schema normalization for real and simulation lidar columns.
- Lidar output definition `eps = D_nom - lidar`.
- ARX regressor construction with `na = 4`, `nb = 4`.
- Optional per-interval ramp features.
- Ridge least-squares fitting with `alpha = 1e-3`.
- Per-angle model serialization to JSON schema version 2.
- Current-artifact tracking.

It deliberately omits:

- `ARX_obstacle_ramps_e.ipynb`
- `experiment.py`
- grid searches over many `D_nom` and interval values
- old generated JSON files
- validation plots
- `__pycache__`

The original repository's checked production run selected `D_nom=2.0` and
`interval_duration=6.72`. That is why the tutorial uses those as fixed defaults
instead of rebuilding the experiment layer.
