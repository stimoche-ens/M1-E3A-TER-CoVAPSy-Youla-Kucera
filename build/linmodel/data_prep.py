#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import conf
    from .columns import canonical_lidar_column, missing_required_columns, normalize_columns, validate_unique_angles
except ImportError:
    import conf
    from columns import canonical_lidar_column, missing_required_columns, normalize_columns, validate_unique_angles


@dataclass(frozen=True)
class SplitFiles:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def load_csv_files(folder, pattern=conf.FILE_PATTERN, max_files=None):
    files = sorted(Path(folder).glob(pattern))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"no CSV files matched {Path(folder) / pattern}")
    return files


def split_files(files, train_ratio=conf.TRAIN_RATIO, val_ratio=conf.VAL_RATIO, test_ratio=conf.TEST_RATIO):
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


def lidar_variation(distance_m, d_nom, mode=conf.LIDAR_DELTA_MODE):
    if mode == "nominal_minus_lidar":
        return d_nom - distance_m
    if mode == "lidar_minus_nominal":
        return distance_m - d_nom
    raise ValueError(f"unsupported lidar delta mode: {mode}")


def preprocess_dataframe(
    df,
    angles_deg=conf.ANGLES_DEG,
    speed_col=conf.SPEED_COL,
    steer_col=conf.STEER_COL,
    v_nom=conf.V_NOM,
    delta_nom=conf.DELTA_NOM,
    d_nom=conf.D_NOM,
    use_lidar_clip=conf.USE_LIDAR_CLIP,
    lidar_min=conf.LIDAR_MIN,
    lidar_max=conf.LIDAR_MAX,
    source_kind="auto",
    lidar_delta_mode=conf.LIDAR_DELTA_MODE,
):
    angles_deg = validate_unique_angles(angles_deg)
    df = normalize_columns(df, source_kind=source_kind)
    missing = missing_required_columns(df, angles_deg)
    if missing:
        raise ValueError(f"missing required columns after normalization: {missing}")

    numeric_columns = [conf.TIME_COL, speed_col, steer_col] + [
        canonical_lidar_column(angle) for angle in angles_deg
    ]
    df = _numeric_frame(df, numeric_columns)

    out = pd.DataFrame(index=df.index)
    out[conf.TIME_COL] = _normalize_time_origin(df[conf.TIME_COL].to_numpy(dtype=float))
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
    angles_deg=conf.ANGLES_DEG,
    na=conf.NA,
    nb=conf.NB,
    speed_col=conf.SPEED_COL,
    steer_col=conf.STEER_COL,
    v_nom=conf.V_NOM,
    delta_nom=conf.DELTA_NOM,
    d_nom=conf.D_NOM,
    interval_len=None,
    ts=conf.TS,
    use_lidar_clip=conf.USE_LIDAR_CLIP,
    lidar_min=conf.LIDAR_MIN,
    lidar_max=conf.LIDAR_MAX,
    include_ramps=True,
    source_kind="auto",
    lidar_delta_mode=conf.LIDAR_DELTA_MODE,
):
    angles_deg = validate_unique_angles(angles_deg)
    interval_len = int(interval_len or round(conf.INTERVAL_DURATION_LIST[0] / ts))
    min_rows = max(int(na), int(nb)) + 1
    by_angle = {
        int(angle): {"files": []}
        for angle in angles_deg
    }
    max_width = int(na) + 2 * int(nb)

    for path in file_list:
        raw = pd.read_csv(path)
        frame = preprocess_dataframe(
            raw,
            angles_deg=angles_deg,
            speed_col=speed_col,
            steer_col=steer_col,
            v_nom=v_nom,
            delta_nom=delta_nom,
            d_nom=d_nom,
            use_lidar_clip=use_lidar_clip,
            lidar_min=lidar_min,
            lidar_max=lidar_max,
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
