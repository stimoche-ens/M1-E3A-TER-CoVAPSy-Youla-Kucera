#!/usr/bin/env python3

import re
from collections import Counter
from pathlib import Path

import pandas as pd

TIME_COL = "time_s"
SPEED_COL = "speed_km_h"
STEER_COL = "steering_angle_rad"

_REAL_LIDAR_RE = re.compile(r"^lidar\[\s*([-+]?\d+)\s*\]$")
_INDEXED_LIDAR_RE = re.compile(r"^lidar_([-+]?\d+)$")
_PLAIN_ANGLE_RE = re.compile(r"^[-+]?\d+$")

DEFAULT_ALIASES = {
    "timestamp": TIME_COL,
    "# time": TIME_COL,
    "time": TIME_COL,
    "cmd_speed": SPEED_COL,
    "vitesse": SPEED_COL,
    "cmd_angle": STEER_COL,
    "angle": STEER_COL,
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


def required_columns(angles, time_col=TIME_COL, speed_col=SPEED_COL, steer_col=STEER_COL):
    angles = validate_unique_angles(angles)
    return [time_col, speed_col, steer_col] + [canonical_lidar_column(angle) for angle in angles]


def missing_required_columns(df, angles, time_col=TIME_COL, speed_col=SPEED_COL, steer_col=STEER_COL):
    required = required_columns(angles, time_col=time_col, speed_col=speed_col, steer_col=steer_col)
    return [column for column in required if column not in df.columns]


def validate_required_columns(df, angles, dataset_name="dataset", time_col=TIME_COL, speed_col=SPEED_COL, steer_col=STEER_COL):
    missing = missing_required_columns(
        df,
        angles,
        time_col=time_col,
        speed_col=speed_col,
        steer_col=steer_col,
    )
    if missing:
        raise ValueError(f"{dataset_name} is missing canonical column(s): {missing}")


def validate_csv_schema(path, angles, source_kind="auto", dataset_name=None):
    path = Path(path)
    df = normalize_columns(pd.read_csv(path, nrows=5), source_kind=source_kind)
    validate_required_columns(df, angles, dataset_name=dataset_name or str(path))
    return required_columns(angles)
