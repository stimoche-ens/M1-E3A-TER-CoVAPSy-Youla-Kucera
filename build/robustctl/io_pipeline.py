#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from lib.signal_schema import canonical_lidar_column, normalize_columns

try:
    from . import conf
    from .kcontroller import closed_loop_right_division, synthesize_static_k0
    from .linear_model import build_lidar_system, load_model_bank, angle_label
except ImportError:
    import conf
    from kcontroller import closed_loop_right_division, synthesize_static_k0
    from linear_model import build_lidar_system, load_model_bank, angle_label

CANONICAL_SPEED_COL = "speed_km_h"
CANONICAL_STEER_COL = "steering_angle_rad"


def fill_previous(values):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    previous = float(finite_values[0]) if finite_values.size else 0.0
    result = values.copy()

    for index, value in enumerate(result):
        if np.isfinite(value):
            previous = float(value)
        else:
            result[index] = previous

    return result


def read_numeric_columns(path, columns):
    df = normalize_columns(pd.read_csv(path))
    if df.empty:
        return {column: np.zeros(0) for column in columns}

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing {missing}")

    table = {}
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        table[column] = fill_previous(values)

    return table


def required_input_columns(bank):
    return [canonical_lidar_column(angle) for angle in bank.angles] + [
        CANONICAL_SPEED_COL,
        CANONICAL_STEER_COL,
    ]


def extract_eps(table, bank):
    return np.column_stack([
        bank.d_nom - table[canonical_lidar_column(angle)]
        for angle in bank.angles
    ])


def extract_reference(table, bank):
    return np.column_stack([
        table[CANONICAL_SPEED_COL] - bank.v_nom,
        table[CANONICAL_STEER_COL] - bank.delta_nom,
    ])


def output_columns(bank):
    return [f"u_q_{angle_label(angle)}" for angle in bank.angles] + [
        "y_q_v",
        "y_q_delta",
    ]


def write_output(path, columns, values):
    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(values)


class YoulaSignalIdentifier:
    def __init__(self, bank, K0, closed_loop):
        self.bank = bank
        self.K0 = np.asarray(K0, dtype=float)
        self.closed_loop = closed_loop

    @classmethod
    def from_parameters(
        cls,
        params_path=None,
        d_nom=None,
        frequency_samples=conf.DEFAULT_FREQUENCY_SAMPLES,
        k0_scale=conf.DEFAULT_K0_SCALE,
        scale_candidates=conf.DEFAULT_SCALE_CANDIDATES,
    ):
        bank = load_model_bank(params_path or conf.LINPARAMS_PATH, d_nom=d_nom)
        H = build_lidar_system(bank)
        K0 = synthesize_static_k0(bank, H, frequency_samples, k0_scale, scale_candidates)
        closed_loop = closed_loop_right_division(H, K0)
        return cls(bank, K0, closed_loop)

    def identify_table(self, table):
        eps = extract_eps(table, self.bank)
        reference = extract_reference(table, self.bank)
        u_k0 = eps @ self.K0.T
        y_k = reference - u_k0
        y_b = self.closed_loop.simulate(y_k)
        u_k = y_b + eps
        return np.column_stack([u_k, y_k])

    def identify_file(self, path, output_dir, prefix=conf.UQYQ_PREFIX):
        table = read_numeric_columns(path, required_input_columns(self.bank))
        values = self.identify_table(table)
        output_path = Path(output_dir) / f"{prefix}{Path(path).name}"
        write_output(output_path, output_columns(self.bank), values)
        return output_path


def input_files(input_dir, prefix=conf.UQYQ_PREFIX, pattern=conf.INPUT_FILE_PATTERN, max_files=None):
    paths = sorted(
        path for path in Path(input_dir).glob(pattern)
        if not path.name.startswith(prefix)
    )
    if max_files is not None:
        return paths[:max_files]
    return paths


def identify_directory(
    input_dir=conf.INPUT_DATA_DIR,
    output_dir=None,
    prefix=conf.UQYQ_PREFIX,
    pattern=conf.INPUT_FILE_PATTERN,
    max_files=None,
    **identifier_kwargs,
):
    output_dir = Path(output_dir or conf.UQYQ_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    identifier = YoulaSignalIdentifier.from_parameters(**identifier_kwargs)
    paths = input_files(input_dir, prefix=prefix, pattern=pattern, max_files=max_files)

    if not paths:
        raise FileNotFoundError(f"no input csv files found in {input_dir}")

    return [
        identifier.identify_file(path, output_dir, prefix)
        for path in paths
    ]
