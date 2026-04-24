#!/usr/bin/env python3

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

CWD = Path(__file__).resolve().parent
LINPARAMS_FILENAME = "data_2026_04_23__15_16_51.json"


@dataclass(frozen=True)
class StateSpace:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Ts: float

    def __post_init__(self):
        object.__setattr__(self, "A", np.asarray(self.A, dtype=float))
        object.__setattr__(self, "B", np.asarray(self.B, dtype=float))
        object.__setattr__(self, "C", np.asarray(self.C, dtype=float))
        object.__setattr__(self, "D", np.asarray(self.D, dtype=float))
        self.validate()

    @property
    def n_states(self):
        return self.A.shape[0]

    @property
    def n_inputs(self):
        return self.B.shape[1]

    @property
    def n_outputs(self):
        return self.C.shape[0]

    def validate(self):
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")
        if self.B.ndim != 2 or self.B.shape[0] != self.A.shape[0]:
            raise ValueError("B has incompatible dimensions")
        if self.C.ndim != 2 or self.C.shape[1] != self.A.shape[0]:
            raise ValueError("C has incompatible dimensions")
        if self.D.ndim != 2 or self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D has incompatible dimensions")

    def simulate(self, inputs):
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2:
            raise ValueError("inputs must be two-dimensional")
        if inputs.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs")

        x = np.zeros(self.n_states)
        outputs = np.zeros((inputs.shape[0], self.n_outputs))

        for k, u in enumerate(inputs):
            outputs[k] = self.C @ x + self.D @ u
            x = self.A @ x + self.B @ u

        return outputs


@dataclass(frozen=True)
class ModelBank:
    angles: np.ndarray
    d_nom: float
    v_nom: float
    delta_nom: float
    Ts: float
    T2_a_coeffs: np.ndarray
    T2_b: np.ndarray
    T3_den: np.ndarray
    T3_num: np.ndarray

    @property
    def n_outputs(self):
        return len(self.angles)

    @property
    def n_inputs(self):
        return self.T2_b.shape[1]

    def transfer_matrix(self, omega):
        q = np.exp(-1j * omega)
        den_powers = q ** np.arange(self.T3_den.shape[2])
        num_powers = q ** np.arange(self.T3_num.shape[2])
        numerator = np.sum(self.T3_num * num_powers[None, None, :], axis=2)
        denominator = np.sum(self.T3_den * den_powers[None, None, :], axis=2)
        return numerator / denominator


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return Path(paths[0])


def default_parameter_path():
    return first_existing([
        CWD / "../Lin_Model" / LINPARAMS_FILENAME,
        CWD / LINPARAMS_FILENAME,
    ])


def default_input_dir():
    return first_existing([
        CWD / "../Lin_Model" / "Lidar_data",
        CWD,
    ])


def angle_key(angle):
    value = float(angle)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def angle_label(angle):
    value = float(angle)
    text = f"{abs(value):g}".replace(".", "_")
    if value < 0:
        return f"m{text}"
    if value > 0:
        return f"p{text}"
    return "0"


def lidar_column(angle):
    return f"lidar[{int(angle)}]"


def load_model_bank(path, d_nom):
    with open(path, "r") as file:
        data = json.load(file)

    config = data["config"]
    models = data["models"]
    angles = np.asarray(config["angles"], dtype=float)
    n_angles = len(angles)
    n_inputs = 2

    T2_a_coeffs = np.asarray(
        [models[angle_key(angle)]["a_coeffs"] for angle in angles],
        dtype=float,
    )
    T2_b_v = np.asarray(
        [models[angle_key(angle)]["b_v"] for angle in angles],
        dtype=float,
    )
    T2_b_delta = np.asarray(
        [models[angle_key(angle)]["b_delta"] for angle in angles],
        dtype=float,
    )

    T2_b = np.stack([T2_b_v, T2_b_delta], axis=1)
    T3_den_base = np.concatenate([np.ones((n_angles, 1)), -T2_a_coeffs], axis=1)
    T3_den = np.repeat(T3_den_base[:, None, :], n_inputs, axis=1)
    T3_num = np.zeros((n_angles, n_inputs, T2_b.shape[2] + 1))
    T3_num[:, :, 1:] = T2_b

    return ModelBank(
        angles=angles,
        d_nom=float(d_nom),
        v_nom=float(config["V_nom"]),
        delta_nom=float(config["delta_nom"]),
        Ts=float(config["Ts"]),
        T2_a_coeffs=T2_a_coeffs,
        T2_b=T2_b,
        T3_den=T3_den,
        T3_num=T3_num,
    )


def block_diag(matrices):
    row_count = sum(matrix.shape[0] for matrix in matrices)
    column_count = sum(matrix.shape[1] for matrix in matrices)
    result = np.zeros((row_count, column_count))
    row_start = 0
    column_start = 0

    for matrix in matrices:
        row_end = row_start + matrix.shape[0]
        column_end = column_start + matrix.shape[1]
        result[row_start:row_end, column_start:column_end] = matrix
        row_start = row_end
        column_start = column_end

    return result


def build_row_system(a_coeffs, b_coeffs, Ts):
    a_coeffs = np.asarray(a_coeffs, dtype=float)
    b_coeffs = np.asarray(b_coeffs, dtype=float)

    na = a_coeffs.shape[0]
    n_inputs = b_coeffs.shape[0]
    nb = b_coeffs.shape[1]
    n_states = na + n_inputs * nb

    A = np.zeros((n_states, n_states))
    B = np.zeros((n_states, n_inputs))
    C = np.zeros((1, n_states))
    D = np.zeros((1, n_inputs))

    C[0, :na] = a_coeffs
    C[0, na:] = b_coeffs.reshape(-1)
    A[0] = C[0]

    if na > 1:
        A[1:na, :na - 1] = np.eye(na - 1)

    for input_index in range(n_inputs):
        start = na + input_index * nb
        B[start, input_index] = 1.0
        if nb > 1:
            A[start + 1:start + nb, start:start + nb - 1] = np.eye(nb - 1)

    return StateSpace(A, B, C, D, Ts)


def build_lidar_system(bank):
    rows = [
        build_row_system(bank.T2_a_coeffs[index], bank.T2_b[index], bank.Ts)
        for index in range(bank.n_outputs)
    ]

    A = block_diag([row.A for row in rows])
    B = np.vstack([row.B for row in rows])
    C = np.zeros((bank.n_outputs, A.shape[0]))
    D = np.zeros((bank.n_outputs, bank.n_inputs))

    start = 0
    for index, row in enumerate(rows):
        end = start + row.n_states
        C[index:index + 1, start:end] = row.C
        D[index:index + 1] = row.D
        start = end

    return StateSpace(A, B, C, D, bank.Ts)


def closed_loop_right_division(H, K):
    K = np.asarray(K, dtype=float)

    if K.shape != (H.n_inputs, H.n_outputs):
        raise ValueError("K has incompatible dimensions")

    M = np.eye(H.n_inputs) + K @ H.D
    Minv = np.linalg.inv(M)

    A = H.A - H.B @ Minv @ K @ H.C
    B = H.B @ Minv
    C = H.C - H.D @ Minv @ K @ H.C
    D = H.D @ Minv

    return StateSpace(A, B, C, D, H.Ts)


def is_stable(system, limit=1.0):
    if system.n_states == 0:
        return True
    poles = np.linalg.eigvals(system.A)
    return np.max(np.abs(poles)) < limit


def closed_loop_peak(bank, K, samples):
    K = np.asarray(K, dtype=float)
    value = 0.0

    for omega in np.linspace(0.0, np.pi, samples):
        H = bank.transfer_matrix(omega)
        M = np.eye(bank.n_inputs) + K @ H
        try:
            response = H @ np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.inf
        value = max(value, np.linalg.norm(response, 2))

    return float(value)


def candidate_scales():
    coarse = np.geomspace(1e-3, 50.0, 120)
    dense = np.linspace(0.05, 8.0, 160)
    return np.unique(np.concatenate([coarse, dense]))


def synthesize_static_k0(bank, H, samples, forced_scale):
    dc_gain = bank.transfer_matrix(0.0).real
    seed = np.linalg.pinv(dc_gain)

    if forced_scale is not None:
        return float(forced_scale) * seed

    best_K = None
    best_score = np.inf

    for scale in candidate_scales():
        K = scale * seed
        closed_loop = closed_loop_right_division(H, K)

        if not is_stable(closed_loop):
            continue

        score = closed_loop_peak(bank, K, samples)
        score += 1e-3 * np.linalg.norm(K, 2)

        if np.isfinite(score) and score < best_score:
            best_score = score
            best_K = K

    if best_K is None:
        return np.zeros_like(seed)

    return best_K


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


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
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return {column: np.zeros(0) for column in columns}

    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing {missing}")

    table = {}
    for column in columns:
        values = [parse_float(row[column]) for row in rows]
        table[column] = fill_previous(values)

    return table


def required_input_columns(bank):
    return [lidar_column(angle) for angle in bank.angles] + [
        "cmd_speed",
        "cmd_angle",
    ]


def extract_eps(table, bank):
    return np.column_stack([
        bank.d_nom - table[lidar_column(angle)]
        for angle in bank.angles
    ])


def extract_reference(table, bank):
    return np.column_stack([
        table["cmd_speed"] - bank.v_nom,
        table["cmd_angle"] - bank.delta_nom,
    ])


def output_columns(bank):
    return [f"u_q_{angle_label(angle)}" for angle in bank.angles] + [
        "y_q_v",
        "y_q_delta",
    ]


def write_output(path, columns, values):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(values)


def identify_file(path, output_dir, prefix, bank, K0, CLS):
    table = read_numeric_columns(path, required_input_columns(bank))

    eps = extract_eps(table, bank)
    reference = extract_reference(table, bank)
    u_k0 = eps @ K0.T
    y_k = reference - u_k0
    y_b = CLS.simulate(y_k)
    u_k = y_b + eps

    values = np.column_stack([u_k, y_k])
    output_path = output_dir / f"{prefix}{path.name}"
    write_output(output_path, output_columns(bank), values)

    return output_path


def input_files(input_dir, prefix):
    return sorted(
        path for path in Path(input_dir).glob("shift*.csv")
        if not path.name.startswith(prefix)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=default_parameter_path())
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="uq_yq_")
    parser.add_argument("--d-nom", type=float, default=4.0)
    parser.add_argument("--frequency-samples", type=int, default=512)
    parser.add_argument("--k0-scale", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bank = load_model_bank(args.params, args.d_nom)
    H = build_lidar_system(bank)
    K0 = synthesize_static_k0(bank, H, args.frequency_samples, args.k0_scale)
    CLS = closed_loop_right_division(H, K0)

    paths = input_files(args.input_dir, args.prefix)
    if not paths:
        raise FileNotFoundError(f"no input csv files found in {args.input_dir}")

    for path in paths:
        print(identify_file(path, output_dir, args.prefix, bank, K0, CLS))


if __name__ == "__main__":
    main()
