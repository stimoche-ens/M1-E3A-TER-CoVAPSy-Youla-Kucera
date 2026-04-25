#!/usr/bin/env python3

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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

    def to_dict(self):
        return {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "C": self.C.tolist(),
            "D": self.D.tolist(),
            "Ts": float(self.Ts),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            A=data["A"],
            B=data["B"],
            C=data["C"],
            D=data["D"],
            Ts=data["Ts"],
        )


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
    lidar_delta_mode: str = "nominal_minus_lidar"

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

    def to_dict(self):
        return {
            "angles": self.angles.tolist(),
            "d_nom": float(self.d_nom),
            "v_nom": float(self.v_nom),
            "delta_nom": float(self.delta_nom),
            "Ts": float(self.Ts),
            "T2_a_coeffs": self.T2_a_coeffs.tolist(),
            "T2_b": self.T2_b.tolist(),
            "lidar_delta_mode": self.lidar_delta_mode,
        }


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


def load_model_bank(path, d_nom=None):
    with Path(path).open("r") as file:
        data = json.load(file)

    config = data["config"]
    metadata = data.get("metadata", {})
    models = data["models"]
    angles = np.asarray(config["angles"], dtype=float)
    n_angles = len(angles)
    n_inputs = 2
    source_mode = (
        config.get("lidar_delta_mode")
        or metadata.get("lidar_delta_mode")
        or "lidar_minus_nominal"
    )

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

    if source_mode == "lidar_minus_nominal":
        # Legacy notebook artifacts modeled lidar - D_nom. Runtime feedback uses
        # eps = D_nom - lidar, so only the input/ramp-side coefficients change
        # sign; the autoregressive coefficients are invariant under y -> -y.
        T2_b_v = -T2_b_v
        T2_b_delta = -T2_b_delta
        runtime_mode = "nominal_minus_lidar"
    elif source_mode == "nominal_minus_lidar":
        runtime_mode = source_mode
    else:
        raise ValueError(f"unsupported lidar delta mode in {path}: {source_mode}")

    T2_b = np.stack([T2_b_v, T2_b_delta], axis=1)
    T3_den_base = np.concatenate([np.ones((n_angles, 1)), -T2_a_coeffs], axis=1)
    T3_den = np.repeat(T3_den_base[:, None, :], n_inputs, axis=1)
    T3_num = np.zeros((n_angles, n_inputs, T2_b.shape[2] + 1))
    T3_num[:, :, 1:] = T2_b

    return ModelBank(
        angles=angles,
        d_nom=float(config["D_nom"] if d_nom is None else d_nom),
        v_nom=float(config["V_nom"]),
        delta_nom=float(config["delta_nom"]),
        Ts=float(config["Ts"]),
        T2_a_coeffs=T2_a_coeffs,
        T2_b=T2_b,
        T3_den=T3_den,
        T3_num=T3_num,
        lidar_delta_mode=runtime_mode,
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
