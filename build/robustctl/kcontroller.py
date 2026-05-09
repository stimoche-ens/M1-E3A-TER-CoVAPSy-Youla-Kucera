#!/usr/bin/env python3

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from . import conf
    from .linear_model import StateSpace, build_lidar_system, load_model_bank
except ImportError:  # Allows direct execution from build/robustctl.
    import conf
    from linear_model import StateSpace, build_lidar_system, load_model_bank


@dataclass(frozen=True)
class ControllerLimits:
    speed_min_m_s: float
    speed_max_m_s: float
    steering_min_deg: float
    steering_max_deg: float

    @classmethod
    def from_dict(cls, data):
        return cls(
            speed_min_m_s=float(data["speed_min_m_s"]),
            speed_max_m_s=float(data["speed_max_m_s"]),
            steering_min_deg=float(data["steering_min_deg"]),
            steering_max_deg=float(data["steering_max_deg"]),
        )

    def to_dict(self):
        return {
            "speed_min_m_s": float(self.speed_min_m_s),
            "speed_max_m_s": float(self.speed_max_m_s),
            "steering_min_deg": float(self.steering_min_deg),
            "steering_max_deg": float(self.steering_max_deg),
        }

    def clip(self, command):
        speed, angle = np.asarray(command, dtype=float)
        return np.array([
            np.clip(speed, self.speed_min_m_s, self.speed_max_m_s),
            np.clip(angle, self.steering_min_deg, self.steering_max_deg),
        ])


@dataclass(frozen=True)
class ControlCommand:
    speed_m_s: float
    steering_angle_deg: float
    eps: np.ndarray
    correction: np.ndarray


@dataclass(frozen=True)
class StaticOutputFeedbackController:
    name: str
    K: np.ndarray
    angles: np.ndarray
    d_nom: float
    v_nom: float
    delta_nom: float
    limits: ControllerLimits
    lidar_input_unit: str = "m"
    lidar_indexing: str = "signed_python"
    lidar_invalid_distance_m: float = 12.0

    def __post_init__(self):
        object.__setattr__(self, "K", np.asarray(self.K, dtype=float))
        object.__setattr__(self, "angles", np.asarray(self.angles, dtype=float))
        if self.K.shape != (2, len(self.angles)):
            raise ValueError("K must have shape (2, number_of_lidar_outputs)")

    @classmethod
    def from_artifact(cls, artifact, name="K0"):
        controller = artifact["controllers"][name]
        model = artifact["model"]
        runtime = artifact.get("runtime", {})
        return cls(
            name=name,
            K=np.asarray(controller["K"], dtype=float),
            angles=np.asarray(model["angles"], dtype=float),
            d_nom=float(model["d_nom"]),
            v_nom=float(model["v_nom"]),
            delta_nom=float(model["delta_nom"]),
            limits=ControllerLimits.from_dict(controller["limits"]),
            lidar_input_unit=runtime.get("lidar_input_unit", "m"),
            lidar_indexing=runtime.get("lidar_indexing", "signed_python"),
            lidar_invalid_distance_m=float(runtime.get("lidar_invalid_distance_m", 12.0)),
        )

    def _distance_to_meters(self, value):
        try:
            distance = float(value)
        except (TypeError, ValueError):
            return self.lidar_invalid_distance_m

        if not np.isfinite(distance) or distance <= 0.0:
            return self.lidar_invalid_distance_m
        if self.lidar_input_unit == "mm":
            distance /= 1000.0
        return min(distance, self.lidar_invalid_distance_m)

    def _lidar_at_angle(self, lidar_scan, angle):
        signed_angle = int(round(float(angle)))
        if self.lidar_indexing == "minus180_to_179":
            index = signed_angle + 180
        elif self.lidar_indexing == "signed_python":
            index = signed_angle
        else:
            raise ValueError(f"unsupported lidar indexing: {self.lidar_indexing}")
        return self._distance_to_meters(lidar_scan[index])

    def eps_from_lidar(self, lidar_scan):
        distances = np.array([
            self._lidar_at_angle(lidar_scan, angle)
            for angle in self.angles
        ])
        return self.d_nom - distances

    def correction_from_eps(self, eps):
        return self.K @ np.asarray(eps, dtype=float)

    def command_from_eps(self, eps):
        eps = np.asarray(eps, dtype=float)
        correction = self.correction_from_eps(eps)
        raw_command = np.array([self.v_nom, self.delta_nom]) + correction
        command = self.limits.clip(raw_command)
        return ControlCommand(
            speed_m_s=float(command[0]),
            steering_angle_deg=float(command[1]),
            eps=eps,
            correction=correction,
        )

    def command_from_lidar(self, lidar_scan):
        return self.command_from_eps(self.eps_from_lidar(lidar_scan))


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
    return bool(np.max(np.abs(poles)) < limit)


def closed_loop_peak(bank, K, samples):
    return closed_loop_peak_from_responses(frequency_response_grid(bank, samples), K)


def frequency_response_grid(bank, samples):
    samples = max(1, int(samples))
    return np.asarray([
        bank.transfer_matrix(omega)
        for omega in np.linspace(0.0, np.pi, samples)
    ])


def closed_loop_peak_from_responses(responses, K):
    K = np.asarray(K, dtype=float)
    responses = np.asarray(responses)
    identity = np.eye(K.shape[0], dtype=responses.dtype)
    matrices = identity[None, :, :] + np.einsum("ij,sjk->sik", K, responses)
    try:
        inverse = np.linalg.inv(matrices)
    except np.linalg.LinAlgError:
        return np.inf
    closed_loop = np.einsum("sij,sjk->sik", responses, inverse)
    return float(np.max(np.linalg.norm(closed_loop, ord=2, axis=(1, 2))))


def candidate_scales(max_candidates=None):
    coarse = np.geomspace(1e-3, 50.0, 120)
    dense = np.linspace(0.05, 8.0, 160)
    scales = np.unique(np.concatenate([coarse, dense]))
    if max_candidates is None or max_candidates >= scales.size:
        return scales
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    indices = np.unique(np.linspace(0, scales.size - 1, int(max_candidates)).round().astype(int))
    return scales[indices]


def synthesize_static_k0(
    bank,
    H,
    samples=conf.DEFAULT_FREQUENCY_SAMPLES,
    forced_scale=None,
    scale_candidates=conf.DEFAULT_SCALE_CANDIDATES,
):
    dc_gain = bank.transfer_matrix(0.0).real
    seed = np.linalg.pinv(dc_gain)

    if forced_scale is not None:
        return float(forced_scale) * seed

    best_K = None
    best_score = np.inf
    responses = frequency_response_grid(bank, samples)

    for scale in candidate_scales(scale_candidates):
        K = scale * seed
        closed_loop = closed_loop_right_division(H, K)

        if not is_stable(closed_loop):
            continue

        score = closed_loop_peak_from_responses(responses, K)
        score += 1e-3 * np.linalg.norm(K, 2)

        if np.isfinite(score) and score < best_score:
            best_score = score
            best_K = K

    if best_K is None:
        print("Warning: best_K is None")
        return np.zeros_like(seed)

    return best_K


def build_controller_artifact(
    params_path=None,
    d_nom=None,
    frequency_samples=conf.DEFAULT_FREQUENCY_SAMPLES,
    k0_scale=conf.DEFAULT_K0_SCALE,
    scale_candidates=conf.DEFAULT_SCALE_CANDIDATES,
    limits=None,
):
    params_path = params_path or conf.LINPARAMS_PATH
    bank = load_model_bank(params_path, d_nom=d_nom)
    H = build_lidar_system(bank)
    K0 = synthesize_static_k0(bank, H, frequency_samples, k0_scale, scale_candidates)
    CLS = closed_loop_right_division(H, K0)
    limits = ControllerLimits.from_dict(limits or conf.DEFAULT_LIMITS)

    return {
        "schema_version": 1,
        "source": {
            "linear_parameters": conf.project_str(params_path),
            "synthesis": "static_dc_pinv_scaled_frequency_search",
            "frequency_samples": int(frequency_samples),
            "k0_scale": None if k0_scale is None else float(k0_scale),
            "scale_candidates": None if scale_candidates is None else int(scale_candidates),
        },
        "model": bank.to_dict(),
        "plant": H.to_dict(),
        "closed_loop_right_division": CLS.to_dict(),
        "controllers": {
            "K0": {
                "kind": "static_output_feedback",
                "K": K0.tolist(),
                "input_names": [f"eps_{int(angle)}" for angle in bank.angles],
                "output_names": ["delta_speed_m_s", "delta_steering_deg"],
                "nominal_output": [float(bank.v_nom), float(bank.delta_nom)],
                "limits": limits.to_dict(),
            },
        },
        "runtime": dict(conf.RUNTIME),
    }


def save_artifact(artifact, path=None):
    path = Path(path or conf.CONTROLLER_ARTIFACT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(artifact, file, indent=2)
        file.write("\n")
    return path


def load_artifact(path):
    with Path(path).open("r") as file:
        return json.load(file)


def load_controller(path=None, name="K0"):
    return StaticOutputFeedbackController.from_artifact(
        load_artifact(path or conf.CONTROLLER_ARTIFACT_PATH),
        name=name,
    )


def synthesize_controller(name="K0", **kwargs):
    artifact = build_controller_artifact(**kwargs)
    return StaticOutputFeedbackController.from_artifact(artifact, name=name)
