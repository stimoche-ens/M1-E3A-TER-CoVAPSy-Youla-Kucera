# Reproduce `build/robustctl` as a Small Standalone Project

This tutorial builds the core of `build/robustctl` from scratch:

1. read a `linmodel_params_*.json` artifact,
2. turn each ARX lidar model into a frequency-response plant,
3. synthesize a static output-feedback controller `K0`,
4. write a runtime `robust_controller.json`,
5. test one command from a fake lidar scan.

It deliberately does not reproduce the whole repository folder. The original
`robustctl` also serializes large state-space matrices and can generate
`u_q/y_q` training CSV files. For pedagogy, this standalone version uses the
state-space model only to test closed-loop stability, and omits the big matrices
from the saved JSON.

## 1. Create the Project

```bash
mkdir robustctl_standalone
cd robustctl_standalone
mkdir -p generated
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

Copy one linear-model artifact into this folder:

```bash
cp /path/to/linmodel_params_2026_04_25__20_10_28.json linmodel_params.json
```

The input JSON must have this shape:

```txt
config.angles
config.D_nom
config.V_nom
config.delta_nom
config.Ts
models[angle].a_coeffs
models[angle].b_v
models[angle].b_delta
```

## 2. Start `robustctl_min.py`

Create `robustctl_min.py` and add this first block:

```python
import argparse
import json
from pathlib import Path

import numpy as np

LIMITS = {
    "speed_min_m_s": 0.0,
    "speed_max_m_s": 28.0 / 3.6,
    "steering_min_deg": -16.0,
    "steering_max_deg": 16.0,
}

RUNTIME = {
    "lidar_input_unit": "mm",
    "lidar_indexing": "signed_python",
    "lidar_invalid_distance_m": 12.0,
}


def angle_key(angle):
    value = float(angle)
    return str(int(value)) if value.is_integer() else f"{value:g}"
```

These defaults match the repository:

- nominal command is read from the linmodel artifact,
- speed is clipped to `0 .. 28 km/h`,
- steering is clipped to `-16 .. 16 deg`,
- runtime lidar values are expected in millimeters.

## 3. Load the Linear Model

Add this block:

```python
def load_bank(path, d_nom_override=None):
    data = json.loads(Path(path).read_text())
    cfg = data["config"]
    meta = data.get("metadata", {})
    models = data["models"]

    angles = np.asarray(cfg["angles"], dtype=float)
    a = np.asarray([models[angle_key(x)]["a_coeffs"] for x in angles], dtype=float)
    b_v = np.asarray([models[angle_key(x)]["b_v"] for x in angles], dtype=float)
    b_delta = np.asarray([models[angle_key(x)]["b_delta"] for x in angles], dtype=float)

    mode = cfg.get("lidar_delta_mode") or meta.get("lidar_delta_mode") or "lidar_minus_nominal"
    if mode == "lidar_minus_nominal":
        # Old notebook artifacts used lidar - D_nom. Runtime uses D_nom - lidar.
        b_v = -b_v
        b_delta = -b_delta
    elif mode != "nominal_minus_lidar":
        raise ValueError(f"unknown lidar_delta_mode: {mode}")

    return {
        "angles": angles,
        "d_nom": float(cfg["D_nom"] if d_nom_override is None else d_nom_override),
        "v_nom": float(cfg["V_nom"]),
        "delta_nom": float(cfg["delta_nom"]),
        "Ts": float(cfg["Ts"]),
        "a": a,
        "b": np.stack([b_v, b_delta], axis=1),
        "lidar_delta_mode": "nominal_minus_lidar",
    }
```

The ARX model for one lidar angle is:

```txt
y[k+1] = a0*y[k] + a1*y[k-1] + ...
       + b_v0*dv[k] + ...
       + b_delta0*ddelta[k] + ...
```

Here `y = eps = D_nom - lidar`.

## 4. Build the Transfer Matrix

Add this block:

```python
def plant_response(bank, omega):
    q = np.exp(-1j * omega)
    a = bank["a"]
    b = bank["b"]

    na = a.shape[1]
    nb = b.shape[2]
    den = 1.0 - np.sum(a * q ** np.arange(1, na + 1), axis=1)
    num = np.sum(b * q ** np.arange(1, nb + 1)[None, None, :], axis=2)
    return num / den[:, None]


def response_grid(bank, samples):
    return np.asarray([
        plant_response(bank, omega)
        for omega in np.linspace(0.0, np.pi, int(samples))
    ])
```

`plant_response()` returns a matrix with shape:

```txt
number_of_lidar_outputs x 2
```

For the current controller angles, that is `5 x 2`.

## 5. Synthesize Static `K0`

First add this compact state-space helper block:

```python
def block_diag(mats):
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = np.zeros((rows, cols))
    r = c = 0
    for m in mats:
        rr, cc = m.shape
        out[r:r + rr, c:c + cc] = m
        r += rr
        c += cc
    return out


def row_state_space(a, b, Ts):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, inputs, nb = len(a), b.shape[0], b.shape[1]
    n = na + inputs * nb

    A = np.zeros((n, n))
    B = np.zeros((n, inputs))
    C = np.zeros((1, n))
    D = np.zeros((1, inputs))

    C[0, :na] = a
    C[0, na:] = b.reshape(-1)
    A[0] = C[0]
    if na > 1:
        A[1:na, :na - 1] = np.eye(na - 1)

    for i in range(inputs):
        s = na + i * nb
        B[s, i] = 1.0
        if nb > 1:
            A[s + 1:s + nb, s:s + nb - 1] = np.eye(nb - 1)

    return A, B, C, D, Ts


def lidar_state_space(bank):
    rows = [row_state_space(a, b, bank["Ts"]) for a, b in zip(bank["a"], bank["b"])]
    A = block_diag([x[0] for x in rows])
    B = np.vstack([x[1] for x in rows])
    C = np.zeros((len(rows), A.shape[0]))
    D = np.zeros((len(rows), bank["b"].shape[1]))

    start = 0
    for i, row in enumerate(rows):
        _, _, C_row, D_row, _ = row
        end = start + C_row.shape[1]
        C[i:i + 1, start:end] = C_row
        D[i:i + 1] = D_row
        start = end

    return A, B, C, D


def stable_closed_loop(bank, K, limit=1.0):
    A, B, C, D = lidar_state_space(bank)
    eye = np.eye(K.shape[0])
    Acl = A - B @ np.linalg.inv(eye + K @ D) @ K @ C
    return bool(np.max(np.abs(np.linalg.eigvals(Acl))) < limit)
```

Then add the controller synthesis block:

```python
def closed_loop_peak(responses, K):
    K = np.asarray(K, dtype=float)
    eye = np.eye(K.shape[0], dtype=complex)
    loops = eye[None, :, :] + np.einsum("ij,sjk->sik", K, responses)
    try:
        inv_loops = np.linalg.inv(loops)
    except np.linalg.LinAlgError:
        return np.inf
    closed = np.einsum("sij,sjk->sik", responses, inv_loops)
    return float(np.max(np.linalg.norm(closed, ord=2, axis=(1, 2))))


def scale_candidates(max_candidates=None):
    coarse = np.geomspace(1e-3, 50.0, 120)
    dense = np.linspace(0.05, 8.0, 160)
    values = np.unique(np.concatenate([coarse, dense]))
    if max_candidates is None or max_candidates >= len(values):
        return values
    idx = np.linspace(0, len(values) - 1, int(max_candidates)).round().astype(int)
    return values[np.unique(idx)]


def synthesize_k0(bank, samples=512, forced_scale=None, max_candidates=None):
    dc_gain = plant_response(bank, 0.0).real
    seed = np.linalg.pinv(dc_gain)

    if forced_scale is not None:
        return float(forced_scale) * seed

    responses = response_grid(bank, samples)
    best_score = np.inf
    best_K = np.zeros_like(seed)

    for scale in scale_candidates(max_candidates):
        K = scale * seed
        if not stable_closed_loop(bank, K):
            continue
        score = closed_loop_peak(responses, K) + 1e-3 * np.linalg.norm(K, 2)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_K = K

    return best_K
```

The seed controller is the pseudo-inverse of the plant DC gain:

```txt
K_seed = pinv(H(0))
```

Then the code searches scalar gains and keeps the one with the smallest sampled
closed-loop peak, while rejecting unstable closed-loop candidates.

## 6. Write the Controller Artifact

Add this block:

```python
def make_artifact(bank, K, params_path, samples, scale, max_candidates):
    return {
        "schema_version": 1,
        "source": {
            "linear_parameters": str(params_path),
            "synthesis": "static_dc_pinv_scaled_frequency_search_minimal",
            "frequency_samples": int(samples),
            "k0_scale": None if scale is None else float(scale),
            "scale_candidates": None if max_candidates is None else int(max_candidates),
        },
        "model": {
            "angles": bank["angles"].tolist(),
            "d_nom": bank["d_nom"],
            "v_nom": bank["v_nom"],
            "delta_nom": bank["delta_nom"],
            "Ts": bank["Ts"],
            "T2_a_coeffs": bank["a"].tolist(),
            "T2_b": bank["b"].tolist(),
            "lidar_delta_mode": bank["lidar_delta_mode"],
        },
        "controllers": {
            "K0": {
                "kind": "static_output_feedback",
                "K": K.tolist(),
                "input_names": [f"eps_{int(a)}" for a in bank["angles"]],
                "output_names": ["delta_speed_m_s", "delta_steering_deg"],
                "nominal_output": [bank["v_nom"], bank["delta_nom"]],
                "limits": LIMITS,
            }
        },
        "runtime": RUNTIME,
    }


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path
```

This is the runtime-useful subset of the original artifact:

```txt
source
model
controllers.K0
runtime
```

The large `plant` and `closed_loop_right_division` matrices are omitted from the
JSON on purpose, even though the compact state-space model was used during
synthesis.

## 7. Add a Tiny Runtime Controller

Add this block:

```python
def meters(value, runtime):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return runtime["lidar_invalid_distance_m"]
    if not np.isfinite(x) or x <= 0.0:
        return runtime["lidar_invalid_distance_m"]
    if runtime["lidar_input_unit"] == "mm":
        x /= 1000.0
    return min(x, runtime["lidar_invalid_distance_m"])


def command_from_scan(artifact, lidar_scan):
    model = artifact["model"]
    ctrl = artifact["controllers"]["K0"]
    runtime = artifact["runtime"]

    angles = np.asarray(model["angles"], dtype=float)
    K = np.asarray(ctrl["K"], dtype=float)
    distances = np.asarray([meters(lidar_scan[int(a)], runtime) for a in angles])
    eps = float(model["d_nom"]) - distances

    raw = np.asarray(ctrl["nominal_output"], dtype=float) + K @ eps
    lim = ctrl["limits"]
    speed = np.clip(raw[0], lim["speed_min_m_s"], lim["speed_max_m_s"])
    steer = np.clip(raw[1], lim["steering_min_deg"], lim["steering_max_deg"])
    return float(speed), float(steer), eps
```

Runtime convention:

```txt
eps = D_nom - lidar_distance
command = [V_nom, delta_nom] + K0 @ eps
```

## 8. Add the CLI

Add this final block:

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("generated/robust_controller.json"))
    parser.add_argument("--d-nom", type=float, default=None)
    parser.add_argument("--frequency-samples", type=int, default=512)
    parser.add_argument("--k0-scale", type=float, default=None)
    parser.add_argument("--scale-candidates", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    bank = load_bank(args.params, d_nom_override=args.d_nom)
    K = synthesize_k0(
        bank,
        samples=args.frequency_samples,
        forced_scale=args.k0_scale,
        max_candidates=args.scale_candidates,
    )
    artifact = make_artifact(
        bank,
        K,
        params_path=args.params,
        samples=args.frequency_samples,
        scale=args.k0_scale,
        max_candidates=args.scale_candidates,
    )
    save_json(artifact, args.out)
    print(args.out)
    print("K0 shape:", K.shape)
    print("K0:")
    print(K)

    if args.smoke_test:
        scan = {int(a): 2000.0 for a in bank["angles"]}
        speed, steer, eps = command_from_scan(artifact, scan)
        print("smoke eps:", eps)
        print("smoke command:", speed, steer)


if __name__ == "__main__":
    main()
```

## 9. Build the Artifact

Run:

```bash
python robustctl_min.py \
  --params linmodel_params.json \
  --out generated/robust_controller.json \
  --smoke-test
```

You should see:

```txt
generated/robust_controller.json
K0 shape: (2, 5)
...
smoke eps: [0. 0. 0. 0. 0.]
smoke command: 3.0 0.0
```

The smoke test uses a fake scan where all selected lidar distances are exactly
`D_nom = 2 m`, so every `eps` is zero and the command stays nominal.

## 10. Use a Fixed Gain Scale

For lectures or debugging, the search can distract from the idea. Use a fixed
scale:

```bash
python robustctl_min.py \
  --params linmodel_params.json \
  --out generated/robust_controller_scale_1.json \
  --k0-scale 1.0 \
  --smoke-test
```

That makes the controller exactly:

```txt
K0 = pinv(H(0))
```

up to the scalar `--k0-scale`.

## 11. Relation to the Repository Version

This tutorial covers the important robustctl loop:

```txt
linmodel JSON -> H(z) -> K0 -> robust_controller.json -> runtime command
```

The repository implementation adds three practical layers:

- `linear_model.py` keeps the discrete state-space plant as a reusable class.
- `kcontroller.py` serializes large plant/closed-loop matrices.
- `io_pipeline.py` uses `K0` and the closed loop to generate `u_q/y_q` CSV files
  for later neural Youla-Kucera training.

Those are useful production details, but the minimal file above is the shortest
self-contained path to understand and rebuild the robust baseline controller.
