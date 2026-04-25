# Linear Model Identification

This directory now contains a Python implementation of the ARX identification
workflow that was originally embedded in `ARX_obstacle_ramps_e.ipynb`.

The runtime convention is intentionally explicit:

- Real CSV files with `lidar[-180] ... lidar[180]` are treated as already fixed.
  `lidar[0]` is the forward ray; no notebook-era roll/rotation is applied.
- Simulation CSV files with `lidar_0 ... lidar_359` are mapped to signed angles,
  with `lidar_0` as forward and `lidar_330` as `-30` degrees.
- Identified outputs use `eps = D_nom - lidar`, matching `robustctl` runtime
  feedback. Positive eps therefore means the obstacle is closer than nominal.

Typical usage:

```bash
./identify.py --dataset real
./identify.py --dataset sim --d-nom 4.0
```

By default, `identify.py` writes a timestamped JSON file in `generated/` and
updates `conf/current_artifacts.json` so downstream tools use that artifact.

The old notebook is kept for reference, but new experiments should go through:

- `columns.py` for CSV schema normalization.
- `data_prep.py` for nominal-variation preprocessing and ARX regressors.
- `regression.py` for ridge least squares, metrics, and JSON serialization.
- `experiment.py` for grid searches over `D_nom` and interval duration.
- `identify.py` for the CLI.
