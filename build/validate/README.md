# Validation Suite

Run from the project root:

```bash
./validate_all.py
```

The suite reads the active artifacts from `conf/current_artifacts.json` and writes
metrics plus PNG figures to `build/validate/generated/<run_id>/`.

Useful quick checks:

```bash
./validate_all.py --linmodel-max-files 1 --nn-max-trajectories 1 --robust-frequency-samples 16
./validate_all.py --no-plots
```

The validation code uses `matplotlib` instead of Plotly because this report is
intended to run unattended and produce static scientific artifacts reliably.
