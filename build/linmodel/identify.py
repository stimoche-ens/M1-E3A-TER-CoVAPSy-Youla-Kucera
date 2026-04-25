#!/usr/bin/env python3

import argparse
from pathlib import Path

try:
    from . import conf
    from .experiment import identify_linear_model
except ImportError:
    import conf
    from experiment import identify_linear_model


def _float_list(values, default):
    if not values:
        return list(default)
    result = []
    for value in values:
        result.extend(float(item.strip()) for item in str(value).split(",") if item.strip())
    return result


def _int_list(value, default):
    if value is None:
        return list(default)
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify ARX lidar models from CSV files without running the notebook."
    )
    parser.add_argument("--dataset", choices=sorted(conf.DATASETS), default="real")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--pattern", default=conf.FILE_PATTERN)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-set-current", action="store_true")
    parser.add_argument("--source-kind", choices=["auto", "real", "sim"], default="auto")
    parser.add_argument("--interval-duration", action="append", default=None)
    parser.add_argument("--d-nom", action="append", default=None)
    parser.add_argument("--angles", default=None, help="Comma-separated signed angles, e.g. -30,0,30")
    parser.add_argument("--no-ramps", action="store_true")
    parser.add_argument("--reg-alpha", type=float, default=conf.REG_ALPHA)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir or conf.DATASETS[args.dataset]
    intervals = _float_list(args.interval_duration, conf.INTERVAL_DURATION_LIST)
    d_noms = _float_list(args.d_nom, conf.D_NOM_LIST)
    angles = _int_list(args.angles, conf.ANGLES_DEG)

    result = identify_linear_model(
        data_dir=data_dir,
        output_path=args.output,
        pattern=args.pattern,
        max_files=args.max_files,
        interval_durations=intervals,
        d_nom_values=d_noms,
        angles_deg=angles,
        include_ramps=not args.no_ramps,
        reg_alpha=args.reg_alpha,
        run_id=args.run_id,
        set_current=not args.no_set_current,
        source_kind=args.source_kind,
    )
    summary = result["best"]["summary"]
    print(result["output_path"])
    print(
        "best: "
        f"D_nom={summary['D_NOM']} "
        f"interval={summary['INTERVAL_DURATION']} "
        f"val_rmse={summary['val_rmse_mean']:.6g} "
        f"test_rmse={summary['test_rmse_mean']:.6g}"
    )


if __name__ == "__main__":
    main()
