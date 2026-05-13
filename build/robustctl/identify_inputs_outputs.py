#!/usr/bin/env python3

import argparse
from pathlib import Path

try:
    from . import conf
    from .io_pipeline import (
        YoulaSignalIdentifier,
        identify_directory,
        input_files,
        output_columns,
        read_numeric_columns,
        required_input_columns,
    )
except ImportError:
    import conf
    from io_pipeline import (
        YoulaSignalIdentifier,
        identify_directory,
        input_files,
        output_columns,
        read_numeric_columns,
        required_input_columns,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate u_q/y_q files used to train a Youla-Kucera Q parameter."
    )
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=conf.INPUT_DATA_DIR)
    parser.add_argument("--input-pattern", default=conf.INPUT_FILE_PATTERN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--uqyq-kind", default="real")
    parser.add_argument("--prefix", default=conf.UQYQ_PREFIX)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-set-current", action="store_true")
    parser.add_argument("--d-nom", type=float, default=None)
    parser.add_argument("--frequency-samples", type=int, default=conf.DEFAULT_FREQUENCY_SAMPLES)
    parser.add_argument("--k0-scale", type=float, default=conf.DEFAULT_K0_SCALE)
    parser.add_argument("--scale-candidates", type=int, default=conf.DEFAULT_SCALE_CANDIDATES)
    parser.add_argument("--hinf-control-weight", type=float, default=conf.DEFAULT_HINF_CONTROL_WEIGHT)
    parser.add_argument("--hinf-gain-regularization", type=float, default=conf.DEFAULT_HINF_GAIN_REGULARIZATION)
    parser.add_argument("--hinf-max-iterations", type=int, default=conf.DEFAULT_HINF_MAX_ITERATIONS)
    parser.add_argument(
        "--hinf-max-optimized-variables",
        type=int,
        default=conf.DEFAULT_HINF_MAX_OPTIMIZED_VARIABLES,
    )
    parser.add_argument(
        "--hinf-max-stability-checks",
        type=int,
        default=conf.DEFAULT_HINF_MAX_STABILITY_CHECKS,
    )
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or conf.make_run_id()
    output_dir = args.output_dir or conf.default_uqyq_output_dir(
        kind=args.uqyq_kind,
        run_id=run_id,
    )
    print(
        "generating UQYQ "
        f"input_dir={args.input_dir} pattern={args.input_pattern} "
        f"max_files={args.max_files} output_dir={output_dir}",
        flush=True,
    )
    outputs = identify_directory(
        input_dir=args.input_dir,
        output_dir=output_dir,
        prefix=args.prefix,
        pattern=args.input_pattern,
        params_path=args.params or conf.LINPARAMS_PATH,
        d_nom=args.d_nom,
        frequency_samples=args.frequency_samples,
        k0_scale=args.k0_scale,
        scale_candidates=args.scale_candidates,
        hinf_control_weight=args.hinf_control_weight,
        hinf_gain_regularization=args.hinf_gain_regularization,
        hinf_max_iterations=args.hinf_max_iterations,
        hinf_max_optimized_variables=args.hinf_max_optimized_variables,
        hinf_max_stability_checks=args.hinf_max_stability_checks,
        max_files=args.max_files,
    )

    for output in outputs:
        print(output)

    if not args.no_set_current:
        glob_path = Path(output_dir) / f"{args.prefix}*.csv"
        conf.update_current_artifact(
            conf.UQYQ_ACTIVE_GLOB_KEY,
            glob_path,
            kind=f"uqyq_{args.uqyq_kind}",
            input_dir=conf.project_str(args.input_dir),
        )
        conf.update_current_artifact(
            f"uqyq.{args.uqyq_kind}_glob",
            glob_path,
            kind=f"uqyq_{args.uqyq_kind}",
            input_dir=conf.project_str(args.input_dir),
        )


if __name__ == "__main__":
    main()
