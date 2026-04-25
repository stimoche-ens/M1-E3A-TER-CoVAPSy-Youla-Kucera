#!/usr/bin/env python3

import argparse
import subprocess
import sys

from conf import linmodel as lin_conf
from conf import nntrain as nn_conf
from conf import robustctl as robust_conf
from lib.artifacts import make_run_id
from lib.signal_schema import validate_csv_schema
from conf.paths import PROJECT_ROOT


def run_step(command):
    print("+", " ".join(str(part) for part in command))
    subprocess.run([str(part) for part in command], cwd=PROJECT_ROOT, check=True)


def first_matching(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no CSV files matched {directory / pattern}")
    return matches[0]


def validate_real_sim_schemas(pattern):
    real_files = sorted(lin_conf.DATASETS["real"].glob(pattern))
    sim_files = sorted(lin_conf.DATASETS["sim"].glob(pattern))
    if not real_files:
        raise FileNotFoundError(f"no real CSV files matched {lin_conf.DATASETS['real'] / pattern}")
    if not sim_files:
        raise FileNotFoundError(f"no sim CSV files matched {lin_conf.DATASETS['sim'] / pattern}")

    expected = None
    for dataset_name, source_kind, files in [
        ("real dataset", "real", real_files),
        ("sim dataset", "sim", sim_files),
    ]:
        for path in files:
            schema = validate_csv_schema(
                path,
                lin_conf.ANGLES_DEG,
                source_kind=source_kind,
                dataset_name=f"{dataset_name} {path}",
            )
            if expected is None:
                expected = schema
            elif schema != expected:
                raise ValueError(
                    "real and sim datasets do not expose the same configured canonical schema: "
                    f"expected={expected}, got={schema} for {path}"
                )

    real_schema = validate_csv_schema(real_files[0], lin_conf.ANGLES_DEG, source_kind="real", dataset_name="real dataset")
    sim_schema = validate_csv_schema(sim_files[0], lin_conf.ANGLES_DEG, source_kind="sim", dataset_name="sim dataset")
    if real_schema != sim_schema:
        raise ValueError(
            "real and sim datasets do not expose the same configured canonical schema: "
            f"real={real_schema}, sim={sim_schema}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-shot TER build: linmodel params, robust controller, UQYQ data, and NN weights."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--linmodel-dataset", choices=sorted(lin_conf.DATASETS), default="real")
    parser.add_argument("--linmodel-source-kind", choices=["auto", "real", "sim"], default="real")
    parser.add_argument("--uqyq-kind", default="real")
    parser.add_argument("--nn-dataset", default="UQYQ")
    parser.add_argument("--nn-model", default="MyLSTM")
    parser.add_argument("--nn-epochs", type=int, default=100)
    parser.add_argument("--no-ipex", action="store_true")
    parser.add_argument("--no-set-current", action="store_true")
    parser.add_argument("--no-deploy", action="store_true")
    parser.add_argument("--no-schema-check", action="store_true")
    parser.add_argument("--linmodel-max-files", type=int, default=None)
    parser.add_argument("--robust-frequency-samples", type=int, default=robust_conf.DEFAULT_FREQUENCY_SAMPLES)
    parser.add_argument("--robust-k0-scale", type=float, default=robust_conf.DEFAULT_K0_SCALE)
    parser.add_argument("--robust-scale-candidates", type=int, default=robust_conf.DEFAULT_SCALE_CANDIDATES)
    parser.add_argument("--uqyq-max-files", type=int, default=None)
    parser.add_argument("--uqyq-input-pattern", default=robust_conf.INPUT_FILE_PATTERN)
    parser.add_argument("--skip-linmodel", action="store_true")
    parser.add_argument("--skip-robustctl", action="store_true")
    parser.add_argument("--skip-uqyq", action="store_true")
    parser.add_argument("--skip-nn-train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or make_run_id()
    linmodel_params = lin_conf.default_output_path(run_id=run_id)
    robust_artifact = robust_conf.default_controller_artifact_path(run_id=run_id)
    uqyq_output_dir = robust_conf.default_uqyq_output_dir(kind=args.uqyq_kind, run_id=run_id)
    nn_weights = nn_conf.default_weights_path(args.nn_model, run_id=run_id)

    if not args.no_schema_check:
        validate_real_sim_schemas(lin_conf.FILE_PATTERN)

    if not args.skip_linmodel:
        command = [
            sys.executable,
            "build/linmodel/identify.py",
            "--dataset",
            args.linmodel_dataset,
            "--source-kind",
            args.linmodel_source_kind,
            "--output",
            linmodel_params,
            "--run-id",
            run_id,
        ]
        if args.linmodel_max_files is not None:
            command.extend(["--max-files", args.linmodel_max_files])
        if args.no_set_current:
            command.append("--no-set-current")
        run_step(command)
    else:
        linmodel_params = lin_conf.CURRENT_PARAMS_PATH

    if not args.skip_robustctl:
        command = [
            sys.executable,
            "build/robustctl/build_controller.py",
            "--params",
            linmodel_params,
            "--artifact",
            robust_artifact,
            "--run-id",
            run_id,
            "--frequency-samples",
            args.robust_frequency_samples,
        ]
        if args.robust_k0_scale is not None:
            command.extend(["--k0-scale", args.robust_k0_scale])
        if args.robust_scale_candidates is not None:
            command.extend(["--scale-candidates", args.robust_scale_candidates])
        if not args.no_deploy:
            command.append("--deploy")
        if not args.skip_uqyq:
            command.extend([
                "--identify",
                "--output-dir",
                uqyq_output_dir,
                "--uqyq-kind",
                args.uqyq_kind,
                "--input-pattern",
                args.uqyq_input_pattern,
            ])
            if args.uqyq_max_files is not None:
                command.extend(["--uqyq-max-files", args.uqyq_max_files])
        if args.no_set_current:
            command.append("--no-set-current")
        run_step(command)

    if not args.skip_nn_train:
        command = [
            sys.executable,
            "build/nntrain/ai_train.py",
            "-d",
            args.nn_dataset,
            "-m",
            args.nn_model,
            "--output",
            nn_weights,
            "--run-id",
            run_id,
            "--epochs",
            args.nn_epochs,
        ]
        if args.no_ipex:
            command.append("--no-ipex")
        if args.no_set_current:
            command.append("--no-set-current")
        run_step(command)

    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
