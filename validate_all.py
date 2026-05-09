#!/usr/bin/env python3

import argparse
from pathlib import Path

from build.validate import linmodel_report, nn_report, robustctl_report
from build.validate.metrics import ensure_dir, write_json
from conf import validation as validation_conf
from lib.artifacts import make_run_id


def _pretty_name(path):
    return Path(path).stem.replace("_", " ")


def _markdown_image_path(output_dir, figure):
    figure_path = Path(figure)
    try:
        text = figure_path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        text = figure_path.as_posix()
    if any(char.isspace() for char in text):
        return f"<{text}>"
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate validation metrics and figures for linmodel, robustctl, and nntrain."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--linmodel-artifact", type=Path, default=None)
    parser.add_argument("--robust-artifact", type=Path, default=None)
    parser.add_argument("--nn-weights", type=Path, default=None)
    parser.add_argument("--uqyq-pattern", default=None)
    parser.add_argument("--linmodel-max-files", type=int, default=validation_conf.LINMODEL_MAX_FILES)
    parser.add_argument("--nn-max-trajectories", type=int, default=validation_conf.NN_MAX_TRAJECTORIES)
    parser.add_argument("--robust-frequency-samples", type=int, default=validation_conf.ROBUST_FREQUENCY_SAMPLES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-linmodel", action="store_true")
    parser.add_argument("--skip-robustctl", action="store_true")
    parser.add_argument("--skip-nn", action="store_true")
    return parser.parse_args()


def _write_index(output_dir, report):
    output_dir = Path(output_dir)
    lines = [
        "# Validation Report",
        "",
        f"Output directory: `{output_dir}`",
        "",
    ]
    for name, summary in report.items():
        if name == "run_id":
            continue
        lines.append(f"## {name}")
        if isinstance(summary, dict):
            for key in ["status", "mean_rmse", "mean_r2", "stable", "closed_peak", "first_step_rmse_mean", "all_horizon_rmse_mean"]:
                if key in summary:
                    lines.append(f"- `{key}`: `{summary[key]}`")
            figures = summary.get("figures", [])
            if figures:
                lines.extend(["", "### Figures", ""])
            for figure in figures:
                title = _pretty_name(figure)
                image_path = _markdown_image_path(output_dir, figure)
                lines.extend([
                    f"#### {title}",
                    "",
                    f"![{title}]({image_path})",
                    "",
                ])
        lines.append("")
    path = output_dir / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    args = parse_args()
    run_id = args.run_id or make_run_id()
    output_dir = ensure_dir(args.output_dir or validation_conf.default_output_dir(run_id))
    plots = not args.no_plots
    report = {"run_id": run_id}

    if not args.skip_linmodel:
        report["linmodel"] = linmodel_report.run(
            output_dir,
            artifact_path=args.linmodel_artifact,
            max_files=args.linmodel_max_files,
            plots=plots,
        )
    if not args.skip_robustctl:
        report["robustctl"] = robustctl_report.run(
            output_dir,
            artifact_path=args.robust_artifact,
            frequency_samples=args.robust_frequency_samples,
            plots=plots,
        )
    if not args.skip_nn:
        report["nntrain"] = nn_report.run(
            output_dir,
            weights_path=args.nn_weights,
            uqyq_pattern=args.uqyq_pattern,
            max_trajectories=args.nn_max_trajectories,
            plots=plots,
        )

    report_path = write_json(output_dir / "validation_report.json", report)
    index_path = _write_index(output_dir, report)
    print(report_path)
    print(index_path)


if __name__ == "__main__":
    main()
