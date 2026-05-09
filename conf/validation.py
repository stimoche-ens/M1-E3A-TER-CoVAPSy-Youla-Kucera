#!/usr/bin/env python3

from lib.artifacts import make_run_id, timestamped_path

from .paths import BUILD_DIR

GENERATED_ROOT = BUILD_DIR / "validate" / "generated"
REPORT_STEM = "validation_report"


def default_output_dir(run_id=None):
    return GENERATED_ROOT / (run_id or make_run_id())


def default_report_path(run_id=None):
    return timestamped_path(default_output_dir(run_id), REPORT_STEM, ".json", run_id=run_id)


PLOT_DPI = 160
PREDICTION_MAX_POINTS = 800
LINMODEL_MAX_FILES = None

ROBUST_FREQUENCY_SAMPLES = 256

NN_MAX_TRAJECTORIES = 8
NN_MAX_WINDOWS_PER_TRAJECTORY = 400
NN_BATCH_SIZE = 256
