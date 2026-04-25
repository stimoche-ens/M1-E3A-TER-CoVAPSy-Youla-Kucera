#!/usr/bin/env python3

from lib.artifacts import (
    current_path,
    latest_matching,
    make_run_id,
    project_str,
    timestamped_path,
    update_current_artifact,
)
from .paths import DATA_DIR, LINMODEL_DIR, PROJECT_ROOT, ROBUSTCTL_DIR, WEBOTS_CONTROLLER_JAUNE_DIR

LINPARAMS_KEY = "linmodel.params"
LINPARAMS_PATH = current_path(
    LINPARAMS_KEY,
    fallback=latest_matching(LINMODEL_DIR / "generated", "*.json") or latest_matching(LINMODEL_DIR, "*.json"),
)

GENERATED_DIR = ROBUSTCTL_DIR / "generated"
CONTROLLER_STEM = "robust_controller"
CONTROLLER_ARTIFACT_KEY = "robustctl.controller"


def default_controller_artifact_path(run_id=None):
    return timestamped_path(GENERATED_DIR, CONTROLLER_STEM, ".json", run_id=run_id)


CONTROLLER_ARTIFACT_PATH = current_path(
    CONTROLLER_ARTIFACT_KEY,
    fallback=latest_matching(GENERATED_DIR, f"{CONTROLLER_STEM}_*.json")
    or GENERATED_DIR / "robust_controller.json",
)

WEBOTS_CONTROLLER_DIR = WEBOTS_CONTROLLER_JAUNE_DIR
WEBOTS_CONTROLLER_ARTIFACT_PATH = WEBOTS_CONTROLLER_DIR / "robust_controller.json"

INPUT_DATA_DIR = DATA_DIR / "Lidar_data_real"
INPUT_FILE_PATTERN = "*.csv"
UQYQ_GENERATED_ROOT = DATA_DIR / "uqyq" / "generated"
UQYQ_PREFIX = "uq_yq_"
UQYQ_ACTIVE_GLOB_KEY = "uqyq.active_glob"


def default_uqyq_output_dir(kind="real", run_id=None):
    return UQYQ_GENERATED_ROOT / (run_id or make_run_id()) / kind


UQYQ_OUTPUT_DIR = default_uqyq_output_dir("real")

DEFAULT_FREQUENCY_SAMPLES = 512
DEFAULT_K0_SCALE = None
DEFAULT_SCALE_CANDIDATES = None

DEFAULT_LIMITS = {
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
