#!/usr/bin/env python3

from lib.artifacts import current_path, latest_matching, timestamped_path, update_current_artifact
from lib.signal_schema import SPEED_COL, STEER_COL, TIME_COL

from .controller import CONTROLLER_ANGLES_DEG
from .paths import DATA_DIR, LINMODEL_DIR, PROJECT_ROOT

GENERATED_DIR = LINMODEL_DIR / "generated"
PARAMS_STEM = "linmodel_params"
CURRENT_PARAMS_KEY = "linmodel.params"


def default_output_path(run_id=None):
    return timestamped_path(GENERATED_DIR, PARAMS_STEM, ".json", run_id=run_id)


CURRENT_PARAMS_PATH = current_path(
    CURRENT_PARAMS_KEY,
    fallback=latest_matching(GENERATED_DIR, "*.json") or latest_matching(LINMODEL_DIR, "*.json"),
)
DEFAULT_OUTPUT_PATH = default_output_path()

DATASETS = {
    "real": DATA_DIR / "Lidar_data_real",
    "sim": DATA_DIR / "Lidar_data_sim",
}

FILE_PATTERN = "*.csv"

ANGLES_DEG = CONTROLLER_ANGLES_DEG

V_NOM = 3.0
DELTA_NOM = 0.0
D_NOM = 2.0

NA = 4
NB = 4
TS = 0.032

INTERVAL_DURATION_LIST = [6.624, 6.656, 6.688, 6.72, 6.752]
D_NOM_LIST = [2.0]
REG_ALPHA = 1e-3

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

V_OBS_REF = 3.0

USE_LIDAR_CLIP = False
LIDAR_MIN = 0.0
LIDAR_MAX = None

LIDAR_DELTA_MODE = "nominal_minus_lidar"
