#!/usr/bin/env python3

from lib.artifacts import current_path, current_value, project_path, timestamped_path, update_current_artifact

from .controller import CONTROLLER_OUTPUT_WIDTH
from .paths import DATA_DIR, NNTRAIN_DIR, PROJECT_ROOT

DATA_PATH = str(DATA_DIR / "webots_straight_track_2026.01.15" / "*.csv")
DATA_UQYQ_REAL_PATH = str(DATA_DIR / "u_q_y_q_real" / "uq_yq*.csv")
DATA_UQYQ_SIM_PATH = str(DATA_DIR / "u_q_y_q_sim" / "uq_yq*.csv")
DATA_UQYQ_PATH = str(project_path(current_value("uqyq.active_glob", DATA_UQYQ_REAL_PATH)))
POOL = str(NNTRAIN_DIR / "pool")
OUTPUT_DIR = str(NNTRAIN_DIR / "out")
TESTS_DIR = str(NNTRAIN_DIR / "tests")

WEIGHTS_STEM_SUFFIX = "_weights"


def default_weights_path(model_name, run_id=None):
    return timestamped_path(NNTRAIN_DIR / "out", f"{model_name}{WEIGHTS_STEM_SUFFIX}", ".pth", run_id=run_id)


MYLSTM_WEIGHTS_PATH = current_path(
    "nntrain.MyLSTM.weights",
    fallback=NNTRAIN_DIR / "out" / "MyLSTM_weights.pth",
)

CMD_SPEED = "speed"
CMD_ANGLE = "angle"
MES_LIDAR = "lidar"
RES_LIDAR = "res_lidar"
CMD_SPEED_Q = "speed_q"
CMD_ANGLE_Q = "angle_q"

UQYQ_LSTM = {
    "past_window": 20,
    "future_window": 20,
    "past_feature_width": 2 + CONTROLLER_OUTPUT_WIDTH,
    "future_feature_width": 2,
    "output_width": CONTROLLER_OUTPUT_WIDTH,
    "hidden_dim": 256,
    "enabled_by_default": True,
    "blend": 0.5,
    "prediction_clip_abs_m": 12.0,
    "prediction_delta_clip_abs_m": 0.08,
}
