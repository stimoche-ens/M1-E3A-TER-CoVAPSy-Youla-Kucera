#!/usr/bin/env python3

import os
from pathlib import Path


def _path_from_env(name, default):
    return Path(os.environ.get(name, default)).expanduser().resolve()


PROJECT_ROOT = _path_from_env("TER_PROJECT_ROOT", Path(__file__).resolve().parent.parent)
CONF_DIR = _path_from_env("TER_CONF_DIR", PROJECT_ROOT / "conf")
BUILD_DIR = _path_from_env("TER_BUILD_DIR", PROJECT_ROOT / "build")
DATA_DIR = _path_from_env("TER_DATA_DIR", PROJECT_ROOT / "data")
INFERENCE_DIR = _path_from_env("TER_INFERENCE_DIR", PROJECT_ROOT / "inference")

LINMODEL_DIR = _path_from_env("TER_LINMODEL_DIR", BUILD_DIR / "linmodel")
ROBUSTCTL_DIR = _path_from_env("TER_ROBUSTCTL_DIR", BUILD_DIR / "robustctl")
NNTRAIN_DIR = _path_from_env("TER_NNTRAIN_DIR", BUILD_DIR / "nntrain")

WEBOTS_SIM_DIR = _path_from_env("TER_WEBOTS_SIM_DIR", INFERENCE_DIR / "webots_sim")
WEBOTS_CONTROLLER_JAUNE_DIR = _path_from_env(
    "TER_WEBOTS_CONTROLLER_JAUNE_DIR",
    WEBOTS_SIM_DIR / "controllers" / "controller_jaune",
)
