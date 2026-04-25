#!/usr/bin/env python3

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = PROJECT_ROOT / "conf"
BUILD_DIR = PROJECT_ROOT / "build"
DATA_DIR = PROJECT_ROOT / "data"
INFERENCE_DIR = PROJECT_ROOT / "inference"

LINMODEL_DIR = BUILD_DIR / "linmodel"
ROBUSTCTL_DIR = BUILD_DIR / "robustctl"
NNTRAIN_DIR = BUILD_DIR / "nntrain"

WEBOTS_SIM_DIR = INFERENCE_DIR / "webots_sim"
WEBOTS_CONTROLLER_JAUNE_DIR = WEBOTS_SIM_DIR / "controllers" / "controller_jaune"
