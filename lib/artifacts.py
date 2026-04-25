#!/usr/bin/env python3

import datetime as _datetime
import json
from pathlib import Path

from .paths import CONF_DIR, PROJECT_ROOT

RUN_ID_FORMAT = "%Y_%m_%d__%H_%M_%S"
CURRENT_ARTIFACTS_PATH = CONF_DIR / "current_artifacts.json"


def make_run_id(now=None):
    now = now or _datetime.datetime.now()
    return now.strftime(RUN_ID_FORMAT)


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_str(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_current_artifacts():
    if not CURRENT_ARTIFACTS_PATH.exists():
        return {}
    with CURRENT_ARTIFACTS_PATH.open("r") as file:
        return json.load(file)


def save_current_artifacts(data):
    CURRENT_ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CURRENT_ARTIFACTS_PATH.open("w") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    return CURRENT_ARTIFACTS_PATH


def current_value(key, fallback=None):
    data = load_current_artifacts()
    value = data.get(key, fallback)
    if value is None:
        return None
    return str(value)


def current_path(key, fallback=None):
    value = current_value(key, fallback=fallback)
    if value is None:
        return None
    return project_path(value)


def update_current_artifact(key, value, **metadata):
    data = load_current_artifacts()
    data[key] = project_str(value)
    if metadata:
        meta = data.setdefault("_meta", {})
        meta[key] = {
            **metadata,
            "updated_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        }
    save_current_artifacts(data)
    return CURRENT_ARTIFACTS_PATH


def timestamped_path(directory, stem, suffix, run_id=None):
    run_id = run_id or make_run_id()
    return Path(directory) / f"{stem}_{run_id}{suffix}"


def latest_matching(directory, pattern):
    matches = sorted(Path(directory).glob(pattern))
    return matches[-1] if matches else None
