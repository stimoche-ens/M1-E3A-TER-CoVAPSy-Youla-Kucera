#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys
import types


def _load_root_conf():
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    path = root / "conf" / "robustctl.py"
    package_name = "_project_conf"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root / "conf")]
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(f"{package_name}.robustctl", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_source = _load_root_conf()

for _name in dir(_source):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_source, _name)
