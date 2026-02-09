#!/usr/bin/env python3

import os
import argparse
import sys
import glob
import importlib.util
import inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import conf
from tqdm import tqdm

def get_file_pool(extra_files=None, verbose=False):
    """
    Creates a pool of candidate python files:
    1. All .py files in the current directory.
    2. Any specific files provided by the user via --files.
    3. Excludes the script itself to prevent recursion.
    """
    pool = set(glob.glob(str(conf.POOL)+"/*.py"))
    if extra_files:
        for f in extra_files:
            if os.path.exists(f):
                pool.add(f)
            else:
                print(f"Warning: User provided file '{f}' does not exist.")
    script_name = os.path.basename(__file__)
    if script_name in pool:
        pool.remove(script_name)
    return list(pool)


def load_class_from_pool(class_name, file_pool, verbose=False):
    """
    Iterates through all files in the pool.
    Returns: (The Class Object, The Module Object)
    Raises: ValueError if class is not found.
    """
    if verbose:
        print(f"Searching for class '{class_name}' in {len(file_pool)} files...")
    for file_path in file_pool:
        #module_name = os.path.basename(file_path).replace('.py', '')
        module_name = "pool." + os.path.splitext(os.path.basename(file_path))[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                #sys.modules[module_name] = module # Register module for pickling support
                spec.loader.exec_module(module)
                if hasattr(module, class_name):
                    found_class = getattr(module, class_name)
                    if inspect.isclass(found_class):
                        if verbose:
                            print(f"✅ Found '{class_name}' in {file_path}")
                        return found_class, module
        except Exception as e:
            print(f"Warning: Skipping {file_path} due to error: {e}")
            pass
    raise ValueError(f"Could not find class '{class_name}' in any of the provided files.")

