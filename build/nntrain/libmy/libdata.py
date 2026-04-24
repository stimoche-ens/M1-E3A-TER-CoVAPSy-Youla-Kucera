#!/usr/bin/env python3

import torch
import copy
import math
from torch.utils.data import random_split


def norm_data_mean_stddev_len(dataset):
    raw_data_keys = dataset.raw_data.keys()
    #absolute_importance = 1e3
    for key in raw_data_keys:
        std, mean = torch.std_mean(dataset.raw_data[key], dim=(0,1))
        total_std = torch.std(dataset.raw_data[key])
        std[std == 0] = 1/(3*total_std)
        width = dataset.stats[key]["size"][-1]
        scale = std*width#/absolute_importance
        dataset.raw_data[key] -= mean
        dataset.raw_data[key] /= scale
        dataset.stats[key]["train_offset"] = mean
        dataset.stats[key]["train_scale"]  = scale




def my_train_val_split(dataset, split_fraction):
    num_trajs = dataset.num_trajs
    
    # Calculate integer boundaries based on provided ratios or raw counts
    split_idx = math.ceil(num_trajs * split_fraction)
        
    # Generate randomized permutation of trajectory indices
    permuted_indices = torch.randperm(num_trajs)
    train_indices = permuted_indices[:split_idx]
    val_indices = permuted_indices[split_idx:]

    splits = []
    for indices in [train_indices, val_indices]:
        # 1. Shallow copy bypasses __init__ and preserves shared dictionaries (stats, io_cfg)
        split_obj = copy.copy(dataset)
        
        # 2. Re-bind the raw_data dictionary to trajectory-sliced tensor views
        split_obj.raw_data = {}
        for key, tensor in dataset.raw_data.items():
            # Advanced indexing via list/tensor creates a contiguous copy of the specific trajectories
            split_obj.raw_data[key] = tensor[indices]
            
        # 3. Update the internal state length mapping
        split_obj.num_trajs = len(indices)
        
        splits.append(split_obj)
        
    return splits
