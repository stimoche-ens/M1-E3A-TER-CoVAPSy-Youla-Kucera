#!/usr/bin/env python3

import torch

def norm_data_mean_stddev_len(dataset):
    raw_data_keys = dataset.raw_data.keys()
    for key in raw_data_keys:
        std, mean = torch.std_mean(dataset.raw_data[key], dim=(0,1))
        width = dataset.stats[key]["size"][-1]
        scale = std*width
        dataset.raw_data[key] -= mean
        dataset.raw_data[key] /= scale
        dataset.stats[key]["train_offset"] = mean
        dataset.stats[key]["train_scale"]  = scale
