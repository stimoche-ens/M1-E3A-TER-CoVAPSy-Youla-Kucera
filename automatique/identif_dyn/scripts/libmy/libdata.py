#!/usr/bin/env python3

import torch
from torch.utils.data import random_split

class DatasetSubsetProxy(Subset):
    def __getattr__(self, name):
        attr = getattr(self.dataset, name)
        if hasattr(attr, '__func__'):
            return types.MethodType(attr.__func__, self)

        return attr

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


def my_random_split(dataset, split_sizes):
    base_splits = random_split(dataset, split_sizes)

    proxy_splits = []
    for split in base_splits:
        proxy_split = DatasetSubsetProxy(split.dataset, split.indices)
        proxy_splits.append(proxy_split)

    return proxy_splits
