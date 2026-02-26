#!/usr/bin/env python3
# from curses import LINES
from pathlib import Path

import numpy as np

COL_60 = 123  # -60deg colon
COL60 = 243  #  60deg colon
COL_60s = 3  # -60deg colon for the small data matrix
COL60s = 4  #  60deg colon for the small data matrix

base_filename = "mockdata_0000.csv"
noise_filename = "mockdata_0010.csv"
arrs_dict = {}

bd = np.loadtxt(base_filename, delimiter=",", skiprows=1)  # base data
nd = np.loadtxt(noise_filename, delimiter=",", skiprows=1)  # noise data
bds = np.concatenate((bd[:, :3], bd[:, COL_60 : COL_60 + 1]), axis=1)  # base data small
bds = np.concatenate((bds, bd[:, COL60 : COL60 + 1]), axis=1)  # base data small
nds = np.concatenate(
    (nd[:, :3], nd[:, COL_60 : COL_60 + 1]), axis=1
)  # noise data small
nds = np.concatenate((nds, nd[:, COL60 : COL60 + 1]), axis=1)  # noise data small

N_LINES = len(bds[:, 1])
LINES_TOL = 10  # Tolerance on the number of lines

print("N_LINES: ", N_LINES)
print(bds.shape)
print(nds.shape)

print(bds)
# arrs_dict[str(error)][str(period)]["cost"] = np.absolute(arr2-arrs_dict["0"]["1"]["arr"]).sum();
# print("cost:", arrs_dict[str(error)][str(period)]["cost"])

bds_ = bds[1:]-bds[:N_LINES-1]
nds_ = nds[1:]-nds[:N_LINES-1]
N_LINES_ = len(bds_[:, 0])
min_costs = np.absolute(bds_[:, COL_60s] - nds_[:, COL_60s])
shift = 0
for i in range(0, len(bds_[:, 1])):
    start = min(max(0, i + shift - LINES_TOL), N_LINES_ - 1)
    center = min(max(0, i + shift), N_LINES_ - 1)
    end = min(N_LINES_, max(i + shift + LINES_TOL + 1, 0))
    costs_60i = np.absolute(
        bds_[start : end, COL_60s]
        - nds_[i, COL_60s]
    )
    shift = max(min(np.argmin(costs_60i) - (center - start) + shift, N_LINES_ - 1 - i), -i)
    min_costs[i] = np.min(costs_60i)
    print("shift", shift, "-i", -i, "min", min_costs[i], "sample of bds_", bds_[i + shift, COL_60s])

min_cost = min_costs.sum()
print(min_cost)
print(min_costs[90:100])
