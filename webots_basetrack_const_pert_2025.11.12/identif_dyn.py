#!/usr/bin/env python3
import numpy as np
from pathlib import Path

#with open("data_0000_period01.csv", "r") as f:
    #print(f.readlines(1))
    #arr = np.loadtxt("data_"+f'{amplitude_pert:04d}'+"_period"+f'{pert_period:02d}'+".csv", data_matrix, delimiter=",", header=header_string);

filenames = ["data_0000_period01.csv", "data_0002_period01.csv", "data_0002_period10.csv", "data_0002_period50.csv", "data_0005_period01.csv", "data_0010_period01.csv", "data_0010_period03.csv", "data_0010_period20.csv", "data_0015_period20.csv", "data_0020_period01.csv", "data_0020_period03.csv", "data_0020_period05.csv", "data_0020_period10.csv", "data_0020_period50.csv"]
arrs_dict = {}
#arrs = np.zeros((9999,123,1))
#print(arrs.shape)

i=0
for error in [0,2,5,10,15,20]:
    arrs_dict[str(error)] = {}
    for period in [1,33,10,20]:
        filename = f"data_{error:04.0f}_period{period:02.0f}.csv";
        if (not Path(filename).exists()):
            continue;
        print(i, ": begin")
        print(filename);
        arr = np.loadtxt(filename, delimiter=",", skiprows=1);
        arr2 = np.concatenate((arr[:,:3], arr[:, 3::3]), axis=1)
        arrs_dict[str(error)][str(period)] = {};
        arrs_dict[str(error)][str(period)]["arr"] = arr2;
        print(arr2.shape)
        print(arrs_dict["0"]["1"]["arr"].shape)
        arrs_dict[str(error)][str(period)]["cost"] = np.absolute(arr2-arrs_dict["0"]["1"]["arr"]).sum();
        print("cost:", arrs_dict[str(error)][str(period)]["cost"])
        #arrs  = np.dstack((arrs, arr2[1:,:].astype(float)))
        i += 1
        print(i, ": end")

"""
for filename in filenames:
    print(i, ": begin")
    arr = np.loadtxt(filename, delimiter=",", dtype=str);
    arr2 = np.concatenate((arr[:,:3], arr[:, 3::3]), axis=1)
    arrs  = np.dstack((arrs, arr2[1:,:].astype(float)))
    i += 1
    print(i, ": end")
"""

print("arrs.shape:")
#print(arrs.shape)

print("done")
