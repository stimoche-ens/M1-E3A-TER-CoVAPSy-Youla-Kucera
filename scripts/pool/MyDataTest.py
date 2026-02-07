#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
import glob

from torch.utils.data import Dataset, DataLoader

NLINES = 5
CTL_WIDTH = 2
LIDAR_WIDTH = 3
WINDOW_PAST=2
WINDOW_FUT=2


class MyDataTest(Dataset):
    def __init__(self):
        self.window_past = WINDOW_PAST
        self.window_fut = WINDOW_FUT
        controls = []
        lidar    = []
        for i in range(0,3):
            controls.append(torch.rand(NLINES,CTL_WIDTH))
            lidar.append(torch.rand(NLINES,LIDAR_WIDTH))
        controls = torch.stack(controls, dim=0)
        lidar    = torch.stack(lidar,    dim=0)
        self.data = torch.cat([controls,lidar],dim=2)
        self.num_trajs,self.traj_len, _ = self.data.shape 
        self.samples_per_traj = self.traj_len - (self.window_past+self.window_fut) + 1
        
    def __len__(self):
        # Total samples = Trajectories * Valid Samples per Trajectory
        return self.num_trajs * self.samples_per_traj

    def __getitem__(self, idx):
        # Determine which trajectory and which start row this index belongs to
        traj_idx = idx // self.samples_per_traj
        start_row = idx % self.samples_per_traj
        # Slicing indices
        past_end = start_row + self.window_past
        fut_end = past_end + self.window_fut
        # Get the single trajectory
        traj = self.data[traj_idx]
        # --- SLICE DATA ---
        # Past: EVERYTHING (Controls + LiDAR)
        past_data = traj[start_row:past_end, :] 
        # Future: Controls ONLY (Indices 0 and 1)
        future_cmds = traj[past_end:fut_end, 0:2]
        # Target: LiDAR ONLY (Indices 2 to the end)
        target_lidar = traj[past_end:fut_end, 2:]
        return past_data, future_cmds, target_lidar


if __name__ == "__main__":
    #print("data loading main")
    print("data")
    mydata = MyDataTest()
    print(mydata[0])
    print(mydata[1])
    print(mydata[2])
    print("loader")
    loader = DataLoader(mydata, batch_size=1, shuffle=False,num_workers=0,pin_memory=False)
    for a in loader:
        print("past",a[0])
        print("fut cmd",a[1])
        print("fut lidar",a[2])
