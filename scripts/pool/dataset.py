#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
import glob
import conf

CONTROL_IMPORTANCE = 10.0

def load_trajectories(datafiles_path):
    sequences = []
    targets = []
    mymax=0
    
    for file in glob.glob(datafiles_path):
        df = pd.read_csv(file, header=0)
        # Col 0: useless, 1-2: controls, 3-362: measures
        controls = df.iloc[:, 1:3].values.astype(np.float32)
        #controls[:, 0] /= conf.MAX_SPEED
        #controls[:, 1] /= conf.MAX_ANGLE
        measures = df.iloc[:, 3:363].values.astype(np.float32)
        curr_max = np.max(measures)
        mymax=max(mymax,curr_max)
        #measures /= conf.MAX_LIDAR
        sequences.append(torch.tensor(controls))
        targets.append(torch.tensor(measures))
    controls_padded = pad_2Dseq_start(sequences, 10, False)
    measures_padded = pad_2Dseq_start(targets, 10, True)
    return controls_padded, measures_padded

def pad_2D_start(mat, pad_len, copy_init_value=False):
    if copy_init_value:
        init_padding = mat[0:1,:].expand(pad_len,-1)
    else:
        pad_width = mat.size(1)
        init_padding = torch.zeros(pad_len, pad_width)
    return torch.cat([init_padding,mat], dim=0)

def pad_2Dseq_start(seq, pad_len, copy_init_value=False):
    max_seqlen = max([s.size(0) for s in seq])
    seq_out = [pad_2D_start(s,pad_len+max_seqlen - s.size(0), copy_init_value) for s in seq]
    output_tensor = torch.stack(seq_out, dim=0)
    return output_tensor


from torch.utils.data import Dataset, DataLoader

class StraightTrack(Dataset):
    def __init__(self):
        # controls, measures = load_trajectories(conf.DATA_PATH)
        # tensor_3d shape: [Depth (Traj), Rows (Samples), Cols (Data)]
        # 1. Define Robust Scaling Vectors (Used for Loading AND Weight Patching)
        # Encoder Input Structure: [Speed, Angle, Lidar_1 ... Lidar_360]
        self.enc_scale = torch.ones(362)
        self.enc_scale[0] = conf.MAX_SPEED / CONTROL_IMPORTANCE
        self.enc_scale[1] = conf.MAX_ANGLE / CONTROL_IMPORTANCE
        self.enc_scale[2:] = conf.MAX_LIDAR

        # Decoder Input Structure: [Lidar_1 ... Lidar_360, Speed, Angle]
        self.dec_scale = torch.ones(362)
        self.dec_scale[0:360] = conf.MAX_LIDAR
        self.dec_scale[360] = conf.MAX_SPEED / CONTROL_IMPORTANCE
        self.dec_scale[361] = conf.MAX_ANGLE / CONTROL_IMPORTANCE

        # Output Structure: [Lidar_1 ... Lidar_360]
        self.out_scale = torch.tensor(conf.MAX_LIDAR)

        # 2. Load and Scale Data
        controls, measures = load_trajectories(conf.DATA_PATH)
        
        # Apply the scaling defined above
        # Controls (Col 0,1) correspond to enc_scale[0,1]
        controls[:, :, 0] /= self.enc_scale[0]
        controls[:, :, 1] /= self.enc_scale[1]
        
        # Measures (Col 2..362) correspond to enc_scale[2:]
        measures /= self.enc_scale[2:] # All lidar points share the same scale

        self.data = torch.cat([controls, measures], dim=2)
        self.num_trajs, self.traj_len, _ = self.data.shape
        self.window_past = conf.PAST_WINDOW
        self.window_fut = conf.FUTURE_WINDOW
        
        # Calculate how many valid 100-step sequences exist PER trajectory
        self.samples_per_traj = self.traj_len - (self.window_past + self.window_fut) + 1
        
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
    print("data loading main")
    controls, measures = load_trajectories(conf.DATA_PATH)

