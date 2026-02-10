#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import conf

class StraightTrack(Dataset):
    def __init__(self, io_config):
        ctl, meas = load_trajectories(conf.DATA_PATH)
        speed_i = 0
        angle_i = 1
        self.raw_data = {
            conf.CMD_SPEED: ctl[:,:,speed_i:speed_i+1],
            conf.CMD_ANGLE: ctl[:,:,angle_i:angle_i+1],
            conf.MES_LIDAR: meas
        }
        
        self.io_cfg = io_config
        self.past_win = self.io_cfg["past_window"]
        self.fut_win  = self.io_cfg["future_window"]
        
        self.traj_len = meas.shape[1]
        self.subtrajs_per_traj = self.traj_len - (self.past_win + self.fut_win) + 1
        self.num_trajs = meas.shape[0]

        # e.g., Speed -> size=1, mean=vec(1), std=vec(1), scale=vec(1) aka 1/28
        # e.g., Lidar -> size=360, mean=vec(360), std=vec(360), scale=1/std
        self.stats = {
            conf.CMD_SPEED: {"size": ctl[0,0,speed_i:speed_i+1].size(), "train_offset": None, "train_scale": None},
            conf.CMD_ANGLE: {"size": ctl[0,0,speed_i:speed_i+1].size(), "train_offset": None, "train_scale": None},
            conf.MES_LIDAR: {"size": meas[0,0,:].size(),                "train_offset": None, "train_scale": None},
        }

    def __len__(self):
        return self.num_trajs * self.subtrajs_per_traj

    def _get_slice(self, traj_idx, start_row, mode, keys):
        """Helper to fetch and concatenate specific columns for a specific time."""
        if mode == "past":
            t_start = start_row
            t_end   = start_row + self.past_win
        elif mode == "future":
            t_start = start_row + self.past_win
            t_end   = t_start + self.fut_win
            
        tensors = []
        for key in keys:
            # Retrieve specific column from raw storage
            # shape: [Batch, Time, Dim] -> Slice [Time_Window, Dim]
            data = self.raw_data[key][traj_idx, t_start:t_end, :]
            tensors.append(data)
            
        # 3. Concatenate Features (dim=1 is feature dim for 2D slice)
        return torch.cat(tensors, dim=1)

    def __getitem__(self, idx):
        traj_idx  = idx // self.subtrajs_per_traj
        start_row = idx % self.subtrajs_per_traj
        
        # Dynamic Input Construction
        input_list = []
        for mode, keys in self.io_cfg["inputs"]:
            input_list.append(self._get_slice(traj_idx, start_row, mode, keys))
            
        # Dynamic Output Construction
        output_list = []
        for mode, keys in self.io_cfg["outputs"]:
            output_list.append(self._get_slice(traj_idx, start_row, mode, keys))

        return {"inputs": tuple(input_list), "outputs": tuple(output_list)}


def load_trajectories(datafiles_path):
    sequences = []
    targets = []
    mymax=0
    
    for file in glob.glob(datafiles_path):
        df = pd.read_csv(file, header=0)
        # Col 0: useless, 1-2: controls, 3-362: measures
        controls = df.iloc[:, 1:3].values.astype(np.float32)
        measures = df.iloc[:, 3:363].values.astype(np.float32)
        curr_max = np.max(measures)
        mymax=max(mymax,curr_max)
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


if __name__ == "__main__":
    print("data loading main")
    controls, measures = load_trajectories(conf.DATA_PATH)




