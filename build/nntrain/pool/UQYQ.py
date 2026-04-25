#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import conf

class UQYQ(Dataset):
    def __init__(self, io_config):
        self.io_cfg = io_config
        T3_uq, T3_yq = load_trajectories(conf.DATA_UQYQ_PATH)
        speed_i = 0
        angle_i = 1
        self.raw_data = {
            conf.CMD_SPEED: T3_yq[:,:,speed_i:speed_i+1],
            conf.CMD_ANGLE: T3_yq[:,:,angle_i:angle_i+1],
            conf.MES_LIDAR: T3_uq
        }
        
        self.past_win = self.io_cfg["past_window"]
        self.fut_win  = self.io_cfg["future_window"]
        
        self.traj_len = T3_yq.shape[1]
        self.subtrajs_per_traj = self.traj_len - (self.past_win + self.fut_win) + 1
        self.num_trajs = T3_yq.shape[0]

        # e.g., Speed -> size=1, mean=vec(1), std=vec(1), scale=vec(1) aka 1/28
        # e.g., Lidar -> size=360, mean=vec(360), std=vec(360), scale=1/std
        self.stats = {
            conf.CMD_SPEED: {"size": T3_yq[0,0,speed_i:speed_i+1].size(), "train_offset": None, "train_scale": None},
            conf.CMD_ANGLE: {"size": T3_yq[0,0,angle_i:angle_i+1].size(), "train_offset": None, "train_scale": None},
            conf.MES_LIDAR: {"size": T3_uq[0,0,:].size(),                 "train_offset": None, "train_scale": None},
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
    
    for file in glob.glob(datafiles_path):
        df = pd.read_csv(file, header=0)
        if df.shape[1] < 5:
            raise ValueError(f"{file} must contain at least three u_q columns and two y_q columns")
        # Robustctl writes all u_q columns first, followed by y_q_v and y_q_delta.
        uq = df.iloc[1:, :-2].values.astype(np.float32)
        yq = df.iloc[1:, -2:].values.astype(np.float32)
        sequences.append(torch.tensor(uq))
        targets.append(torch.tensor(yq))
    if not sequences:
        raise FileNotFoundError(f"no UQYQ files matched {datafiles_path}")
    seqs_torch = pad_2Dseq_start(sequences, 10, False)
    tarjs_torch = pad_2Dseq_start(targets, 10, True)
    return seqs_torch, tarjs_torch

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
