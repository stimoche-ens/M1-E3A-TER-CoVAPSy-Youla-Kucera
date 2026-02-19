#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import conf
else:
    import conf

class MyArmax(Dataset):
    def __init__(self):
        self.past_win = 5
        self.lidar_delta = 2 # number of adjacent lidar angles
        self.lidar_min = -90
        self.lidar_max = 90
        self.lidar_step = 10 # step between two lidar angles
        if (self.lidar_min - self.lidar_step*self.lidar_delta < -180) or (self.lidar_max + self.lidar_step*self.lidar_delta > 179):
            print("Error: self.lidar_min and/or self.lidar_max, conjugated with self.lidar_step*self.lidar_delta go out of [-180, 179] bounds")
            return None
        self.lidar_index_range=range(self.lidar_min,self.lidar_max + self.lidar_step,self.lidar_step)
        #self.fut_win  = 20

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        ctl, meas = load_trajectories(os.path.abspath(os.path.join(parent_dir, conf.DATA_PATH)))
        speed_i = 0
        angle_i = 1
        self.raw_data = torch.cat((ctl,meas),dim=2)
        self.initial_conditions = self.raw_data[0,0,3:]
        self.raw_data[:,:,3:] -= self.initial_conditions
        
        self.traj_len = meas.shape[1]
        self.subtrajs_per_traj = self.traj_len - (self.past_win+1) + 1 # (self.past_win+1) because nth order implies a0+a1+....+an (n+1 samples)
        self.num_trajs = meas.shape[0]

    def __len__(self):
        return len(self.lidar_index_range)

    def __getitem__(self, idx):
        #traj_idx  = idx# // self.subtrajs_per_traj
        #start_row = idx % self.subtrajs_per_traj
        
        #t_start = start_row
        #t_end   = start_row + self.past_win + 1
        lidari=3+180+self.lidar_index_range[idx]
        totalmatrices = [0 for i in range(0, self.num_trajs)]
        start_row = self.past_win
        for traj_idx in range(0, self.num_trajs):
            cmdmatrix = self.raw_data[traj_idx, start_row:, 0:2]
            lidarmatrix = torch.cat([self.raw_data[traj_idx, start_row-i:-i, lidari:lidari+1] for i in range(1,self.past_win+1)], dim=1)
            if self.lidar_delta:
                lidarmatrix_deltabefore = torch.cat([torch.cat([self.raw_data[traj_idx, start_row-i:-i, lidari+i_lidar_delta*self.lidar_step:lidari+i_lidar_delta*self.lidar_step+1] for i in range(1,self.past_win+1)], dim=1) for i_lidar_delta in range(-self.lidar_delta,0)], dim=1)
                lidarmatrix_deltaafter  = torch.cat([torch.cat([self.raw_data[traj_idx, start_row-i:-i, lidari+i_lidar_delta*self.lidar_step:lidari+i_lidar_delta*self.lidar_step+1] for i in range(1,self.past_win+1)], dim=1) for i_lidar_delta in range(1, self.lidar_delta+1)], dim=1)
                totalmatrices[traj_idx] = torch.cat([cmdmatrix,lidarmatrix_deltabefore,lidarmatrix,lidarmatrix_deltaafter], dim=1)
            else:
                totalmatrices[traj_idx] = torch.cat([cmdmatrix,lidarmatrix], dim=1)
        totalmatrix = torch.cat(totalmatrices, dim=0)
        return totalmatrix

    def get_datavec(self):
        thelen = len(self.lidar_index_range)
        datavec_blocks = [0 for i in range(0, thelen)]
        start_row = self.past_win
        print(f"start_row: {start_row}")
        for lidar_index_index in range(0, thelen):
            lidari = 3+180+self.lidar_index_range[lidar_index_index]
            datavec_blocks[lidar_index_index] = self.raw_data[:,start_row:,lidari].contiguous().view(-1, 1)
        return torch.squeeze(torch.cat(datavec_blocks,dim=0))




def load_trajectories(datafiles_path):
    sequences = []
    targets = []
    mymax=0
    
    print(datafiles_path)
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
    armax = MyArmax()
    print("done loading")
    print(armax[0].size())
    block0 = armax[0]
    nb_blocks = len(armax)
    block_width = block0.size(1)
    block_height = block0.size(0)
    toeplitz_rows = [0 for i in range(0, nb_blocks)]
    for i in range(0,nb_blocks):
        print("block", i, ": begin")
        cols_before = torch.zeros((block_height, i*block_width))
        cols_after  = torch.zeros((block_height, (nb_blocks-i-1)*block_width))
        print("catting colons...")
        new_row = torch.cat((cols_before, armax[i], cols_after), dim=1)
        toeplitz_rows[i] = new_row
    print(f"Done with blocks, creating giant toeplitz of size ({block_height*nb_blocks},{block_width*nb_blocks})...")
    toeplitz = torch.cat(toeplitz_rows, dim=0)
    print("Done creating giant toeplitz")
    AT = torch.transpose(toeplitz,0,1)
    ATA_1 = torch.inverse(torch.matmul(AT, toeplitz))
    ATA_1AT = torch.matmul(ATA_1, AT)
    print("Done calculating final MATRIX. Final size: ", ATA_1AT.size())
    parameters = torch.matmul(ATA_1AT, armax.get_datavec())
    print(f"Done calculating final parameters. Size of parameters: {parameters.size()}")
    p_np = parameters.numpy()
    df = pd.DataFrame(p_np)
    df.to_csv("parameters.csv",index=False)
    


    

