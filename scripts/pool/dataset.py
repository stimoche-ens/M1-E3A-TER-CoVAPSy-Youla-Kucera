#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
import glob
import conf

class StraightTrack:
    def __init__(self):
        ctl, meas = load_trajectories(conf.DATA_PATH)
        self.datadict = {conf.DATADICT_CMD_SPEED: ctl[:,:,0:1],
                         conf.DATADICT_CMD_ANGLE: ctl[:,:,1:2],
                         conf.DATADICT_MES_LIDAR: meas}

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


if __name__ == "__main__":
    print("data loading main")
    controls, measures = load_trajectories(conf.DATA_PATH)
