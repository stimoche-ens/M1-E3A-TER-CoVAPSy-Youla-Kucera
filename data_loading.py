#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import pandas as pd
import numpy as np
import glob

DATA_PATH = "webots_straight_track_2026.01.15/*.csv"

def load_trajectories(datafiles_path):
    sequences = []
    targets = []
    
    for file in glob.glob(datafiles_path):
        df = pd.read_csv(file, header=0)
        # Col 0: useless, 1-2: controls, 3-362: measures
        controls = df.iloc[:, 1:3].values.astype(np.float32)
        measures = df.iloc[:, 3:363].values.astype(np.float32)
        
        # Prepend 10 rows of zeros for initial conditions
        init_zeros_c = np.zeros((10, 2), dtype=np.float32)
        init_zeros_m = np.zeros((10, 360), dtype=np.float32)
        
        seq_x = np.vstack([init_zeros_c, controls])
        seq_y = np.vstack([init_zeros_m, measures])
        
        sequences.append(torch.tensor(seq_x))
        targets.append(torch.tensor(seq_y))
        
    return sequences, targets

sequences, targets = load_trajectories(DATA_PATH)
# Pad sequences so they fit in a single tensor [Batch, Max_Len, Features]
# However, we keep track of original lengths for the LSTM
X_padded = pad_sequence(sequences, batch_first=True)
Y_padded = pad_sequence(targets, batch_first=True)
lengths = torch.tensor([len(s) for s in sequences])


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

a = torch.tensor([
[8,3,6,1,99],
])
b = torch.ones(2, 5)
c = torch.ones(3, 5)

print(a)
print(b)
print(c)
seq = [a,b,c]


print("thing")

#print(pad_sequence([a,b,c], batch_first=False))
print(pad_2Dseq_start(seq, 5, False))


