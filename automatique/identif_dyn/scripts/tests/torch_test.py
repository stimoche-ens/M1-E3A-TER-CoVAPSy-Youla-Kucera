#!/usr/bin/env python3

import torch
#from torch.nn.utils.rnn import pad_sequence

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


