#!/usr/bin/env python3

import torch
import torch.nn as nn
import conf

class MyLSTM(nn.Module):
    @staticmethod
    def get_onnx_metadata(device='cpu'):
        """
        Returns metadata for ONNX export, ensuring tensors are on the correct device.
        """
        past_dim = conf.CMD_DIM + conf.LIDAR_DIM # 362
        return {
            "input_dummies": (torch.randn(1, conf.PAST_WINDOW, past_dim, device=device), torch.randn(1, conf.FUTURE_WINDOW, conf.CMD_DIM, device=device)),
            "input_names": ['past_50_steps', 'future_50_cmds'],
            "output_names": ['future_50_lidar']
        }

    def __init__(self, cmd_dim=2, lidar_dim=360, hidden_dim=256):
        super(MyLSTM, self).__init__()
        
        self.encoder = nn.LSTM(input_size=cmd_dim + lidar_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(cmd_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, lidar_dim)

    def forward(self, past_data, future_cmds, target_lidar=None):
        """
        past_data:   [Batch, 50, 362] (Past Commands + Past LiDAR)
        future_cmds: [Batch, 50, 2]   (Future Commands only)
        """
        # Run the LSTM over the past 50 steps to get the final "mental state" (hidden state)
        # _ contains all outputs, we only care about the final (hidden, cell) state
        _, (hidden, cell) = self.encoder(past_data)
        decoder_out, _ = self.decoder(future_cmds, (hidden, cell))
        return self.output_layer(decoder_out)


if __name__ == "__main__":
    print("LSTM.py")
