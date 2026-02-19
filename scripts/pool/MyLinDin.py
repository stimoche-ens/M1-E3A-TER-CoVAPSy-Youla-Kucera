#!/usr/bin/env python3

#import torch
import torch.nn as nn
import conf
import libmy.libmodel as lmodel

class MyLSTM(lmodel.NormAwareModule):
    IO_CONFIG = { # Single Source of Truth
        "past_window": 3,
        "future_window": 3,
        "inputs": [
            # mode (either "past" or "future") , keys (a list)
            ("past",   [conf.CMD_SPEED, conf.CMD_ANGLE, conf.MES_LIDAR]), # Index 0: Main Input
            ("future", [conf.CMD_SPEED, conf.CMD_ANGLE])              # Index 1: Future Cmds
        ],
        "outputs": [
            ("future", [conf.MES_LIDAR])                          # Index 0: Prediction
        ]
    }

    def __init__(self, dataset_stats, hidden_dim=256):
        #super(MyLSTM, self).__init__()
        # Initialize Base (builds the Packs from IO_CONFIG)
        super().__init__(dataset_stats)
        
        # --- BUILD LAYERS ---
        
        self.encoder = self.build_input_layer(
            input_idx=0, 
            LayerClass=nn.Linear, 
            hidden_size=hidden_dim, 
            batch_first=True
        )

        self.decoder = self.build_input_layer(
            output_idx=0,
            LayerClass=nn.Linear,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, past_data, future_cmds, target_lidar=None):
        """
        past_data:   [Batch, 50, 362] (Past Commands + Past LiDAR)
        future_cmds: [Batch, 50, 2]   (Future Commands only)
        """
        _, (hidden, cell) = self.encoder(past_data)
        return self.decoder(future_cmds, (hidden, cell))

    #@staticmethod
    #def get_onnx_metadata(device='cpu'):
    """
    Returns metadata for ONNX export, ensuring tensors are on the correct device.
    """
    #past_dim = conf.CMD_DIM + conf.LIDAR_DIM # 362
    #return {
        #"input_dummies": (torch.randn(1, conf.PAST_WINDOW, past_dim, device=device), torch.randn(1, conf.FUTURE_WINDOW, conf.CMD_DIM, device=device)),
        #"input_names": ['past_50_steps', 'future_50_cmds'],
        #"output_names": ['future_50_lidar']
        #},
    #}


if __name__ == "__main__":
    print("LSTM.py")
