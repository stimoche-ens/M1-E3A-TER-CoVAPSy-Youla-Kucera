#!/usr/bin/env python3

import torch
import torch.nn as nn
import conf
import libmy.libmodel as lmodel

class MyQParameter(lmodel.NormAwareModule):
    IO_CONFIG = {
        "past_window": 50,
        "future_window": 50,
        "inputs": [
            ("past", [conf.CMD_SPEED, conf.CMD_ANGLE, conf.RES_LIDAR])
        ],
        "outputs": [
            ("future", [conf.CMD_SPEED_Q, conf.CMD_ANGLE_Q])
        ]
    }

    def __init__(self, dataset_stats, hidden_dim=128):
        super().__init__(dataset_stats)

        self.q_parameter_conv = self.build_input_layer(
            input_idx=0,
            LayerClass=nn.Conv1d,
            out_channels=hidden_dim,
            kernel_size=5,
            padding=4
        )

        self.seq_expander = nn.Linear(hidden_dim, self.IO_CONFIG["future_window"] * hidden_dim)

        self.output_proj = self.build_output_layer(
            output_idx=0,
            LayerClass=nn.Linear,
            in_features=hidden_dim
        )

    def forward(self, past_data, target_cmds=None):
        past_data_t = past_data.transpose(1, 2)
        q_feat_t = self.q_parameter_conv(past_data_t)[..., :-4]
        h_q = q_feat_t[..., -1]

        seq_feat = self.seq_expander(torch.tanh(h_q))
        seq_feat = seq_feat.view(seq_feat.size(0), self.IO_CONFIG["future_window"], -1)

        return self.output_proj(seq_feat)

    @staticmethod
    def get_onnx_metadata(device='cpu'):
        past_dim = 1 + 1 + 360
        return {
            "input_dummies": (torch.randn(1, 50, past_dim, device=device),),
            "input_names": ['past_50_u_and_residuals'],
            "output_names": ['future_50_q_cmds']
        }
