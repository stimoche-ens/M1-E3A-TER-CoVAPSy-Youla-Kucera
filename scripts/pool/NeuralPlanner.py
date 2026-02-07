#!/usr/bin/env python3

import torch
import torch.nn as nn
import conf

class NeuralPlanner(nn.Module):
    @staticmethod
    def get_onnx_metadata(device='cpu'):
        """
        Returns metadata for ONNX export, ensuring tensors are on the correct device.
        """
        past_dim = conf.CMD_DIM + conf.LIDAR_DIM # 362
        return {
            "input_dummies": (torch.randn(1, conf.PAST_WINDOW, past_dim, device=device)),
            "input_names": ['past_50_steps'],
            "output_names": ['future_50_lidar']
        }

    def __init__(self, hidden_dim=128):
        super().__init__()
        
        input_dim = conf.CMD_DIM + conf.LIDAR_DIM
        output_dim = conf.CMD_DIM
        
        # Encoder: Digests the past
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Decoder: Generates the plan (50 steps)
        # We use a simple MLP that expands the hidden state into 50 steps
        self.plan_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, conf.FUTURE_WINDOW * output_dim)
        )

    def forward(self, past_data, dummy1=None, dummy2=None):
        # 1. ENCODE
        _, (hidden, _) = self.encoder(past_data)
        last_hidden = hidden[-1]
        
        # 2. PLAN
        # Output shape: [Batch, 50 * 2]
        flat_plan = self.plan_head(last_hidden)
        
        # Reshape to [Batch, 50, 2]
        plan = flat_plan.view(-1, conf.FUTURE_WINDOW, conf.CMD_DIM)
        
        return plan



if __name__ == "__main__":
    print("LSTM.py")
