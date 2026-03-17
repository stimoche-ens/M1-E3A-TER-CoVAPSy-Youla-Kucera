#!/usr/bin/env python3

import torch
import torch.nn as nn
import conf
import libmy.libmodel as lmodel

class MyYKController(lmodel.NormAwareModule):
    """
    Data-Driven Youla-Kucera Controller.
    
    Structure: u = u_nom + u_Q
    Where:
      - u_nom is a nominal baseline controller (linear map of the latest state).
      - u_Q is the Youla parameter Q(z) implemented as a stable FIR filter 
        (1D Convolution) over the past trajectory window.
    """
    IO_CONFIG = {
        "past_window": 50,
        "future_window": 50, 
        "inputs": [
            # Input 0: Past trajectory (Speed, Angle, Lidar)
            ("past", [conf.CMD_SPEED, conf.CMD_ANGLE, conf.MES_LIDAR]) 
        ],
        "outputs": [
            # Output 0: Future predicted control commands
            ("future", [conf.CMD_SPEED, conf.CMD_ANGLE])
        ]
    }

    def __init__(self, dataset_stats, hidden_dim=128):
        super().__init__(dataset_stats)
        
        # --- 1. NOMINAL CONTROLLER (K_nom) ---
        # A simple linear layer acting on the most recent timestep
        self.nominal_controller = self.build_input_layer(
            input_idx=0, 
            LayerClass=nn.Linear, 
            out_features=hidden_dim
        )
        
        # --- 2. YOULA PARAMETER Q(z) ---
        # An FIR filter is inherently Bounded-Input Bounded-Output (BIBO) stable.
        # We implement this as a causal 1D Convolution over the past window.
        pack = self.input_packs[0]
        self.q_parameter_conv = nn.Conv1d(
            in_channels=pack.total_width, 
            out_channels=hidden_dim, 
            kernel_size=5,  # FIR filter length
            padding=4       # Causal padding (kernel_size - 1)
        )
        # Register the Conv1d layer manually for normalization awareness
        # since it uses in_channels instead of in_features/input_size
        self._register(self.q_parameter_conv, pack, mode="input")
        
        # --- 3. OUTPUT PROJECTION ---
        self.output_proj = self.build_output_layer(
            output_idx=0,
            LayerClass=nn.Linear,
            in_features=hidden_dim
        )

    def forward(self, past_data, target_cmds=None):
        """
        past_data: [Batch, Time=50, Features=362]
        Returns:   [Batch, Time=50, Features=2] (Future Commands)
        """
        batch_size, seq_len, features = past_data.shape
        
        # --- Nominal Path ---
        # K_nom processes the input directly. For sequence-to-sequence generation,
        # we process the whole past window through the linear nominal layer.
        nominal_feat = self.nominal_controller(past_data) # [B, T, hidden_dim]
        
        # --- Q(z) Parameter Path ---
        # PyTorch Conv1d expects [Batch, Channels, Time]
        past_data_t = past_data.transpose(1, 2) 
        
        # Apply causal FIR filter
        q_feat_t = self.q_parameter_conv(past_data_t)
        
        # Remove future-leaking padded elements to maintain causality
        q_feat_t = q_feat_t[:, :, :-4] 
        q_feat = q_feat_t.transpose(1, 2) # Back to [B, T, hidden_dim]
        
        # Apply a stable activation (e.g., Tanh keeps the Q operator bounded)
        q_feat = torch.tanh(q_feat)
        
        # --- Combine and Project ---
        # u = u_nom + Q(epsilon)
        combined_feat = nominal_feat + q_feat
        
        # Project to output dimensions (Speed, Angle)
        # We expand the single-step projection across the future window length
        out_single_step = self.output_proj(combined_feat[:, -1, :]) # Take last timestep
        
        # Repeat the predicted action to match future_window=50
        # (Alternatively, you can use an RNN decoder here if dynamic future planning is needed)
        future_cmds = out_single_step.unsqueeze(1).repeat(1, self.IO_CONFIG["future_window"], 1)
        
        return future_cmds

    @staticmethod
    def get_onnx_metadata(device='cpu'):
        past_dim = 1 + 1 + 360 # Speed + Angle + Lidar
        return {
            "input_dummies": (torch.randn(1, 50, past_dim, device=device),),
            "input_names": ['past_50_steps'],
            "output_names": ['future_50_cmds']
        }

if __name__ == "__main__":
    print("MyYKController loaded successfully.")
