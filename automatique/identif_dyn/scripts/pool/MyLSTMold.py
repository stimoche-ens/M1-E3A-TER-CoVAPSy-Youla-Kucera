#!/usr/bin/env python3

import torch
import torch.nn as nn
import conf

class MyLSTMold(nn.Module):
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
        
        # RNN 1: The Encoder
        # Processes the Past (50 steps). Input: Commands + LiDAR
        self.encoder = nn.LSTM(input_size=cmd_dim + lidar_dim, hidden_size=hidden_dim, batch_first=True)
        
        # RNN 2: The Decoder Cell
        # Processes the Future (50 steps). Input: Previous LiDAR + Current Command
        #self.decoder_cell = nn.LSTMCell(input_size=lidar_dim + cmd_dim, hidden_size=hidden_dim)
        self.decoder = nn.LSTM(lidar_dim + cmd_dim, hidden_dim, batch_first=True)
        
        # The output layer (Feed Forward)
        self.output_layer = nn.Linear(hidden_dim, lidar_dim)

    def forward(self, past_data, future_cmds, target_lidar=None):
        """
        past_data:   [Batch, 50, 362] (Past Commands + Past LiDAR)
        future_cmds: [Batch, 50, 2]   (Future Commands only)
        """
        
        # --- PHASE 1: ENCODE PAST ---
        # Run the LSTM over the past 50 steps to get the final "mental state" (hidden state)
        # _ contains all outputs, we only care about the final (hidden, cell) state
        _, (hidden, cell) = self.encoder(past_data)
        if target_lidar is not None:
            # We construct the 50-step input tensor for the decoder all at once.
            # Input t requires the Lidar from t-1.
            
            # First element: Last lidar from the past
            lidar_init = past_data[:, -1:, 2:] 
            
            # Next 49 elements: The first 49 targets
            lidar_future = target_lidar[:, :-1, :] 
            
            # Combine them to get the "Previous Lidar" for all 50 steps
            decoder_lidar_inputs = torch.cat([lidar_init, lidar_future], dim=1)
            
            # Concatenate with Future Commands
            decoder_inputs = torch.cat([decoder_lidar_inputs, future_cmds], dim=2)
            
            # Run the ENTIRE 50-step sequence through the LSTM in one single C++ operation
            decoder_out, _ = self.decoder(decoder_inputs, (hidden, cell))
            
            #print("size(decoder_lidar_inputs)", decoder_lidar_inputs.size())
            #print("size(future_cmds)", future_cmds.size())
            #print("size(decoder_inputs)", decoder_inputs.size())
            return self.output_layer(decoder_out)

        # 3. EVALUATION REGIME (For eval.py - Standard loop)
        else:
            predictions = []
            curr_lidar = past_data[:, -1:, 2:]
            
            for t in range(future_cmds.size(1)):
                curr_cmd = future_cmds[:, t:t+1, :]
                decoder_input = torch.cat([curr_lidar, curr_cmd], dim=2)
                
                # Step the decoder 1 time step
                out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
                
                curr_lidar = self.output_layer(out)
                predictions.append(curr_lidar)
                
            return torch.cat(predictions, dim=1)


"""
        # We need the hidden state for the LSTMCell. 
        # LSTM outputs (1, Batch, Hidden), we need (Batch, Hidden)
        h_t = hidden.squeeze(0)
        c_t = cell.squeeze(0)
        
        # The first input to the decoder is the very last LiDAR reading from the past
        # past_data is [Batch, 50, 362]. 
        # We assume cols 2-end are LiDAR (as per your description)
        last_lidar_prediction = past_data[:, -1, 2:] 
        
        predictions = []

        # --- PHASE 2: DECODE FUTURE (Step-by-Step) ---
        # Loop for 50 steps into the future
        for t in range(future_cmds.size(1)):
            
            # Get the command we KNOW will happen at this future step
            next_cmd = future_cmds[:, t, :]
            
            # Prepare input: Concat [Last LiDAR prediction, Next Command]
            # This restores the input size to 362
            decoder_input = torch.cat([last_lidar_prediction, next_cmd], dim=1)
            
            # Run the RNN Cell one step
            h_t, c_t = self.decoder_cell(decoder_input, (h_t, c_t))
            
            # Feed Forward to predict LiDAR
            lidar_out = self.output_layer(h_t)
            
            # Save prediction
            predictions.append(lidar_out.unsqueeze(1))
            
            # Update the "last lidar" for the next loop iteration
            last_lidar_prediction = lidar_out
            
        # Concatenate all 50 predictions
        return torch.cat(predictions, dim=1)
"""


if __name__ == "__main__":
    print("LSTM.py")
