#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import conf

PAST_WINDOW = 50
FUTURE_WINDOW = 50

SPEED_IMPORTANCE = 200
ANGLE_IMPORTANCE = 200
LIDAR_IMPORTANCE = 400

IDX = (
conf.DATADICT_CMD_SPEED,
conf.DATADICT_CMD_ANGLE,
conf.DATADICT_MES_LIDAR,
)




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

    def __init__(self, datadict, hidden_dim=256):
        super(MyLSTM, self).__init__()
 
        self.dataset = self._Dataset(self, datadict)
        self.cmd_dim   = 2
        self.lidar_dim = self.dataset.lidar_len
        self.encoder = nn.LSTM(input_size=self.cmd_dim + self.lidar_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=self.cmd_dim,                  hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, self.lidar_dim)
        #print("self.output_layer.weight.size()", self.output_layer.weight.size())
        #print("self.dataset.out_scale[0,0,:].unsqueeze(1).size()", self.dataset.out_scale[0,0,:].unsqueeze(1).size())

    def denormalise_weights(self):
        """
        Modifies the trained model weights in-place.
        It divides input weights by the dataset's scaling factors,
        so the model accepts RAW world units (12000, 28) at runtime.
        """
        print("🔧 Patching weights to accept Raw World Units...")
        device = self.output_layer.weight.device
        lmean = self.dataset.lidar_mean.view(-1).to(device)
        std  = self.dataset.lidar_std.view(-1).to(device)

        input_dim = self.encoder.input_size
        global_mean = torch.zeros(input_dim, device=device)
        global_mean[self.dataset.LIDAR_IDX:self.dataset.LIDAR_IDX+self.dataset.lidar_len] = lmean
        global_std  = torch.ones(input_dim, device=device)
        global_std[self.dataset.LIDAR_IDX:self.dataset.LIDAR_IDX+self.dataset.lidar_len] = std

        with torch.no_grad():
            self.encoder.weight_ih_l0 *= self.dataset.enc_in_scale[0,0,:].to(device) / global_std
            self.encoder.bias_ih_l0   -= torch.matmul(self.encoder.weight_ih_l0, global_mean)
            self.decoder.weight_ih_l0 *= self.dataset.dec_in_scale[0,0,:].to(device)
            self.output_layer.weight  *= std.view(-1, 1)/ self.dataset.out_scale[0,0,:].unsqueeze(1).to(device)
            self.output_layer.bias    *= std.view(-1)   / self.dataset.out_scale[0,0,:].to(device)
            self.output_layer.bias    += lmean
        print("✅ Weights patched successfully.")

    def forward(self, past_data, future_cmds, target_lidar=None):
        """
        past_data:   [Batch, 50, 362] (Past Commands + Past LiDAR)
        future_cmds: [Batch, 50, 2]   (Future Commands only)
        """
        _, (hidden, cell) = self.encoder(past_data)
        decoder_out, _ = self.decoder(future_cmds, (hidden, cell))
        return self.output_layer(decoder_out)

    class _Dataset(Dataset):
        #data_dict size (trajectory_idx, line_idx, col_idx)
        def __init__(self, parent, datadict):
            self.parent = parent
            datadict = copy.deepcopy(datadict)
            self.num_trajs = datadict[conf.DATADICT_MES_LIDAR].size(0)
            self.traj_len  = datadict[conf.DATADICT_MES_LIDAR].size(1)
            self.lidar_len = datadict[conf.DATADICT_MES_LIDAR].size(2)
            self.SPEED_IDX = IDX.index(conf.DATADICT_CMD_SPEED) if IDX.index(conf.DATADICT_CMD_SPEED) > IDX.index(conf.DATADICT_MES_LIDAR) else IDX.index(conf.DATADICT_CMD_SPEED) + self.lidar_len - 1
            self.ANGLE_IDX = IDX.index(conf.DATADICT_CMD_ANGLE) if IDX.index(conf.DATADICT_CMD_ANGLE) > IDX.index(conf.DATADICT_MES_LIDAR) else IDX.index(conf.DATADICT_CMD_ANGLE) + self.lidar_len - 1
            self.LIDAR_IDX = IDX.index(conf.DATADICT_MES_LIDAR)
            SPEED_MAX = torch.max(torch.abs(datadict[conf.DATADICT_CMD_SPEED]))
            ANGLE_MAX = torch.max(torch.abs(datadict[conf.DATADICT_CMD_ANGLE]))
            LIDAR_MAX = torch.max(torch.abs(datadict[conf.DATADICT_MES_LIDAR]))
            if SPEED_MAX == 0: SPEED_MAX = 1.0
            if ANGLE_MAX == 0: ANGLE_MAX = 1.0
            if LIDAR_MAX == 0: LIDAR_MAX = 1.0

            flat_lidar = datadict[conf.DATADICT_MES_LIDAR].view(-1, datadict[conf.DATADICT_MES_LIDAR].shape[-1]) 
            self.lidar_mean = torch.mean(flat_lidar, dim=0).view(1, 1, -1)
            self.lidar_std  = torch.std(flat_lidar, dim=0).view(1, 1, -1)
            self.lidar_std[self.lidar_std < 1e-6] = 1.0
            datadict[conf.DATADICT_MES_LIDAR] = (datadict[conf.DATADICT_MES_LIDAR] - self.lidar_mean) / self.lidar_std

            self.speed_coeff = SPEED_IMPORTANCE / SPEED_MAX
            self.angle_coeff = ANGLE_IMPORTANCE / ANGLE_MAX
            self.lidar_coeff = LIDAR_IMPORTANCE / self.lidar_len

            self.enc_in_scale = torch.ones(self.num_trajs, self.traj_len, 2+self.lidar_len)
            self.enc_in_scale[:,:,self.SPEED_IDX] = self.speed_coeff
            self.enc_in_scale[:,:,self.ANGLE_IDX] = self.angle_coeff
            self.enc_in_scale[:,:,self.LIDAR_IDX:self.LIDAR_IDX+self.lidar_len] = self.lidar_coeff

            for unitIDX in IDX:
                print("unitIDX:", unitIDX, datadict[unitIDX].size())
            self.datatensor = torch.cat([datadict[unitIDX] for unitIDX in IDX], dim=2)
    
            self.dec_in_scale = torch.ones(self.num_trajs, self.traj_len, 2)
            self.dec_in_scale[:,:,0] *= self.speed_coeff if self.SPEED_IDX < self.ANGLE_IDX else self.angle_coeff
            self.dec_in_scale[:,:,1] *= self.angle_coeff if self.SPEED_IDX < self.ANGLE_IDX else self.speed_coeff
    
            self.out_scale = torch.ones(self.num_trajs, self.traj_len, self.lidar_len) * self.lidar_coeff
    
            self.datatensor = torch.mul(self.datatensor, self.enc_in_scale)

            self.window_past = PAST_WINDOW
            self.window_fut  = FUTURE_WINDOW
            
            self.subtrajs_per_traj = self.traj_len - (self.window_past + self.window_fut) + 1
            
        def __len__(self):
            # Total samples = Trajectories * Valid Samples per Trajectory
            return self.num_trajs * self.subtrajs_per_traj
    
        def __getitem__(self, idx):
            # Determine which trajectory and which start row this index belongs to
            traj_idx = idx // self.subtrajs_per_traj
            start_row = idx % self.subtrajs_per_traj
            # Slicing indices
            past_end = start_row + self.window_past
            fut_end = past_end + self.window_fut
            # Get the single trajectory
            traj = self.datatensor[traj_idx]
            # --- SLICE DATA ---
            # Past: EVERYTHING (Controls + LiDAR)
            past_data = traj[start_row:past_end, :] 
            # Future: Controls ONLY (Indices 0 and 1)
            future_cmds = torch.cat((traj[past_end:fut_end, self.SPEED_IDX:self.SPEED_IDX+1],traj[past_end:fut_end, self.ANGLE_IDX:self.ANGLE_IDX+1]), dim=1)
            # Target: LiDAR ONLY (Indices 2 to the end)
            target_lidar = traj[past_end:fut_end, self.LIDAR_IDX:self.LIDAR_IDX+self.lidar_len]
            return {"inputs": (past_data, future_cmds), "outputs": (target_lidar,)}



if __name__ == "__main__":
    print("LSTM.py")
