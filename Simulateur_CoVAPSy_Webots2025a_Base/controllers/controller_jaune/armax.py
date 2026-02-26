#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import conf
else:
    import conf

class MyLinPerturb(Dataset):
    def init_cfg(self):
        self.win_radius = 5
        self.lidar_delta = 2 # number of adjacent lidar angles
        self.lidar_min = -90
        self.lidar_max = 90
        self.lidar_step = 10 # step between two lidar angles
        self.lidar_offset0 = 3+180
        self.lidar_maxstep = 30

    def init_lidar_idx(self):
        self.lidar_index_range=np.array(range(self.lidar_offset0 + self.lidar_min, self.lidar_offset0 + self.lidar_max + self.lidar_step, self.lidar_step))
        self.lidar_index_size=len(self.lidar_index_range)
        self.lidar_extindex_range=np.array(range(self.lidar_offset0 + self.lidar_min - self.lidar_delta*self.lidar_step, self.lidar_offset0 + self.lidar_max + self.lidar_delta*self.lidar_step + self.lidar_step, self.lidar_step))
        self.lidar_extindex_size=len(self.lidar_extindex_range)

    def init_params(self, params)
        #self.params_lidar = [parameters[2+(2+(2*self.lidar_delta+1)*self.win_radius)*lidar_index_index:(2+(2*self.lidar_delta+1)*self.win_radius)*(lidar_index_index+1)] for lidar_index_index in range(0,self.lidar_index_size)]
        #self.params_cmd   = [parameters[(2+(2*self.lidar_delta+1)*self.win_radius)*lidar_index_index:2+(2+(2*self.lidar_delta+1)*self.win_radius)*lidar_index_index] for lidar_index_index in range(0,self.lidar_index_size)]
        self.params_lidar = params[:self.lidartoep.size(1)]
        self.params_cmd = params[self.lidartoep.size(1):]
        self.params_lidar_inv = np.zeros([len(self.params_lidar)])
        self.params_cmd_inv = np.zeros([len(self.params_cmd)])
        self.params_lidar_inv[self.lidartoep_subblock_idx_w] = -self.params_lidar[self.lidartoep_subblock_idx_w]/self.params_cmd[self.cmdtoep_subblock_idx_w_1[:,1:2]]
        self.params_cmd_inv[self.lidartoep_subblock_idx_w]   = -self.params_cmd[self.cmdtoep_subblock_idx_w]/self.params_cmd[self.cmdtoep_subblock_idx_w_1[:,1:2]]
        self.params_cmd_inv[self.lidartoep_subblock_idx_w_1[:,1:2]] = 1/self.params_cmd[self.cmdtoep_subblock_idx_w_1[:,1:2]]

    def init_states(self): #lidar_rdy = lidar[self.lidar_extindex_range] - self.lidar0
        self.lidars_fut = np.zeros([self.win_radius+1,self.lidar_extindex_size])
        self.cmds_fut   = np.zeros([self.win_radius+1,2])
        self.lidartoep_block_height    = 2*self.lidar_delta+1
        self.lidartoep_subblock_width_min  = (self.lidar_delta+1)
        self.lidartoep_subblock_width_max  = (2*self.lidar_delta+1)
        self.lidartoep_subblock_idx_w  = [0 for i in self.lidar_extindex_range]
        self.lidartoep_subblock_idx_w_1= [0 for i in self.lidar_extindex_range]
        self.cmdtoep_subblock_idx_w  = np.array([[u+2*i*self.win_range for u in range(0, 2*(self.win_range+1))] for i in self.lidar_extindex_range])
        self.cmdtoep_subblock_idx_w_1= self.cmdtoep_subblock_idx_w[:,::(self.win_range+1)]
        self.lidartoep_subblock_idx_h  = np.array([[i] for i in range(0, self.lidar_extindex_size)])
        self.lidarrdy_subblock_idx     = [0 for i in self.lidar_extindex_range]
        block_offset = 0
        for angle_i in range(0,self.lidar_extindex_size):
            size = min([self.lidartoep_subblock_width_min+self.lidar_extindex_size-1-angle_i,self.lidartoep_subblock_width_min+angle_i,self.lidartoep_subblock_width_max])
            self.lidarrdy_subblock_idx[angle_i] = np.array(range(0,size))+sup(0,angle_i-self.win_range)
            self.lidartoep_subblock_idx_w[angle_i] = np.arraynp.array(range(0,size*self.win_range))+block_offset
            self.lidartoep_subblock_idx_w_1[angle_i] = self.lidartoep_subblock_idx_w[angle_i][::self.win_range]
            block_offset += size*self.win_range
        self.lidarrdy_subblock_idx = np.array(self.lidarrdy_subblock_idx)
        self.lidartoep_subblock_idx_w = np.array(self.lidartoep_subblock_idx_w)
        self.lidartoep_subblock_idx_w_1 = np.array(self.lidartoep_subblock_idx_w_1)
        self.cmdtoep = np.zeros((self.lidartoep_block_height*(self.win_radius+1), self.lidar_extindex_size*2*self.win_radius)) # all Toeplitz blocks
        self.lidartoep = np.zeros((self.lidartoep_block_height*(self.win_radius+1), self.win_radius*(2*sum(range(self.lidar_delta+1, 2*self.lidar_delta+1)) + self.lidar_index_size))) # all Toeplitz blocks

    def __init__(self, parameters, goal_speed, first_lidar):
        self.speed0 = goal_speed
        self.lidar0 = self.filter_lidar(first_lidar)
        self.tick = 0
        self.init_cfg()
        if (self.lidar_min - self.lidar_step*self.lidar_delta < -180) or (self.lidar_max + self.lidar_step*self.lidar_delta > 179):
            print("Error: self.lidar_min and/or self.lidar_max, conjugated with self.lidar_step*self.lidar_delta go out of [-180, 179] bounds")
            return None
        self.init_lidar_idx()
        self.init_states()
        self.init_params(parameters)

    def propagate_past_state_1time(self, past_idx, toep, toep_subblock_idx_w):
        toep[self.lidartoep_subblock_idx_h+self.lidartoep_block_height*(past_idx+1), toep_subblock_idx_w[:,1:]] = toep[self.lidartoep_subblock_idx_h+self.lidartoep_block_height*past_idx, toep_subblock_idx_w[:,:-1]]

    def update_state_instant(self, state_idx, past_idx, toep, toep_subblock_idx_w_1, val_array, val_subblock_idx=None):
        toep[self.lidartoep_subblock_idx_h+(self.lidartoep_block_height*state_idx), self.lidartoep_subblock_idx_w_1+past_idx] = val_array[val_subblock_idx] if val_subblock_idx else val_array

    def state_timestep(self, toep, step_height):
        toep[:-step_height,:] = toep[step_height:,:]
        toep[-step_height:,:] = 0

    def plan_lidar_trajectory(self):
        f = np.vectorize(lambda x: -np.sign(min(abs(x), self.lidar_maxstep)))
        self.cmds_fut[:,0] = self.speed0
        for step in range(0, win_radius):
            propagate_past_state_1time(step, self.cmdtoep, self.cmdtoep_subblock_idx_w)
            propagate_past_state_1time(step, self.cmdtoep, self.cmdtoep_subblock_idx_w)
            self.update_state_instant(step+1, 0, self.lidartoep, self.lidartoep_subblock_idx_w_1, self.lidars_fut[step,:], self.lidarrdy_subblock_idx)
            self.update_state_instant(step+1, 1, self.cmdtoep, self.cmdtoep_subblock_idx_w_1, self.cmds_fut[step,:], None)
            lidar_step_size = f(self.lidars_fut[step,:])
            self.lidars_fut[step+1,:] = self.lidars_fut[step,:] + lidar_step_size
            self.update_state_instant(step+1, 0, self.cmdtoep, self.cmdtoep_subblock_idx_w_1, [self.speed0, self.lidars_fut[step+1,:]], [i for i in range(0, self.lidar_extindex_size)])
            predicted_angles = self.lidartoep@self.params_lidar_inv + self.cmdtoep@self.params_cmd_inv
            self.cmds_fut[step+1,1]   = np.mean(predicted_angles)
            self.update_state_instant(step+1, 0, self.cmdtoep, self.cmdtoep_subblock_idx_w_1, [self.speed0, self.cmds_fut[step+1,1]], None)
            

    def filter_lidar(self, lidar):
        return lidar[self.lidar_extindex_range] #[lidar[i] for i in self.lidar_extindex_range]

    def save_lidar_state(self, lidar_rdy):
        self.propagate_past_state_1time(0, self.lidartoep, self.lidartoep_subblock_idx_w)
        self.update_state_past(1, 1, self.lidartoep, self.lidartoep_subblock_idx_w_1, lidar_rdy, self.lidarrdy_subblock_idx)
        self.state_timestep(self.lidartoep, self.lidartoep_block_height)

    def save_cmd_state(self, cmd_rdy):
        self.propagate_past_state_1time(0, self.cmdtoep, self.cmdtoep_subblock_idx_w)
        self.update_state_past(1, 1, self.cmdtoep, self.cmdtoep_subblock_idx_w_1, cmd_rdy, None)
        self.state_timestep(self.lidartoep, self.lidartoep_block_height)

"""
    def lidar_to_lidar_state(self):
        block_width = self.lidar_1_idx.size(1)
        for lidar_col in range(0, self.lidar_index_size):
            for k in range(0, block_width):
                t0_col, t0_time = self.lidar_1_idx[0,k]
                self.lidar_1_state[0, k] = self.lidar[self.lidar_delta + lidar_col + t0_col, t0_time+0]
            self.lidar_state[(self.win_radius+1)*lidar_col:(self.win_radius+1)*lidar_col+1, block_width*lidar_col:(block_width+1)*lidar_col] = self.lidar_1_state

        for t0 in range(1, self.win_radius+1):
            for lidar_col in range(0, self.lidar_index_size):
                blockcol=0
                for i in range(-self.lidar_delta, self.lidar_delta+1):
                    self.lidar_1_state[0, blockcol] = numpy.dot(self.lidar_state[t0-1+(self.win_radius+1)*lidar_col, block_width*lidar_col+blockcol-1:block_width*lidar_col+blockcol-1])
                    for j in range(1, self.win_radius):
                        self.lidar_1_state[0, blockcol+j] = self.lidar_state[t0-1+(self.win_radius+1)*lidar_col, block_width*lidar_col+blockcol+j-1]
                    blockcol += self.win_radius

                for k in range(0, block_width):
                    
            self.lidar_state[t0+(self.win_radius+1)*lidar_col:t0+(self.win_radius+1)*lidar_col+1, block_width*lidar_col:(block_width+1)*lidar_col] = self.lidar_1_state
"""

    def control(self, cmd_speed, cmd_angle, lidar_meas):
        lidarrdy = lidar_meas[self.lidar_extindex_range] - self.lidar0
        cmdrdy = cmd_angle - np.array([self.speed0,0])
        self.save_lidar_state(lidarrdy)
        self.save_cmd_state(cmdrdy)
        self.state_timestep(self.lidars_fut, 1)
        self.state_timestep(self.cmds_fut, 1)
        self.lidars_fut[0,:] = lidarrdy
        self.cmds_fut[0,:] = cmdrdy
        if (self.tick == 0):
            self.plan_lidar_trajectory()
        return self.cmds_fut[1,:]
        self.tick = (self.tick+1)%self.win_radius


if __name__ == "__main__":
    print("data loading main")
    armax = MyArmax()
    print("done loading")
    print(armax[0].size())
    block0 = armax[0]
    nb_blocks = len(armax)
    block_width = block0.size(1)
    block_height = block0.size(0)
    toeplitz_rows = [0 for i in range(0, nb_blocks)]
    for i in range(0,nb_blocks):
        print("block", i, ": begin")
        cols_before = torch.zeros((block_height, i*block_width))
        cols_after  = torch.zeros((block_height, (nb_blocks-i-1)*block_width))
        print("catting colons...")
        new_row = torch.cat((cols_before, armax[i], cols_after), dim=1)
        toeplitz_rows[i] = new_row
    print(f"Done with blocks, creating giant toeplitz of size ({block_height*nb_blocks},{block_width*nb_blocks})...")
    toeplitz = torch.cat(toeplitz_rows, dim=0)
    print("Done creating giant toeplitz")
    AT = torch.transpose(toeplitz,0,1)
    ATA_1 = torch.inverse(torch.matmul(AT, toeplitz))
    ATA_1AT = torch.matmul(ATA_1, AT)
    print("Done calculating final MATRIX. Final size: ", ATA_1AT.size())
    parameters = torch.matmul(ATA_1AT, armax.get_datavec())
    print(f"Done calculating final parameters. Size of parameters: {parameters.size()}")
    p_np = parameters.numpy()
    df = pd.DataFrame(p_np)
    df.to_csv("parameters.csv",index=False)
    
