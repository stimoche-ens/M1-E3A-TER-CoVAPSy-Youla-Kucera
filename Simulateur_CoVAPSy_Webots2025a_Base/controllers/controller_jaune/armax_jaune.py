#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
import glob
import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../automatique/identif_dyn/scripts')))
    import conf
else:
    import conf


def load_trajectories(datafiles_path):
    sequences = []
    targets = []
    mymax=0
    
    print(datafiles_path)
    for file in glob.glob(datafiles_path):
        df = pd.read_csv(file, header=0)
        # Col 0: useless, 1-2: controls, 3-362: measures
        controls = df.iloc[:, 1:3].values.astype(np.float32)
        measures = df.iloc[:, 3:363].values.astype(np.float32)
        curr_max = np.max(measures)
        mymax=max(mymax,curr_max)
        sequences.append(torch.tensor(controls))
        targets.append(torch.tensor(measures))
    controls_padded = pad_2Dseq_start(sequences, 10, False)
    measures_padded = pad_2Dseq_start(targets, 10, True)
    return controls_padded, measures_padded

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


class MyLinPerturb:
    def init_cfg(self):
        self.win_radius = 5
        self.lidar_delta = 2 # number of adjacent lidar angles
        self.lidar_min = -90
        self.lidar_max = 90
        self.lidar_step = 10 # step between two lidar angles
        self.lidar_offset0 = 3+180
        self.lidar_maxstep = 30
        self.my_lambda=0.005

    def init_lidar_idx(self):
        self.lidar_index_range=np.array(range(self.lidar_offset0 + self.lidar_min, self.lidar_offset0 + self.lidar_max + self.lidar_step, self.lidar_step))
        self.lidar_index_size=len(self.lidar_index_range)
        self.lidar_extindex_range=np.array(range(self.lidar_offset0 + self.lidar_min - self.lidar_delta*self.lidar_step, self.lidar_offset0 + self.lidar_max + self.lidar_delta*self.lidar_step + self.lidar_step, self.lidar_step))
        self.lidar_extindex_size=len(self.lidar_extindex_range)


    def init_states(self): #lidar_rdy = lidar[self.lidar_extindex_range] - self.lidar0
        self.lidars_fut = np.zeros([self.win_radius+1,self.lidar_extindex_size])
        self.cmds_fut   = np.zeros([self.win_radius+1,2])
        self.lidartoep_block_height    = self.lidar_extindex_size#4*self.lidar_delta+1
        self.lidartoep_subblock_width_min  = (self.lidar_delta+1)
        self.lidartoep_subblock_width_max  = (2*self.lidar_delta+1)
        self.lidartoep_subblock_idx_w  = [0 for i in self.lidar_extindex_range]
        self.lidartoep_subblock_idx_w_1= [0 for i in self.lidar_extindex_range]
        self.cmdtoep_subblock_idx_w  = np.array([[u+2*i*(self.win_radius+1) for u in range(0, 2*(self.win_radius+1))] for i in range(self.lidar_extindex_size)])
        self.cmdtoep_subblock_idx_w_1= self.cmdtoep_subblock_idx_w[:,::(self.win_radius+1)]
        self.lidartoep_subblock_idx_h  = np.array([[i] for i in range(0, self.lidar_extindex_size)])
        self.lidarrdy_subblock_idx     = [0 for i in self.lidar_extindex_range]
        block_offset = 0
        for angle_i in range(0,self.lidar_extindex_size):
            size = min([self.lidartoep_subblock_width_min+self.lidar_extindex_size-1-angle_i,self.lidartoep_subblock_width_min+angle_i,self.lidartoep_subblock_width_max])
            self.lidarrdy_subblock_idx[angle_i] = np.array(range(0,size))+max([0,angle_i-self.lidar_delta])
            self.lidartoep_subblock_idx_w[angle_i] = np.array(range(0,size*self.win_radius))+block_offset
            self.lidartoep_subblock_idx_w_1[angle_i] = self.lidartoep_subblock_idx_w[angle_i][::self.win_radius]
            block_offset += size*self.win_radius
        #self.lidarrdy_subblock_idx = np.stack(self.lidarrdy_subblock_idx, axis=0)
        #self.lidartoep_subblock_idx_w = np.stack(self.lidartoep_subblock_idx_w, axis=0)
        #self.lidartoep_subblock_idx_w_1 = np.stack(self.lidartoep_subblock_idx_w_1, axis=0)
        self.cmdtoep = np.zeros((self.lidartoep_block_height*(self.win_radius+1), self.lidar_extindex_size*2*(self.win_radius+1))) # all Toeplitz blocks
        self.lidartoep_width = self.win_radius*(2*sum(range(self.lidar_delta+1, 2*self.lidar_delta+1)) + self.lidar_index_size*(2*self.lidar_delta+1))
        self.lidartoep = np.zeros((self.lidartoep_block_height*(self.win_radius+1), self.lidartoep_width)) # all Toeplitz blocks
        #print("self.cmdtoep_subblock_idx_w.shape",self.cmdtoep_subblock_idx_w)
        #print("self.lidartoep_subblock_idx_w.shape",self.lidartoep_subblock_idx_w)

    def init_training_data(self):
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../automatique/identif_dyn/scripts'))
        ctl, meas = load_trajectories(os.path.abspath(os.path.join(scripts_dir, conf.DATA_PATH)))
        self.ctl_flip = ctl.numpy()
        self.meas_flip = meas.numpy()
        self.meas_flip -= self.meas_flip[:,0:1,:]
        self.meas_flip = self.meas_flip[:,:,self.lidar_extindex_range-self.lidar_offset0]
        self.ctl_flip  -= [self.speed0,0]
        self.ctl_flip = np.flip(self.ctl_flip, axis=1)
        self.meas_flip = np.flip(self.meas_flip, axis=1)
        self.ctl_flip = self.ctl_flip[:,::5,:]
        self.meas_flip = self.meas_flip[:,::5,:]
        self.num_trajs = self.meas_flip.shape[0]
        self.traj_len = self.meas_flip.shape[1]
        print(f"traj_len: {self.traj_len}")
        self.subtrajs_per_traj = self.traj_len - (self.win_radius+1) + 1 # (self.win_radius+1) because nth order implies a0+a1+....+an (n+1 samples)

    def init_params(self):
        df = pd.read_csv('parameters.csv')
        params = df.to_numpy()
        self.params_lidar = params[:self.lidartoep.shape[1]]
        self.params_cmd = params[self.lidartoep.shape[1]:]
        self.params_lidar_inv = np.zeros([len(self.params_lidar)])
        self.params_cmd_inv = np.zeros([len(self.params_cmd)])
        """
        print("self.params_lidar_inv[self.lidartoep_subblock_idx_w].shape:", [len(self.params_lidar_inv[i]) for i in self.lidartoep_subblock_idx_w])
        print("self.params_lidar[self.lidartoep_subblock_idx_w].shape", [len(self.params_lidar[i]) for i in self.lidartoep_subblock_idx_w])
        print("self.params_cmd[self.cmdtoep_subblock_idx_w_1[:,1:2]].shape", self.params_cmd[self.cmdtoep_subblock_idx_w_1[:,1:2]].shape)
        print("###")
        print("self.params_cmd_inv[self.cmdtoep_subblock_idx_w].shape:", [len(self.params_cmd_inv[i]) for i in self.cmdtoep_subblock_idx_w])
        print("self.params_cmd[self.cmdtoep_subblock_idx_w].shape", self.params_cmd[self.cmdtoep_subblock_idx_w].shape)
        print("self.params_cmd[self.cmdtoep_subblock_idx_w_1[:,1:2]].shape", self.params_cmd[self.cmdtoep_subblock_idx_w_1[:,1:2]].shape)
        print("###")
        print("self.params_cmd_inv[self.cmdtoep_subblock_idx_w_1[:,1:2]].shape:", self.params_cmd_inv[self.cmdtoep_subblock_idx_w_1[:,1:2]].shape)
        print("self.params_cmd[self.cmdtoep_subblock_idx_w_1[:,1:2]].shape", self.params_cmd[self.cmdtoep_subblock_idx_w_1[:,1:2]].shape)
        """
        for i in range(len(self.lidartoep_subblock_idx_w)):
            self.params_lidar_inv[self.lidartoep_subblock_idx_w[i]] = np.reshape(-self.params_lidar[self.lidartoep_subblock_idx_w[i]]/self.params_cmd[self.cmdtoep_subblock_idx_w_1[i,1:2]], (-1), order='F')
            self.params_cmd_inv[self.cmdtoep_subblock_idx_w[i]]   = np.reshape(-self.params_cmd[self.cmdtoep_subblock_idx_w[i]]/self.params_cmd[self.cmdtoep_subblock_idx_w_1[i,1:2]], (-1), order='F')
            self.params_cmd_inv[self.cmdtoep_subblock_idx_w_1[i,1:2]] = 1/self.params_cmd[self.cmdtoep_subblock_idx_w_1[i,1:2]]

    def __init__(self, goal_speed, first_lidar, rebuild=False):
        self.init_cfg()
        self.speed0 = goal_speed # =1
        self.init_lidar_idx()
        self.lidar0 = self.filter_lidar(first_lidar)
        self.tick = 0
        if (self.lidar_min - self.lidar_step*self.lidar_delta < -180) or (self.lidar_max + self.lidar_step*self.lidar_delta > 179):
            print("Error: self.lidar_min and/or self.lidar_max, conjugated with self.lidar_step*self.lidar_delta go out of [-180, 179] bounds")
            return None
        self.init_states()
        self.init_training_data()
        if rebuild:
            self.train_params()
        self.init_params()

    def __len__(self):
        return self.lidar_index_size

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
            lidar_step_size = f(self.lidars_fut[step,:])
            future_lidar_target = self.lidars_fut[step,:] + lidar_step_size
            self.lidars_fut[step+1,:] = future_lidar_target
            self.update_state_instant(step+1, 0, self.cmdtoep, self.cmdtoep_subblock_idx_w_1, [self.speed0, self.lidars_fut[step+1,:]], [i for i in range(0, self.lidar_extindex_size)])
            cmdsinv_fut = [self.cmds_fut[step,0], future_lidar_target]
            self.update_state_instant(step+1, 1, self.cmdtoep, self.cmdtoep_subblock_idx_w_1, self.cmdsinv_fut, None)
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

    def control(self, cmd_speed, cmd_angle, lidar_meas):
        lidarrdy = lidar_meas[self.lidar_extindex_range] - self.lidar0
        cmdrdy = np.array([cmd_speed, cmd_angle]) - np.array([self.speed0,0])
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

    def get_training_toep(self):
        start_row = self.win_radius
        lidar_toeps = [np.zeros((self.lidar_extindex_size*self.subtrajs_per_traj,self.lidartoep_width)) for i in range(0, self.num_trajs)]
        cmd_toeps = [np.zeros((self.lidar_extindex_size*self.subtrajs_per_traj,(self.win_radius+1)*2*self.lidar_extindex_size)) for i in range(0, self.num_trajs)]
        currentstep_lidars = np.array((self.lidartoep_width))
        cmdarrdy_subblock_idx = [[0,1] for i in range(self.lidar_extindex_size)]
        print(self.lidartoep_width)
        for traj_idx in range(self.num_trajs):
            for subtraj in range(0, self.subtrajs_per_traj):
                currentstep_lidars = self.meas_flip[traj_idx,subtraj:start_row+subtraj,np.concatenate(self.lidarrdy_subblock_idx)].reshape((-1), order='F')
                currentstep_cmds   = self.ctl_flip[traj_idx, subtraj:start_row+subtraj+1,   np.concatenate(cmdarrdy_subblock_idx)].reshape((-1), order='F')
                for idx in range(len(self.lidartoep_subblock_idx_w)):
                    lidar_toeps[traj_idx][self.lidartoep_subblock_idx_h[idx]+subtraj*self.lidartoep_block_height,self.lidartoep_subblock_idx_w[idx]] = currentstep_lidars[self.lidartoep_subblock_idx_w[idx]]
                    cmd_toeps[traj_idx][self.lidartoep_subblock_idx_h[idx]+subtraj*self.lidartoep_block_height,self.cmdtoep_subblock_idx_w[idx]] = currentstep_cmds[self.cmdtoep_subblock_idx_w[idx]]
            print(f"traj {traj_idx+1}/{self.num_trajs} done")
        lidar_toep = np.concatenate(lidar_toeps, axis=0)
        lidar_toeps = 0
        cmd_toep = np.concatenate(cmd_toeps, axis=0)
        cmd_toeps = 0
        total_toep = np.concatenate((lidar_toep,cmd_toep),axis=1)
        lidar_toep = 0
        cmd_toep = 0
        return total_toep

    def get_datavec(self):
        start_row = self.win_radius
        datavecs = [0 for i in range(self.num_trajs)]
        for traj_idx in range(self.num_trajs):
            datavecs[traj_idx] = self.meas_flip[traj_idx, start_row:, :]

        datavec = np.concatenate(datavecs, axis=0).reshape((-1), order='C')
        print("datavec shape: ", datavec.shape)
        return datavec

    def train_params(self):
        print("Creating giant toeplitz of size")
        toeplitz = torch.from_numpy(self.get_training_toep())
        col_norms = torch.norm(toeplitz, dim=0)
        print("min col norm:", col_norms.min())
        print("zero columns:", torch.where(col_norms==0)[0])
        coverage = torch.sum(torch.abs(toeplitz), dim=0) > 0
        print("coverage ratio:", coverage.float().mean())
        print("missing columns:", torch.where(~coverage)[0])
        print("rank:", torch.linalg.matrix_rank(toeplitz))
        s = torch.linalg.svdvals(toeplitz)
        print("svdvals[-20:]:",s[-20:])
        print("Done creating giant toeplitz")
        AT = torch.transpose(toeplitz,0,1)
        print(f"Done transposing giant toeplitz. size: {AT.size()}")
        ATA = torch.matmul(AT, toeplitz)
        print(f"Done creating ATA. size: {ATA.size()}")
        print(ATA[9:16,9:16])
        print(ATA[:,10])
        ATA_lI = ATA + torch.diag(torch.ones(ATA.size(1)),0)*self.my_lambda
        ATA_1 = torch.inverse(ATA_lI)
        ATA_1AT = torch.matmul(ATA_1, AT)
        print("Done calculating final MATRIX. Final size: ", ATA_1AT.size())
        parameters = torch.matmul(ATA_1AT, torch.from_numpy(self.get_datavec()).to(ATA_1AT.dtype))
        print(f"Done calculating final parameters. Size of parameters: {parameters.size()}")
        p_np = parameters.numpy()
        df = pd.DataFrame(p_np)
        df.to_csv("parameters.csv",index=False)


if __name__ == "__main__":
    print("data loading main")
    scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../automatique/identif_dyn/scripts'))
    ctl, meas = load_trajectories(os.path.abspath(os.path.join(scripts_dir, conf.DATA_PATH)))
    armax = MyLinPerturb(1,meas[0,0,:].numpy(), rebuild=False)
    print("done loading")
