#!/usr/bin/env python3

import torch
from pool.StraightTrack import StraightTrack, load_trajectories
import conf
 
class StraightTrackResidual(StraightTrack):
    def __init__(self, io_config):
        self.io_cfg = io_config
        ctl, meas = load_trajectories(conf.DATA_PATH, clip_angle=self.io_cfg.get("clip_angle", False))
        meas = torch.clamp(meas, max=12000.0)
        speed_i = 0
        angle_i = 1
        
        nominal_prediction = torch.zeros_like(meas)
        residual = meas - nominal_prediction
        
        self.raw_data = {
            conf.CMD_SPEED: ctl[:,:,speed_i:speed_i+1],
            conf.CMD_ANGLE: ctl[:,:,angle_i:angle_i+1],
            conf.MES_LIDAR: meas,
            conf.RES_LIDAR: residual,
            conf.CMD_SPEED_Q: torch.zeros_like(ctl[:,:,speed_i:speed_i+1]),
            conf.CMD_ANGLE_Q: torch.zeros_like(ctl[:,:,angle_i:angle_i+1])
        }
        
        self.past_win = self.io_cfg["past_window"]
        self.fut_win  = self.io_cfg["future_window"]
        
        self.traj_len = meas.shape[1]
        self.subtrajs_per_traj = self.traj_len - (self.past_win + self.fut_win) + 1
        self.num_trajs = meas.shape[0]

        self.stats = {
            conf.CMD_SPEED: {"size": ctl[0,0,speed_i:speed_i+1].size(), "train_offset": None, "train_scale": None},
            conf.CMD_ANGLE: {"size": ctl[0,0,speed_i:speed_i+1].size(), "train_offset": None, "train_scale": None},
            conf.MES_LIDAR: {"size": meas[0,0,:].size(),                "train_offset": None, "train_scale": None},
            conf.RES_LIDAR: {"size": residual[0,0,:].size(),            "train_offset": None, "train_scale": None},
            conf.CMD_SPEED_Q: {"size": ctl[0,0,speed_i:speed_i+1].size(), "train_offset": None, "train_scale": None},
            conf.CMD_ANGLE_Q: {"size": ctl[0,0,speed_i:speed_i+1].size(), "train_offset": None, "train_scale": None},
        }