#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
from armax2_jaune import MyLinPerturb

# Add the scripts directory to path so libmy and pool can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../automatique/identif_dyn/scripts')))
import conf
from pool.MyQParameter import MyQParameter

class YKControllerJaune:
    def __init__(self, goal_speed, first_lidar, q_weights_path):
        # 1. Initialize ARMAX nominal controller (K0)
        self.armax = MyLinPerturb(goal_speed, first_lidar, rebuild=False)
        
        # 2. Initialize Q-Parameter Neural Network
        self.device = torch.device("cpu") # Execute strictly on CPU for Webots compatibility
        
        # Note: dataset_stats must be provided if your NormAwareModule requires it for denorm.
        # Assuming stats are bypassed or baked-in during ONNX/JIT export. For raw PyTorch:
        dummy_stats = {
            conf.CMD_SPEED: {"size": (1,), "train_offset": torch.zeros(1), "train_scale": torch.ones(1)},
            conf.CMD_ANGLE: {"size": (1,), "train_offset": torch.zeros(1), "train_scale": torch.ones(1)},
            conf.CMD_SPEED_Q: {"size": (1,), "train_offset": torch.zeros(1), "train_scale": torch.ones(1)},
            conf.CMD_ANGLE_Q: {"size": (1,), "train_offset": torch.zeros(1), "train_scale": torch.ones(1)},
            conf.RES_LIDAR: {"size": (360,), "train_offset": torch.zeros(360), "train_scale": torch.ones(360)},
        }
        
        self.Q_net = MyQParameter(dataset_stats=dummy_stats)
        
        if os.path.exists(q_weights_path):
            self.Q_net.load_state_dict(torch.load(q_weights_path, map_location=self.device))
        else:
            print(f"[WARNING] Q-weights not found at {q_weights_path}. Operating with uninitialized Q.")
            
        self.Q_net.eval()
        
        # 3. Initialize state buffers for Q-network (past_window = 50)
        self.past_window = 50
        self.lidar_buffer = np.zeros((self.past_window, 360))
        self.speed_buffer = np.zeros((self.past_window, 1))
        self.angle_buffer = np.zeros((self.past_window, 1))

    def predict_nominal_plant(self, speed_hist, angle_hist):
        """
        CRITICAL ROADBLOCK:
        armax2_jaune.py identifies the inverse dynamics (K0: y -> u).
        Youla-Kucera requires the forward dynamics (P0: u -> y) to generate r = y - P0(u).
        Until P0 is identified, this returns zeros, effectively collapsing Q to a standard 
        disturbance observer rather than a mathematically rigorous YK parameter.
        """
        return np.zeros_like(self.lidar_buffer)

    def control(self, cmd_speed, cmd_angle, lidar_meas):
        # 1. Update State Buffers
        self.lidar_buffer = np.roll(self.lidar_buffer, -1, axis=0)
        self.speed_buffer = np.roll(self.speed_buffer, -1, axis=0)
        self.angle_buffer = np.roll(self.angle_buffer, -1, axis=0)
        
        self.lidar_buffer[-1, :] = lidar_meas
        self.speed_buffer[-1, 0] = cmd_speed
        self.angle_buffer[-1, 0] = cmd_angle
        
        # 2. Evaluate Nominal Controller (K0)
        # Note: armax returns a (speed, angle) tuple or array
        u_nom = self.armax.control(cmd_speed, cmd_angle, lidar_meas)
        
        # 3. Generate Coprime Residuals
        lidar_nom = self.predict_nominal_plant(self.speed_buffer, self.angle_buffer)
        residual_lidar = self.lidar_buffer - lidar_nom
        
        # 4. Evaluate Q-Parameter (requires tensor dimensions: [Batch, Time, Dim])
        t_speed = torch.tensor(self.speed_buffer, dtype=torch.float32)
        t_angle = torch.tensor(self.angle_buffer, dtype=torch.float32)
        t_resid = torch.tensor(residual_lidar, dtype=torch.float32)
        
        q_input = torch.cat([t_speed, t_angle, t_resid], dim=-1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            u_Q = self.Q_net(q_input) # Output shape: [1, future_window, 2]
            
        # Extract immediate control step from Q
        u_Q_step = u_Q[0, 0, :].numpy()
        
        # 5. Execute YK Superposition
        cmd_speed_final = u_nom[0] + u_Q_step[0]
        cmd_angle_final = u_nom[1] + u_Q_step[1]
        
        return cmd_speed_final, cmd_angle_final
    
