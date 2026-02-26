#!/usr/bin/env python3

import torch
import torch.nn as nn
import os
import conf
# Import your plugins using the pool namespace to be safe
from pool.MyLSTM import MyLSTM
from pool.NeuralPlanner import NeuralPlanner

class ClosedLoopAgent(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. The Brain (What we want to train)
        self.planner = NeuralPlanner()
        
        # 2. The Physics Engine (Pre-trained & Frozen)
        self.plant = MyLSTM()
        
        # Load weights for the plant
        weights_path = os.path.join(conf.OUTPUT_DIR, "MyLSTM_weights.pth")
        
        if os.path.exists(weights_path):
            print(f"Loading Physics Engine from {weights_path}")
            
            # --- FIX: Clean the keys before loading ---
            raw_state_dict = torch.load(weights_path, map_location='cpu')
            clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
            
            self.plant.load_state_dict(clean_state_dict)
        else:
            raise FileNotFoundError(f"Could not find physics weights at {weights_path}. Train MyLSTM first!")
            
        # FREEZE THE PLANT (We don't want to update physics, just control)
        for param in self.plant.parameters():
            param.requires_grad = False
        self.plant.eval()

    def forward(self, past_data, expert_future_cmd, target_path):
        """
        past_data: History [Batch, 50, 362]
        expert_future_cmd: What the human did (Ignored, or used for regularization)
        target_path: The actual path (LiDAR) we want to achieve
        """
        
        # 1. CONTROLLER ACTION
        # The planner sees the past and invents a plan
        generated_plan = self.planner(past_data)
        
        # 2. PHYSICS SIMULATION (Differentiable!)
        # We feed the *generated* plan into the *frozen* plant.
        # Gradients will flow from predicted_outcome -> plant -> generated_plan -> planner
        predicted_outcome = self.plant(past_data, generated_plan)
        
        return predicted_outcome

    def custom_patch_weights(self, dataset):
        """
        This is called by mylib at the end of training.
        We ONLY want to patch the Planner, because the Plant is already fixed.
        """
        print("🔧 Patching NeuralPlanner weights inside ClosedLoopAgent...")
        with torch.no_grad():
            # Patch Encoder (Inputs)
            enc_scale = dataset.enc_scale.to(self.planner.encoder.weight_ih_l0.device)
            self.planner.encoder.weight_ih_l0 /= enc_scale
            
            # Patch Output Head
            # The planner outputs COMMANDS. We trained on normalized commands (0..10).
            # We want raw commands (0..28).
            # Rule: Weight *= Scale
            
            # We need the scale factor for commands.
            # dataset.enc_scale[0] is Speed Scale, [1] is Angle Scale
            cmd_scale = dataset.enc_scale[0:2] # [Speed_Scale, Angle_Scale]
            
            # Since the output is flattened [Speed, Angle, Speed, Angle...], 
            # we repeat the scale vector 50 times
            full_scale = cmd_scale.repeat(conf.FUTURE_WINDOW).to(self.planner.plan_head[-1].weight.device)
            
            self.planner.plan_head[-1].weight *= full_scale
            self.planner.plan_head[-1].bias   *= full_scale
            
        print("✅ NeuralPlanner is now ready for RAW deployment.")


