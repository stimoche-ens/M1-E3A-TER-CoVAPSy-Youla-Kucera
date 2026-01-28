#!/usr/bin/env python3


import torch
import torch.nn as nn

from MyLSTM import MyLSTM

# ==========================================
# MAIN EXECUTION
# ==========================================

# 2. Instantiate the "skeleton" of the model
model = MyLSTM()

# 3. Load the weights from the hard drive
# map_location='cpu' ensures it loads even if you don't have a GPU available
#model.load_state_dict(torch.load("MyLSTM_weights.pth", map_location='cpu'))
compiled_state_dict = torch.load("MyLSTM_weights.pth", map_location='cpu')
clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in compiled_state_dict.items()}
model.load_state_dict(clean_state_dict)
print("Weights loaded successfully!")

# 4. CRITICAL: Switch to Evaluation Mode
# This turns off dropout and batch normalization layers (if any)
model.eval()

print("Model loaded successfully. Ready for inference.")

# --- RUNNING LIVE PREDICTIONS ---
# Example: Stream in your 50 live past steps and 50 future planned steps
# Ensure they are torch tensors of float32
for i in range(0, 50):
    live_past = torch.randn(1, 50, 362)    # [Batch=1, Time=50, Features=362]
    planned_future = torch.randn(1, 50, 2) # [Batch=1, Time=50, Features=2]

    # Turn off gradient calculation to save memory and CPU
    with torch.no_grad():
        predicted_50_lidar = model(live_past, planned_future)

print("Predicted future LiDAR shape:", predicted_50_lidar.shape)
