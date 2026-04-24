#!/usr/bin/env python3

import torch
import intel_extension_for_pytorch as ipex
import torch.nn as nn
import torch.optim as optim
 
# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
 
# Move the model to IPEX device
device = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
#model = model.to(device)
#model = ipex.optimize(model)


model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
model, optimizer = ipex.optimize(model, optimizer=optimizer)
model = torch.compile(model, backend="ipex")
criterion = nn.MSELoss()
 
# Generate some dummy data
inputs = torch.randn(32, 10)
labels = torch.randn(32, 1)
 
# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
