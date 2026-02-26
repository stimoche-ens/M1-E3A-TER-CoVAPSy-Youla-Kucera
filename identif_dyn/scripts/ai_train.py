#!/usr/bin/env python3

import os
import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex

import conf
from tqdm import tqdm
from libmy import libpool, libdata

def train_step(model, optimizer, criterion, inputs, outputs):
    def closure():
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*inputs, *outputs)
        loss = criterion(prediction, *outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return loss.item()
    
    loss_val = optimizer.step(closure)
    return loss_val

def scheduler_ReduceLROnPlateau(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        threshold=0.005,    
        threshold_mode='rel' # Relative improvement
    )

class AccumulatingOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, accumulation_steps=1):
        self.optimizer = optimizer
        self.acc_steps = accumulation_steps
        self.step_count = 0
        self.param_groups = optimizer.param_groups
        self.defaults = optimizer.defaults
        self.state = optimizer.state

    def zero_grad(self, set_to_none=True):
        if self.step_count % self.acc_steps == 0:
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.step_count += 1
        if self.step_count % self.acc_steps == 0:
            self.optimizer.step()
        return loss

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

def optimizer_Adam_accumulate(model, lr):
    accumulation_steps=4
    base_opt = torch.optim.Adam(model.parameters(), lr=accumulation_steps*lr)
    return AccumulatingOptimizer(base_opt, accumulation_steps=accumulation_steps)

def optimizer_Adam(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)

def optimizer_LBFGS(model, lr):
    return torch.optim.LBFGS(
        model.parameters(), 
        lr=lr, 
        history_size=10, 
        max_iter=4,        # Limit internal iterations to cap overhead
        #line_search_fn="strong_wolfe" # NECESSARY for the "basin" stability you requested
    )

def criterion_MSELoss():
    return nn.MSELoss(reduction='sum') # Standard loss for regression (measurements)

def my_train(dataset, ModelClass, scheduler_fn, optimizer_fn, criterion_fn, verbose=False):
    libdata.norm_data_mean_stddev_len(dataset)
    device = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
    print(f"Training on: {device}")

    model = ModelClass(dataset_stats=dataset.stats)
    model = model.to(device)
    optimizer = optimizer_fn(model=model, lr=1e-4)
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    model = torch.compile(model, backend="ipex")
    scheduler = scheduler_fn(optimizer)
    criterion = criterion_fn()
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    output_fullname = str(conf.OUTPUT_DIR)+str(ModelClass.__name__)+"_weights.pth"
    if verbose:
        print("model name:", str(ModelClass.__name__))
        print("output full name:", str(output_fullname))
    model.train()
    prev_loss = 1
    for epoch in range(50):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/50")
        total_loss = 0
        for traj_tuple in loop:
            inputs = [x.to(device, non_blocking=True) for x in traj_tuple["inputs"]]
            outputs = [x.to(device, non_blocking=True) for x in traj_tuple["outputs"]]
            loss_val = train_step(model=model, optimizer=optimizer, criterion=criterion,
                              inputs=inputs, outputs=outputs)
            total_loss += loss_val
            loop.set_postfix(loss=loss_val)
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f} | Loss improvement: {(prev_loss-avg_loss)/prev_loss*100:.4f}% | Current LR: {current_lr:.2e}")
        scheduler.step(avg_loss)
        print(f"New LR: {optimizer.param_groups[0]['lr']:.2e}")
        prev_loss = avg_loss
    model.denormalize_weights()
    torch.save(model.state_dict(), output_fullname)


def main():
    parser = argparse.ArgumentParser(
        description="AI trainer",
        epilog="Examples: ./ai_train.py -v --dataset=StraightTrack --model=MyLSTM"
    )
    parser.add_argument(
        '-d', '--dataset',      # Aliases
        type=str,            # Formal type enforcement
        help='Class name of training dataset',
        required=True,       # Critical constraint: fails if missing
        metavar='FILE'       # Placeholder name in help text
    )
    parser.add_argument(
        '-m', '--model',      # Aliases
        type=str,            # Formal type enforcement
        help='Class name of model to be trained',
        required=True,       # Critical constraint: fails if missing
        metavar='FILE'       # Placeholder name in help text
    )
    parser.add_argument(
        '-v', '--verbose',   # Aliases
        action='store_true', # Takes NO sub-arguments
        help='Enable verbose output'
    )
    parser.add_argument(
        '-f', '--files',
        nargs='+',           # Greedily consumes remaining args
        help='List of specific python files containing classes'
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(1)
    file_pool = libpool.get_file_pool(args.files, verbose=args.verbose)
    if args.verbose:
        print(f"Dataset class:    {args.dataset}")
        print(f"Model class:    {args.model}")
        print(f"File Pool: {file_pool}")
    
    try:
        DatasetClass, ds_module = libpool.load_class_from_pool(args.dataset, file_pool, verbose=args.verbose)
        ModelClass, _ = libpool.load_class_from_pool(args.model, file_pool, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    torch.set_num_threads(8)

    print(f"Instantiating Dataset: {DatasetClass.__name__}")
    dataset = DatasetClass(ModelClass.IO_CONFIG)
    print(f"Starting training with model: {ModelClass.__name__}")
    my_train(dataset, ModelClass, scheduler_ReduceLROnPlateau, optimizer_Adam, criterion_MSELoss, verbose=args.verbose)

if __name__ == '__main__':
    main()

