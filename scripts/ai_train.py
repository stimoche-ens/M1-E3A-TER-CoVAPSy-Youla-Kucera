#!/usr/bin/env python3

import os
import argparse
import sys
import glob
import importlib.util
import inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import conf
from tqdm import tqdm
import mylib



# --- TRAINING ---
def my_train(dataset, ModelClass, verbose=False):
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
    model = ModelClass()
    model = torch.compile(model)
    output_fullname = str(conf.OUTPUT_DIR)+str(ModelClass.__name__)+"_weights.pth"
    if verbose:
        print("model name:", str(ModelClass.__name__))
        print("output full name:", str(output_fullname))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        threshold=0.005,       # <--- THE MAGIC NUMBER
        threshold_mode='rel'   # Relative improvement
    )
    criterion = nn.MSELoss() # Standard loss for regression (measurements)
    model.train()
    prev_loss = 1
    for epoch in range(50):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/50")
        total_loss = 0
        for past, fut_cmd, target in loop:
            optimizer.zero_grad(set_to_none=True) 
            prediction = model(past, fut_cmd, target)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            """
            # Measure the health of the gradients
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Gradient Flow: {total_norm:.6f}") # Should be between 0.1 and 5.0
            """
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f} | Loss improvement: {(prev_loss-avg_loss)/prev_loss*100:.4f}% | Current LR: {current_lr:.2e}")
        scheduler.step(avg_loss)
        print(f"New LR: {optimizer.param_groups[0]['lr']:.2e}")
        prev_loss = avg_loss
    mylib.make_model_raw_ready(dataset, model)
    torch.save(model.state_dict(), output_fullname)


def main():
    parser = argparse.ArgumentParser(
        description="AI trainer",
        epilog="Examples: ./ai_train.py -v --dataset=IndependentTrajectoryDataset --model=MyLSTM"
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
    file_pool = mylib.get_file_pool(args.files, verbose=args.verbose)
    if args.verbose:
        print(f"Dataset class:    {args.dataset}")
        print(f"Model class:    {args.model}")
        print(f"File Pool: {file_pool}")
    
    try:
        DatasetClass, ds_module = mylib.load_class_from_pool(args.dataset, file_pool, verbose=args.verbose)
        ModelClass, _ = mylib.load_class_from_pool(args.model, file_pool, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    torch.set_num_threads(8)

    print(f"Instantiating Dataset: {DatasetClass.__name__}")
    dataset = DatasetClass()
    print(f"Starting training with model: {ModelClass.__name__}")
    my_train(dataset, ModelClass, verbose=args.verbose)

if __name__ == '__main__':
    main()
