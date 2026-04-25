#!/usr/bin/env python3

import argparse
import os
import sys

import torch

import conf
from libmy import libpool, libdata
from training_algorithms import criterion_MSELoss, optimizer_Adam, scheduler_BloatSimple
from training_loop import my_train


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI trainer",
        epilog="Examples: ./ai_train.py -v --dataset=UQYQ --model=MyLSTM",
    )
    parser.add_argument("-d", "--dataset", type=str, help="Class name of training dataset", required=True)
    parser.add_argument("-m", "--model", type=str, help="Class name of model to be trained", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-f", "--files", nargs="+", help="List of specific python files containing classes")
    parser.add_argument("--no-ipex", action="store_true", help="Skip Intel Extension for PyTorch optimization startup.")
    parser.add_argument("--output", type=str, default=None, help="Optional output .pth path.")
    parser.add_argument("--run-id", default=None, help="Run identifier used for default timestamped outputs.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no-set-current", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    file_pool = libpool.get_file_pool(args.files, verbose=args.verbose)
    if args.verbose:
        print(f"Dataset class: {args.dataset}")
        print(f"Model class: {args.model}")
        print(f"File Pool: {file_pool}")

    try:
        DatasetClass, _ = libpool.load_class_from_pool(args.dataset, file_pool, verbose=args.verbose)
        ModelClass, _ = libpool.load_class_from_pool(args.model, file_pool, verbose=args.verbose)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    torch.set_num_threads(8)

    print(f"Instantiating Dataset: {DatasetClass.__name__}")
    dataset = DatasetClass(ModelClass.IO_CONFIG)
    train_data, _ = libdata.my_train_val_split(dataset, 2 / 3)
    output_path = args.output or conf.default_weights_path(ModelClass.__name__, run_id=args.run_id)

    print(f"Starting training with model: {ModelClass.__name__}")
    saved_path = my_train(
        train_data,
        ModelClass,
        scheduler_BloatSimple,
        optimizer_Adam,
        criterion_MSELoss,
        verbose=args.verbose,
        use_ipex=not args.no_ipex,
        output_path=output_path,
        epochs=args.epochs,
    )

    if not args.no_set_current:
        conf.update_current_artifact(
            f"nntrain.{ModelClass.__name__}.weights",
            saved_path,
            kind="nn_weights",
            dataset=DatasetClass.__name__,
        )


if __name__ == "__main__":
    main()
