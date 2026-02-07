#!/usr/bin/env python3

import sys
import os
import torch
import torch.nn as nn
import conf
import argparse
import mylib

def export_onnx(pthfile, ModelClass, device="cpu", verbose=False):
    model = ModelClass()
    raw_state_dict = torch.load(pthfile, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    
    # 3. Export to ONNX
    # This translates PyTorch operations into universal C++ graph instructions
    basename_stem = os.path.splitext(os.path.basename(pthfile))[0]
    output_fullname = os.path.join(str(conf.OUTPUT_DIR),str(basename_stem)+".onnx")

    if hasattr(ModelClass, 'get_onnx_metadata'):
        #device = next(model.parameters()).device
        meta = ModelClass.get_onnx_metadata(device=device)
        if ('input_dummies' not in meta):
            raise AttributeError(f"The ONNX_METADATA of model {ModelClass.__name__} must implement 'input_dummies' for export.")
        input_names = meta.get('input_names', ['default_in'])
        output_names = meta.get('output_names', ['default_out'])
        input_dummies = meta.get('input_dummies', (0,0))
        if ('input_names' not in meta):
            # --- DYNAMIC NAME GENERATION ---
            # Result example: "input_0_1x50x362"
            input_names = [ f"input_{i}_{'x'.join(map(str, t.shape))}" for i, t in enumerate(input_dummies) ]
            if verbose:
                print(f"Auto-detected Inputs: {input_names}")
        if ('output_names' not in meta):
            # 2. Run a dummy pass to detect Output shapes/count
            with torch.no_grad():
                dummy_output = model(*input_dummies)
            # Handle single tensor vs tuple of tensors
            outputs_list = dummy_output if isinstance(dummy_output, (list, tuple)) else (dummy_output,)
            output_names = [ f"output_{i}_{'x'.join(map(str, t.shape))}" for i, t in enumerate(outputs_list) ]
            if verbose:
                print(f"Auto-detected Outputs: {output_names}")
    else:
        raise AttributeError(f"Model {ModelClass.__name__} must implement ONNX_METADATA for export.")
        


    torch.onnx.export(
        model, 
        input_dummies, 
        output_fullname,
        export_params=True,
        opset_version=18,          # Version compatible with STM32Cube.AI
        do_constant_folding=True,  # Optimizes the graph for embedded
        input_names=input_names,
        output_names=output_names
    )
    print("Model exported to "+output_fullname+" successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Exporter of onnx",
        epilog="Examples: ./export_onnx.py -v --pthfile=MyLSTM_weights.pth --model=MyLSTM"
    )
    parser.add_argument(
        '-p', '--pthfile',      # Aliases
        type=str,            # Formal type enforcement
        help='Saved parameters of pytorch model',
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
        '-d', '--device',
        type=str,
        default='cpu',
        help='Target device for export (e.g., "cpu", "cuda", "cuda:0")'
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
        print(f"Pth file:    {args.pthfile}")
        print(f"Model class:    {args.model}")
        print(f"Target device:    {args.device}")
        print(f"File Pool: {file_pool}")
    
    try:
        ModelClass, _ = mylib.load_class_from_pool(args.model, file_pool, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Starting exporting model parameters: {ModelClass.__name__}")
    export_onnx(pthfile=args.pthfile, ModelClass=ModelClass, device=args.device, verbose=args.verbose)

if __name__ == '__main__':
    main()
