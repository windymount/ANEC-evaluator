from argparse import ArgumentParser
from .sparse_utils import ANEC_test, SAGAArguments
import pandas as pd
import os
import torch

def main():
    parser = ArgumentParser(description="ANEC Evaluator")
    # Data loading arguments
    parser.add_argument("--load_path", type=str, required=True,
                      help="Path to directory containing activations and labels")
    
    # SAGA arguments
    parser.add_argument("--saga_step_size", type=float, default=0.1,
                      help="Step size for SAGA optimization")
    parser.add_argument("--saga_n_iters", type=int, default=1000,
                      help="Number of iterations for SAGA optimization")
    parser.add_argument("--device", type=str, default='cuda',
                      help="Device to run computations on (cuda/cpu)")
    parser.add_argument("--lam", type=float, default=0.1,
                      help="Maximum lambda value for regularization")
    parser.add_argument("--batch_size", type=int, default=256,
                      help="Batch size for training")
    parser.add_argument("--val_size", type=float, default=0.1,
                      help="Fraction of training data to use for validation")
    parser.add_argument("--measure_level", type=int, nargs='+', 
                      default=[5, 10, 15, 20, 25, 30],
                      help="List of concept numbers to measure accuracy at")
    parser.add_argument("--output_dir", type=str, default='output',
                      help="Directory to save outputs")
    
    args = parser.parse_args()

    # Load activations and labels
    train_act = torch.load(os.path.join(args.load_path, "train_activations.pt"))
    train_labels = torch.load(os.path.join(args.load_path, "train_labels.pt"))
    test_act = torch.load(os.path.join(args.load_path, "test_activations.pt"))
    test_labels = torch.load(os.path.join(args.load_path, "test_labels.pt"))
    
    # Try to load validation data if available
    val_act = None
    val_labels = None
    try:
        val_act = torch.load(os.path.join(args.load_path, "val_activations.pt"))
        val_labels = torch.load(os.path.join(args.load_path, "val_labels.pt"))
        print("Using provided validation data")
    except FileNotFoundError:
        print(f"No validation data found in {args.load_path}, will split from training data")

    # Create SAGA arguments from command line args
    saga_args = SAGAArguments(
        saga_step_size=args.saga_step_size,
        saga_n_iters=args.saga_n_iters,
        device=args.device,
        max_lam=args.lam,
        batch_size=args.batch_size,
        val_size=args.val_size,
        measure_level=tuple(args.measure_level),
        output_dir=args.output_dir
    )

    accs = ANEC_test(saga_args, train_act, train_labels, test_act, test_labels, val_act, val_labels)

if __name__ == "__main__":
    main() 