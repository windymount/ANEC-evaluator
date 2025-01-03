from argparse import ArgumentParser
from .sparse_utils import ANEC_test, SAGAArguments
import pandas as pd
import os
import torch

def main():
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, help="Path to directory containing activations and labels")
    parser.add_argument("--lam", type=float, default=0.1, help="Maximum lambda value for regularization")
    parser.add_argument("--filter", type=float, default=0)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--result_file", type=str, default=None)

    args = parser.parse_args()

    # Load activations and labels
    train_act = torch.load(os.path.join(args.load_path, "train_activations.pt"))
    train_labels = torch.load(os.path.join(args.load_path, "train_labels.pt"))
    test_act = torch.load(os.path.join(args.load_path, "test_activations.pt"))
    test_labels = torch.load(os.path.join(args.load_path, "test_labels.pt"))

    # Create SAGA arguments
    saga_args = SAGAArguments(
        max_lam=args.lam,
        output_dir=args.load_path
    )

    accs = ANEC_test(saga_args, train_act, train_labels, test_act, test_labels)

    if args.result_file:
        if os.path.exists(args.result_file):
            df = pd.read_csv(args.result_file)
        else:
            df = pd.DataFrame(columns=['ACC@5', "AVGACC"])
        row = pd.Series({'ACC@5': accs[0], "AVGACC": sum(accs) / len(accs)})
        df.loc[len(df.index)] = row
        df.to_csv(args.result_file, index=False)

if __name__ == "__main__":
    main() 