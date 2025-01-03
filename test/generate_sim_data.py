import torch
import os
import numpy as np
from pathlib import Path
import subprocess

def generate_synthetic_data(num_samples=1000, num_concepts=100, num_classes=10, test_ratio=0.2):
    """Generate synthetic activation data and labels"""
    # Generate training data
    train_size = int(num_samples * (1 - test_ratio))
    test_size = num_samples - train_size
    
    # Generate sparse concept activations
    train_activations = torch.randn(train_size, num_concepts)
    test_activations = torch.randn(test_size, num_concepts)
    
    # Generate random weights for concept-to-class mapping
    true_weights = torch.zeros(num_concepts, num_classes)
    # Make it sparse - only 10% of concepts are actually relevant
    active_concepts = np.random.choice(num_concepts, size=num_concepts//10, replace=False)
    true_weights[active_concepts, :] = torch.randn(len(active_concepts), num_classes)
    
    # Generate labels
    train_logits = train_activations @ true_weights
    test_logits = test_activations @ true_weights
    train_labels = torch.argmax(train_logits, dim=1)
    test_labels = torch.argmax(test_logits, dim=1)
    
    return {
        'train_activations': train_activations,
        'train_labels': train_labels,
        'test_activations': test_activations,
        'test_labels': test_labels
    }

def main():
    # Create test directory
    test_dir = Path("test/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    data = generate_synthetic_data(
        num_samples=1000,
        num_concepts=100,
        num_classes=10
    )
    
    # Save data
    for name, tensor in data.items():
        torch.save(tensor, test_dir / f"{name}.pt")
    
    # Create results directory
    results_dir = Path("test/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run sparse_evaluation.py
    cmd = [
        "python", 
        "sparse_evaluation.py",
        "--load_path", str(test_dir),
        "--lam", "0.1",
        "--result_file", str(results_dir / "test_results.csv")
    ]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
