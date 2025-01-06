from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from glm_saga.elasticnet import glm_saga, IndexedTensorDataset
import pandas as pd
import os



@dataclass
class SAGAArguments:
    saga_step_size: float = 0.1
    saga_n_iters: int = 500
    device: str = 'cuda'
    max_lam: float = 0.01
    batch_size: int = 1024
    val_size: float = 0.1
    measure_level: tuple[int] = (5, 10, 15, 20, 25, 30)
    output_dir: str = 'output'


def weight_truncation(weight: torch.Tensor, sparsity: float):
    numel = weight.numel()
    num_zeros = int((1 - sparsity) * numel)
    threshold = torch.sort(weight.flatten().abs())[0][num_zeros]
    sparse_weight = weight.clone().detach()
    sparse_weight[weight.abs() < threshold] = 0
    return sparse_weight


def measure_acc(num_concepts, num_classes, num_samples, train_loader, val_loader, test_concept_loader,
                saga_step_size=0.1,
                saga_n_iters=500,
                device='cuda', 
                max_lam=0.01, measure_level=(5, 10, 15, 20, 25, 30)):
    linear = torch.nn.Linear(num_concepts, num_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = max_lam
    # Solve the GLM path
    max_sparsity = measure_level[-1] / num_concepts
    output_proj = glm_saga(linear, train_loader, saga_step_size, saga_n_iters, ALPHA, k=50, epsilon=0.005,
                    val_loader=val_loader, test_loader=test_concept_loader, do_zero=False, metadata=metadata, n_ex=num_samples, n_classes=num_classes,
                    max_sparsity=max_sparsity)
    path = output_proj['path']
    sparsity_list = [(params['weight'].abs() > 1e-5).float().mean().item() for params in path]

    # Measure accuracy on test set
    final_layer = torch.nn.Linear(num_concepts, num_classes)
    accs = []
    weights = []
    for eff_concept_num in measure_level:
        target_sparsity = eff_concept_num / num_concepts
        # Pick the lam with sparsity closest to target
        for i, sparsity in enumerate(sparsity_list):
            if sparsity >= target_sparsity: break
        params = path[i]
        W_g, b_g, lam = params['weight'], params['bias'], params['lam']
        print(eff_concept_num, lam, sparsity)
        print(f"Num of effective concept: {eff_concept_num}. Choose lambda={lam:.6f} with sparsity {sparsity:.4f}")
        W_g_trunc = weight_truncation(W_g, target_sparsity)
        weight_contribs = torch.sum(torch.abs(W_g_trunc), dim=0)
        print("Num concepts with outgoing weights:{}/{}".format(torch.sum(weight_contribs>1e-5), len(weight_contribs)))
        print(target_sparsity, (W_g_trunc.abs() > 0).sum())
        final_layer.load_state_dict({"weight":W_g_trunc, 'bias': b_g})
        final_layer = final_layer.to(device)
        weights.append((W_g_trunc, b_g))
        # Test final weights
        correct = []
        for x, y in test_concept_loader:
            x, y = x.to(device), y.to(device)
            pred = final_layer(x).argmax(dim=-1)
            correct.append(pred == y)
        correct = torch.cat(correct)
        accs.append(correct.float().mean().item())
        print(f"Test Acc: {correct.float().mean():.4f}")
    print(f"Average acc: {sum(accs) / len(accs):.4f}")
    return path, {NEC: weight for NEC, weight in zip(measure_level, weights)}, accs


def ANEC_test(saga_args: SAGAArguments, train_act, train_labels, test_act, test_labels, val_act=None, val_labels=None):
    # Load arguments
    n_concepts = train_act.shape[1]
    n_classes = train_labels.max() + 1
    n_samples = train_act.shape[0]
    
    # Create indexed dataset for train
    train_dataset = IndexedTensorDataset(train_act, train_labels)
    
    # Handle validation data
    if val_act is not None and val_labels is not None:
        # Use provided validation data
        val_dataset = IndexedTensorDataset(val_act, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=saga_args.batch_size, shuffle=True)
    else:
        # Split training data for validation
        train_size = int((1 - saga_args.val_size) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=saga_args.batch_size, shuffle=True)
    
    # Create regular dataset for test (no index needed)
    test_dataset = TensorDataset(test_act, test_labels)
    
    # Create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=saga_args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=saga_args.batch_size, shuffle=False)
    
    path, truncated_weights, accs = measure_acc(
                            n_concepts,
                            n_classes,
                            n_samples,
                            train_loader,
                            val_loader,
                            test_loader,
                            saga_step_size=saga_args.saga_step_size,
                            saga_n_iters=saga_args.saga_n_iters,
                            device=saga_args.device,
                            max_lam=saga_args.max_lam,
                            measure_level=saga_args.measure_level)
    
    sparsity_list = [(params['weight'].abs() > 1e-5).float().mean().item() for params in path]
    NEC = [n_concepts * sparsity for sparsity in sparsity_list]
    acc = [params['metrics']['acc_test'] for params in path]
    df = pd.DataFrame(data={'NEC': NEC, 'Accuracy': acc})
    if not os.path.exists(saga_args.output_dir):
        os.makedirs(saga_args.output_dir)
    df.to_csv(os.path.join(saga_args.output_dir, "metrics.csv"))
    # Save truncated weights
    for NEC in truncated_weights:
        W, b = truncated_weights[NEC]
        torch.save(W, os.path.join(saga_args.output_dir, f"W_g@NEC={NEC:d}.pt"))
        torch.save(b, os.path.join(saga_args.output_dir, f"b_g@NEC={NEC:d}.pt"))
    return accs

