# ANEC Evaluator

This repository contains a tool for evaluating Accuracy under specified Number of Effective Concepts(ANEC), which is a novel metric for Concept Bottleneck Model (CBM), proposed in our work [VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance](https://arxiv.org/abs/2408.01432). We separate this tool from [original codebase](https://github.com/Trustworthy-ML-Lab/VLG-CBM) for easier use. The code is based on the [glm_saga](https://github.com/MadryLab/glm_saga) repository, and modified to fit our needs.

<div align="center">
<h2>
<a href="https://arxiv.org/abs/2408.01432">Paper</a> | <a href="https://github.com/trustworthy-ml-lab/vlg-cbm">Code</a> | <a href="https://lilywenglab.github.io/VLG-CBM/">Project Page</a>
</h2>
</div>

## Installation 

To install the ANEC Evaluator package, clone this repository and install using pip:

```bash
pip install -e .
```

## Usage
To run the ANEC Evaluator, first prepare the activation and label tensors for the training, validation, and test datasets.
The data should be arranged in a folder with the following files:
- `train_activations.pt`: Tensor of shape (N_train, D) containing the conceptactivations of the training dataset
- `train_labels.pt`: Tensor of shape (N_train,) containing the labels of the training dataset
- `test_activations.pt`: Tensor of shape (N_test, D) containing the conceptactivations of the test dataset
- `test_labels.pt`: Tensor of shape (N_test,) containing the labels of the test dataset
- `val_activations.pt`(optional): Tensor of shape (N_val, D) containing the concept activations of the validation dataset. If not provided, the validation set will be split from the training set.
- `val_labels.pt`(optional): Tensor of shape (N_val,) containing the labels of the validation dataset. If not provided, the validation set will be split from the training set.

Then, run the following command:

```bash
get_anec --load_path <path_to_your_data_folder> --output_dir <path_to_save_results>
```

### Command Line Arguments

#### Required Arguments:
- `--load_path`: Directory containing the activation and label tensors

#### SAGA Configuration:
- `--saga_step_size`: Step size for SAGA optimization (default: 0.1)
- `--saga_n_iters`: Number of iterations for SAGA optimization (default: 1000)
- `--device`: Device to run computations on (cuda/cpu) (default: 'cuda')
- `--lam`: Maximum lambda value for regularization. The algorithm will sweep lambda from `0.001 * lam` to `lam` (default: 0.1)
- `--batch_size`: Batch size for training (default: 128)
- `--val_size`: Fraction of training data to use for validation (default: 0.1)
- `--measure_level`: List of NECs to measure accuracy at (default: 5 10 15 20 25 30)
- `--output_dir`: Directory to save outputs (default: 'output')

## Common Issues
### How to choose `--lam`
We recommend starting with the default value as the algorithm will automatically sweep a range of lambda to get target NECs. 
If the starting lambda is not appropriate, the algorithm will warn you and provide adjustment suggestions.
