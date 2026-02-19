
Multi-Input / Multi-Output

This project walks through a complete progression from basic multi-input prediction to a full multi-input, multi-output training pipeline in PyTorch.

## What We Built

1. Multi-input single-output prediction (manual tensor math)
2. Multi-input single-output training (`nn.Linear(2, 1)`)
3. Multi-input multi-output prediction (manual tensor math)
4. Multi-input multi-output training (`nn.Linear(2, 2)`)
5. Final end-to-end pipeline with:
   - Custom `Dataset`
   - Train/validation/test split
   - `DataLoader`
   - Early stopping
   - Loss curves with Matplotlib
   - Learned parameter inspection

## Project Files

- `step1_multi_input_prediction.py`  
  Manual prediction using `Y = X @ W + b` for 2 inputs -> 1 output.

- `step2_multi_input_training.py`  
  Trains a linear model with `MSELoss` and `SGD` for 2 inputs -> 1 output.

- `step3_multi_output_prediction.py`  
  Manual prediction for 2 inputs -> 2 outputs using matrix multiplication.

- `step4_multi_output_training.py`  
  Trains a 2-output linear model with `MSELoss` and `Adam`.

- `final_multi_io_pipeline.py`  
  Full workflow using synthetic data, `Dataset`, `DataLoader`, train/val/test split, early stopping, and training-vs-validation loss plotting.

## Concepts Used

- PyTorch tensors
- Matrix multiplication (`@`)
- `nn.Linear`
- Forward pass
- Loss calculation with `nn.MSELoss`
- Backpropagation (`loss.backward()`)
- Optimizers (`SGD`, `Adam`)
- Custom `Dataset` and batching with `DataLoader`
- Model evaluation with `torch.no_grad()`
- Early stopping for regularization
- Train/validation/test workflow

## Synthetic Relationships in Final Pipeline

In `final_multi_io_pipeline.py`, targets are generated as:

- `y1 = 2*x1 + 3*x2 + 1`
- `y2 = 4*x1 + 5*x2 + 2`

The model learns these mappings from data.

## Requirements

This project uses Python 3.11+ and the following packages:

- `torch`
- `matplotlib`

Install them with:

```bash
pip install torch matplotlib
```

## How to Run

Run each step from the project root:

```bash
python step1_multi_input_prediction.py
python step2_multi_input_training.py
python step3_multi_output_prediction.py
python step4_multi_output_training.py
python final_multi_io_pipeline.py
```

## Expected Learning Outcome

After completing this project, you should be comfortable with:

- Building linear models with multiple input features
- Producing multiple outputs in a single model
- Training and validating PyTorch models
- Structuring a simple but realistic training pipeline
