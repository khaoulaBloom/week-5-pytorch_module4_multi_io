# Week 5 PyTorch: Multi-Input / Multi-Output

A compact PyTorch learning project that walks through:

1. Manual multi-input prediction
2. Multi-input model training
3. Manual multi-output prediction
4. Multi-output model training
5. A complete multi-input/multi-output training pipeline with train/val/test split and early stopping

## Project Structure

- `step1_multi_input_prediction.py`  
  Manual prediction with 2 input features and 1 output using matrix multiplication.

- `step2_multi_input_training.py`  
  Trains `nn.Linear(2, 1)` with MSE loss and SGD.

- `step3_multi_output_prediction.py`  
  Manual prediction with 2 input features and 2 outputs.

- `step4_multi_output_training.py`  
  Trains `nn.Linear(2, 2)` with MSE loss and Adam.

- `final_multi_io_pipeline.py`  
  End-to-end synthetic dataset workflow using:
  - custom `Dataset`
  - `DataLoader`
  - train/validation/test split
  - early stopping
  - loss plotting with Matplotlib

## Learning Objective

Understand how PyTorch linear models scale from:

- single-output regression (`Linear(2, 1)`) to
- multi-output regression (`Linear(2, 2)`),

while using standard training practices (batching, validation, early stopping).

## Requirements

- Python `3.11+`
- Packages:
  - `torch`
  - `matplotlib`

Install dependencies:

```bash
pip install torch matplotlib
```

## How To Run

From project root:

```bash
python step1_multi_input_prediction.py
python step2_multi_input_training.py
python step3_multi_output_prediction.py
python step4_multi_output_training.py
python final_multi_io_pipeline.py
```

## What Each Script Demonstrates

### 1) `step1_multi_input_prediction.py`

Uses a fixed formula:

- `y = x1*2 + x2*3 + 1`

implemented via tensor operations:

- `Y_pred = X @ W + b`

### 2) `step2_multi_input_training.py`

Learns the same relationship from data with:

- model: `nn.Linear(2, 1)`
- loss: `nn.MSELoss()`
- optimizer: `optim.SGD(lr=0.01)`

The learned weight should approach approximately `[2, 3]` and bias near `1`.

### 3) `step3_multi_output_prediction.py`

Computes 2 outputs at once:

- `Y_pred = X @ W + b` where `W` has shape `(2, 2)`

### 4) `step4_multi_output_training.py`

Learns two targets simultaneously:

- `y1 = 2*x1 + 3*x2`
- `y2 = 4*x1 + 5*x2`

with:

- model: `nn.Linear(2, 2)`
- loss: `nn.MSELoss()`
- optimizer: `optim.Adam(lr=0.01)`

### 5) `final_multi_io_pipeline.py`

Builds a full training pipeline on synthetic data:

- Inputs: 2 features
- Outputs:
  - `y1 = 2*x1 + 3*x2 + 1`
  - `y2 = 4*x1 + 5*x2 + 2`
- Split:
  - Train: 70%
  - Validation: 15%
  - Test: 15%
- Early stopping patience: `25`
- Visualization: training vs validation loss curves

At the end it prints:

- final test loss
- learned weight matrix
- learned bias vector

## Notes

- Results vary slightly run-to-run due to random initialization and random dataset sampling.
- `final_multi_io_pipeline.py` opens a Matplotlib plot window. In headless environments, configure a non-interactive backend or remove `plt.show()`.

## Suggested Next Steps

1. Add fixed random seeds for reproducible runs.
2. Save and reload model checkpoints (`torch.save` / `load_state_dict`).
3. Track metrics per output (separate loss for `y1` and `y2`).
4. Extend to a deeper network (`nn.Sequential`) and compare performance.
