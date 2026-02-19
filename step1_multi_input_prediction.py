import torch
import torch.nn as nn




X = torch.tensor([
    [2.0, 3.0],
    [4.0, 1.0],
    [1.0, 5.0]
])


W = torch.tensor([[2.0],
                  [3.0]])   # shape (2,1)

b = torch.tensor([1.0])



Y_pred = X @ W + b

print("Predictions:\n", Y_pred)
