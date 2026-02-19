import torch



X = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])


W = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

b = torch.tensor([1.0, 2.0])

Y_pred = X @ W + b

print("Predictions:\n", Y_pred)
