import torch
import torch.nn as nn
import torch.optim as optim



X = torch.tensor([
    [2.0, 3.0],
    [4.0, 1.0],
    [1.0, 5.0],
    [3.0, 2.0]
])

Y = 2 * X[:, 0:1] + 3 * X[:, 1:2] + 1


# nn.Linear(in_features, out_features)

model = nn.Linear(2, 1)


# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Training Loop

epochs = 200

for epoch in range(epochs):

    # Forward pass
    Y_pred = model(X)

    # Compute loss
    loss = criterion(Y_pred, Y)

    # Clear gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Print learned parameters

print("\nLearned Weights:", model.weight.data)
print("Learned Bias:", model.bias.data)
