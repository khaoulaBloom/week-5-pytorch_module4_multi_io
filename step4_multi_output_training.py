import torch
import torch.nn as nn
import torch.optim as optim



X = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
])

Y1 = 2 * X[:, 0:1] + 3 * X[:, 1:2]
Y2 = 4 * X[:, 0:1] + 5 * X[:, 1:2]

Y = torch.cat((Y1, Y2), dim=1)


# Model

model = nn.Linear(2, 2)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training


epochs = 300

for epoch in range(epochs):

    Y_pred = model(X)

    loss = criterion(Y_pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


print("\nLearned Weight Matrix:\n", model.weight.data)
print("Learned Bias:\n", model.bias.data)
