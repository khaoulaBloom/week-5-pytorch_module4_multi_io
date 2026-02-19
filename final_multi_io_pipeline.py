import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split



class MultiIODataset(Dataset):
    """
    Synthetic dataset:
    
    y1 = 2x1 + 3x2 + 1
    y2 = 4x1 + 5x2 + 2
    """

    def __init__(self):
        
        # Generate 200 samples with 2 input features
        
        self.X = torch.rand(200, 2) * 10

        # True relationships
        
        y1 = 2 * self.X[:, 0:1] + 3 * self.X[:, 1:2] + 1
        y2 = 4 * self.X[:, 0:1] + 5 * self.X[:, 1:2] + 2

        # Combine outputs into one tensor (200,2)
        
        self.Y = torch.cat((y1, y2), dim=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]



class EarlyStopping:
    
    """
    Stops training if validation loss does not improve
    for a given number of epochs .
    """

    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True



class MultiIOModel(nn.Module):
    
    """
    Linear layer with:
    - 2 input features
    - 2 output features
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)



dataset = MultiIODataset()

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


model = MultiIOModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

early_stopping = EarlyStopping(patience=25)

epochs = 300
train_losses = []
val_losses = []



for epoch in range(epochs):


    model.train()
    train_loss = 0

    for X_batch, Y_batch in train_loader:

        # Forward pass
        
        Y_pred = model(X_batch)

        # Compute loss
        
        loss = criterion(Y_pred, Y_batch)

        # Backpropagation
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)


    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Early Stopping Check
    
    early_stopping(val_loss)

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break



model.eval()
test_loss = 0

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"\nFinal Test Loss: {test_loss:.4f}")


plt.figure()
plt.plot(train_losses)
plt.plot(val_losses)
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()



print("\nLearned Weight Matrix:\n", model.linear.weight.data)
print("\nLearned Bias Vector:\n", model.linear.bias.data)
