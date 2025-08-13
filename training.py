import model as md
import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
seq_length = 20
batch_size = 16

df = pd.read_csv("close_prices.csv")
close_prices = df["Close"].values

df = df.dropna(subset=["Close"])
close_prices = df["Close"].values

scaler = MinMaxScaler()
close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

X_seq = []
y_seq = []

for i in range(len(close_prices_scaled) - seq_length):
    X_seq.append(close_prices_scaled[i:i+seq_length])
    y_seq.append(close_prices_scaled[i+seq_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X = torch.tensor(X_seq[:, :, np.newaxis], dtype=torch.float32)
y = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

dataset = StockDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = md.LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print(np.isnan(close_prices).any()) 
print(np.isinf(close_prices).any())

epochs = 30
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "lstm_model.pth")

model2 = md.LSTMModel(input_size, hidden_size, num_layers, output_size)
model2.load_state_dict(torch.load("lstm_model.pth"))
model2.eval()


