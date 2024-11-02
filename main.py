import torch
from torch import nn
from matplotlib import pyplot as plt
import torch.utils.data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import create_sequences
import numpy as np

scaler = MinMaxScaler()

device = 'cuda'

train_dataset = pd.read_csv('/datasets/DailyDelhiClimateTrain.csv')
test_dataset = pd.read_csv('/datasets/DailyDelhiClimateTest.csv') 

train_dataset = train_dataset.drop(columns=['date'])
test_dataset = test_dataset.drop(columns=['date'])

#scaling
train_scaled = scaler.fit_transform(train_dataset)
test_scaled = scaler.transform(test_dataset)

#conversion
train_tensor = torch.tensor(train_scaled, dtype=torch.float32, device=device)
test_tensor = torch.tensor(test_scaled, dtype=torch.float32, device=device)

#sequencing
seq_length = 5
train_sequences, train_labels = create_sequences(train_tensor, seq_length)
test_sequences, test_labels = create_sequences(test_tensor, seq_length)

#device change
train_sequences, train_labels =  train_sequences.to(device), train_labels.to(device)
test_sequences, test_labels = test_sequences.to(device), test_labels.to(device)

#batching
batch_size = 32
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_sequences, train_labels), batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_sequences, test_labels), batch_size=batch_size)

#GRU Model
class weatherGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
model = weatherGRU(4, 64, 4, 2).to(device)

loss_fn = nn.MSELoss()
optim = torch.optim.Adam(lr=0.001, params=model.parameters(), weight_decay=0.002)

epochs = 10

for epoch in range(epochs):
    model.train()
    current_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        outputs = model(X)
        loss = loss_fn(outputs, y)

        optim.zero_grad()

        loss.backward()

        optim.step()

        current_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {current_loss/len(train_loader):.4f}')

model.eval()
test_loss = 0.0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = loss_fn(outputs, y)
        test_loss += loss.item()

print(f'Test Loss: {test_loss/len(test_loader):.4f}')

y = y.cpu()
y = y.numpy()
y = y.flatten()
outputs = outputs.cpu()
outputs = outputs.numpy()
outputs = outputs.flatten()

plt.figure()
plt.plot(y, label='True Values')
plt.plot(outputs, label='Predictions')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'model.pth')
