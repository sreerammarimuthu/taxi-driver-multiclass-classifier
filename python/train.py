# train.py

import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import time
from model import TaxiDriverLSTM


class CustomDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.long)
  def __len__(self):
    return len(self.X)
  def __getitem__(self, idx):
    return {'X': self.X[idx], 'y': self.y[idx]}


def train_model(train_loader, val_loader, input_dim, hidden_dim, output_dim):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  class TaxiDriverLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
      super(TaxiDriverLSTM, self).__init__()
      self.hidden_dim = hidden_dim
      self.num_layers = num_layers
      self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
      h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
      c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
      out, _ = self.lstm(x, (h0, c0))
      out = self.fc(out[:, -1, :])
      return out

  model = TaxiDriverLSTM(input_dim, hidden_dim, output_dim).to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  best_loss = float('inf')
  for epoch in range(40):
    start_time = time.time()
    model.train()
    train_loss = 0.0

    for batch_idx, sample in enumerate(train_loader):
      inputs, labels = sample['X'].to(device), sample['y'].to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for batch_idx, sample in enumerate(val_loader):
        inputs, labels = sample['X'].to(device), sample['y'].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f'Epoch [{epoch + 1}/40], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f} sec'
    )

    if val_loss < best_loss:
      best_loss = val_loss
      # Saving the model in both .pth and .pt formats
      torch.save(model.state_dict(), 'best_model.pth')
      #torch.save(model, 'best_model.pt')
