# test.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def test_model(test_loader, input_dim, hidden_dim, output_dim):
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
  model.load_state_dict(torch.load('best_model.pth'))
  model.eval()

  correct = 0
  total = 0
  with torch.no_grad():
    for sample in test_loader:
      inputs, labels = sample['X'].to(device), sample['y'].to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = correct / total
  print(f'Test Accuracy: {accuracy:.4f}')
