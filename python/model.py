# model.py

import torch.nn as nn

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
