import torch.nn as nn
from modules import constants

# hidden_size: 32-64
# num_layers: 1-2
# dropout: 0.1-0.2

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=constants.HIDDEN_SIZE, num_layers=constants.NUM_LAYERS, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.05),      # 0.05 - 0.1 (optional)
            nn.Linear(32, constants.NUM_QUANTILES * 1)       # 3 quantiles: low, mean, high * predictions
            # nn.Linear(32, 1) # single output
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]         # last timestep
        return self.fc(h)