import torch.nn as nn

# hidden_size: 32-64
# num_layers: 1-2
# dropout: 0.1-0.2

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.15):
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
            nn.Linear(32, 2)       # P_revert, E_return, calculate C_score later
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]         # last timestep
        return self.fc(h)