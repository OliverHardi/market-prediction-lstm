import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from modules import constants
from modules import lstm
from modules.generator import data_generator

from torch.utils.data import DataLoader

from modules.generator import StockDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = lstm.LSTM(input_size=constants.NUM_FEATURES, hidden_size=constants.HIDDEN_SIZE, num_layers=constants.NUM_LAYERS)
model.load_state_dict(torch.load('state/model.pt', map_location=device))
model = model.to(device)
model.eval()


val_dataset = StockDataset("data/AAPL_val.csv")
val_loader = DataLoader(val_dataset, batch_size=None)

model.eval()
val_loss_total = 0
val_batches = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for X_val, y_val in val_loader:
        if len(X_val) == 0 or len(y_val) == 0:
            continue

        X_val = X_val.to(device)
        y_val = y_val.to(device)

        preds = model(X_val).cpu().numpy()

        target = y_val.cpu().numpy().ravel()

        all_preds.append(preds)

        all_targets.append(target)

all_preds = np.concatenate(all_preds).ravel()
all_targets = np.concatenate(all_targets)

all_preds *= (1/500)

# pred_std = np.std(all_preds, ddof=1)
# actual_std = np.std(all_targets, ddof=1)

# print(f"Predicted std (sample): {pred_std:.6f}")
# print(f"Actual std (sample):    {actual_std:.6f}")

corr = np.corrcoef(all_preds, all_targets)[0, 1]
print(f"Correlation (pred vs actual): {corr:.6f}")

direction_acc = np.mean(np.sign(all_preds) == np.sign(all_targets))
print(f"Directional accuracy: {direction_acc:.4f}")

print(all_preds.min(), all_preds.max(), all_preds.std())


# scale_factor = actual_std / pred_std
# all_preds *= scale_factor

# center predicted values around 0
# pred_mean = np.mean(all_preds)
# all_preds -= pred_mean

plt.figure(figsize=(12,6))
plt.plot(all_targets, label="Actual", color="black", alpha=0.7)
plt.plot(all_preds, label="Median Prediction", color="green")

plt.plot([0, len(all_targets)], [0, 0], color='red', linestyle='--')

plt.xlabel("Sample Index")
plt.ylabel("Future % Change")
plt.title("Validation: Actual vs Predicted Quantiles")
plt.legend()
plt.show()
