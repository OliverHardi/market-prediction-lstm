import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from modules import constants
from modules import lstm
from modules.generator import data_generator

from torch.utils.data import DataLoader
from modules.loss import quantile_loss
from modules.generator import StockDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = lstm.LSTM(input_size=constants.NUM_FEATURES, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load('state/model.pt', map_location=device))
model = model.to(device)
model.eval()


val_dataset = StockDataset("data/AAPL_val.csv")
val_loader = DataLoader(val_dataset, batch_size=None)

model.eval()
val_loss_total = 0
val_batches = 0
all_lower = []
all_median = []
all_upper = []
all_targets = []

with torch.no_grad():
    for X_val, y_val in val_loader:
        if len(X_val) == 0 or len(y_val) == 0:
            continue

        X_val = X_val.to(device)
        y_val = y_val.to(device)

        preds = model(X_val)

        lower_q = preds[:, 0].cpu().numpy().ravel()
        median_q = preds[:, 1].cpu().numpy().ravel()
        upper_q = preds[:, 2].cpu().numpy().ravel()

        target = y_val.cpu().numpy().ravel()

        all_lower.append(lower_q)
        all_median.append(median_q)
        all_upper.append(upper_q)
        all_targets.append(target)

all_lower = np.concatenate(all_lower)
all_median = np.concatenate(all_median)
all_upper = np.concatenate(all_upper)
all_targets = np.concatenate(all_targets)

plt.figure(figsize=(12,6))
plt.plot(all_targets, label="Actual", color="black", alpha=0.7)
plt.plot(all_median, label="Median Prediction", color="blue")
plt.fill_between(
    np.arange(len(all_targets)),
    all_lower,
    all_upper,
    color="orange",
    alpha=0.3,
    label="Predicted Quantile Range"
)
plt.xlabel("Sample Index")
plt.ylabel("Future % Change")
plt.title("Validation: Actual vs Predicted Quantiles")
plt.legend()
plt.show()

inside = np.logical_and(all_targets >= all_lower, all_targets <= all_upper)
coverage = np.mean(inside)
print(f"Fraction of targets inside predicted quantile range: {coverage*100:.2f}%")


outside_lower = np.mean(all_targets < all_lower)
outside_upper = np.mean(all_targets > all_upper)
print(f"Fraction below lower quantile: {outside_lower*100:.2f}%")
print(f"Fraction above upper quantile: {outside_upper*100:.2f}%")