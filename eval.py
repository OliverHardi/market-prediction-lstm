# evaluation of the performance/accuracy of the model over the validation set
# then testing a simple trading strategy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from modules import constants
from modules import lstm
from modules.generator import data_generator

from torch.utils.data import DataLoader

from modules.generator import StockDataset

import pandas as pd

TICKER = "AAPL"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = lstm.LSTM(input_size=constants.NUM_FEATURES, hidden_size=constants.HIDDEN_SIZE, num_layers=constants.NUM_LAYERS)
model.load_state_dict(torch.load('state/model.pt', map_location=device))
model = model.to(device)
model.eval()


val_dataset = StockDataset(f"data/{TICKER}_val.csv")
val_loader = DataLoader(val_dataset, batch_size=None)

df = pd.read_csv(f"data/{TICKER}_val.csv")
close_prices = df["close"].values

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

close_prices = close_prices[:len(all_preds)]

corr = np.corrcoef(all_preds, all_targets)[0, 1]
print(f"correlation (pred vs actual): {corr:.6f}")

direction_acc = np.mean(np.sign(all_preds) == np.sign(all_targets))
print(f"directional accuracy %: {direction_acc:.4f}")

print(f"mean: {all_preds.mean()}, std: {all_preds.std()}")


rsi_values = df["rsi14"].values[:len(all_preds)]

entry_threshold = 0.001
exit_threshold = 0.0005

stop_loss_pct = 0.02 # 2% stop loss

current_state = "neutral" 
trades = []
current_pos = None

for i, pred in enumerate(all_preds):
    current_rsi = rsi_values[i]
    current_price = close_prices[i]

    if current_state == "neutral":
        # long entry
        if pred > entry_threshold and current_rsi < 0.35:
            current_state = "long"
            current_pos = {'type': 'long', 'start': i, 'entry_price': current_price}
            print(f"entering LONG")
            
        # short entry
        elif pred < -entry_threshold and current_rsi > -0.35:
            current_state = "short"
            current_pos = {'type': 'short', 'start': i, 'entry_price': current_price}
            print(f"entering SHORT")

    elif current_state == "long":
        price_return = (current_price - current_pos['entry_price']) / current_pos['entry_price']
        # long exit
        if pred < exit_threshold or current_rsi > 0.6 or price_return <= -stop_loss_pct: 
            trades.append({'type': 'long', 'start': current_pos['start'], 'end': i})
            current_state = "neutral"
            current_pos = None
            print(f"exiting LONG, return: {price_return:.4%}")
            
    elif current_state == "short":
        price_return = (current_pos['entry_price'] - current_price) / current_pos['entry_price']
        # short exit
        if pred > -exit_threshold or current_rsi < -0.6 or price_return >= stop_loss_pct:
            trades.append({'type': 'short', 'start': current_pos['start'], 'end': i})
            current_state = "neutral"
            current_pos = None
            print(f"exiting SHORT, return: {price_return:.4%}")

# plotting
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(close_prices, label="AAPL Price", color="black", alpha=0.6, linewidth=1)

num_wins = 0;
pct_increase = 0;

for trade in trades:
    start_idx = trade['start']
    end_idx = trade['end']
    
    start_price = close_prices[start_idx]
    end_price = close_prices[end_idx]
    

    if trade['type'] == 'long':
        is_profit = end_price > start_price
        pct_increase += (end_price - start_price) / start_price
    else:  # short
        is_profit = end_price < start_price
        pct_increase += (start_price - end_price) / start_price
        
    color = 'green' if is_profit else 'red'

    num_wins += 1 if is_profit else 0
    
    ax.axvspan(start_idx, end_idx, color=color, alpha=0.3)
    
    marker = '^' if trade['type'] == 'long' else 'v'
    ax.scatter(start_idx, start_price, marker=marker, color='blue', s=40, zorder=5)
    ax.scatter(end_idx, end_price, marker='x', color='red', s=40, zorder=5)



print(f"Win Rate: {num_wins/len(trades):.2%}")
print(f"Total % change: {pct_increase:.2%}")

ax.set_title("trade results")
ax.set_ylabel("price in USD")
ax.set_xlabel("time")

plt.show()