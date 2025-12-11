import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import matplotlib.pyplot as plt

from modules import constants
from modules.generator import data_generator, StockDataset
from modules import lstm
from modules.loss import quantile_loss

import matplotlib.pyplot as plt


num_lines = 1

plt.ion()
fig, ax = plt.subplots()
ax.axhline(0, linestyle='--', linewidth=1, color='gray')
histories = [[] for _ in range(num_lines)]
line_labels = [f"Line {i+1}" for i in range(num_lines)]
line_colors = ['tab:orange', 'tab:blue'][:num_lines]

lines = []
for label, color in zip(line_labels, line_colors):
    line, = ax.plot([], [], label=label, color=color)
    lines.append(line)

ax.set_xlabel("Batch")
ax.set_ylabel("Value")
ax.legend()



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = lstm.LSTM(input_size=constants.NUM_FEATURES).to(device)
optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)

train_dataset = StockDataset("data/AAPL_train.csv")
train_dataloader = DataLoader(train_dataset, batch_size=None)

val_dataset = StockDataset("data/AAPL_val.csv")
val_dataloader = DataLoader(val_dataset, batch_size=None)

def count_batches(csv_path):
    count = 0
    for _ in data_generator(csv_path):
        count += 1
    return count

total_train_batches = count_batches("data/AAPL_train.csv")


val_x = []

avg_ting = 0

for epoch in range( constants.NUM_EPOCHS ):

    total_loss = 0
    total_batches = 0

    for X_batch, y_batch in train_dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)

        # loss
        # preds_min  = preds[:, 0:3]
        # preds_max  = preds[:, 3:6]
        preds_max = preds


        # target_min = y_batch[:, 0]
        # target_max = y_batch[:, 1]
        target_max = y_batch
        
        # loss = quantile_loss(preds_min, target_min) + quantile_loss(preds_max, target_max)
        loss = quantile_loss(preds_max, target_max)
        
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        total_batches += 1

        # spread = (preds[:,2] - preds[:,0]).mean()

        # print("loss: ", loss.item(), end='\r')
        # print("batch", total_batches, " | loss: ", f"{loss.item():.5f}", " | spread: ", f"{spread.item():.5f}", end='\r')
        percent = (total_batches / total_train_batches) * 100
        print(f"Batch {total_batches}/{total_train_batches}  ({percent:.2f}%)", end="\r")
        
        # print(f"Batch {total_batches} | Loss: {loss.item():.5f}", end='\r')

        if(total_batches % 20 == 19):
            histories[0].append(avg_ting/20)
            lines[0].set_xdata(np.arange(len(histories[0])))
            lines[0].set_ydata(histories[0])
            avg_ting = 0
        else:
            avg_ting += loss.item()


        # histories[1].append(preds_min[2, 1].item())
        # lines[1].set_xdata(np.arange(len(histories[1])))
        # lines[1].set_ydata(histories[1])

        # histories[2].append(preds_max[2, 1].item())
        # lines[2].set_xdata(np.arange(len(histories[2])))
        # lines[2].set_ydata(histories[2])

        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

    # evaluate
    model.eval()
    val_loss_total = 0
    val_batches = 0

    with torch.no_grad():
        for X_val, y_val in val_dataloader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            preds_val = model(X_val)
            preds_max_val = preds_val
            target_max_val = y_val

            val_loss = quantile_loss(preds_max_val, target_max_val)

            val_loss_total += val_loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss_total / val_batches
        
    print(f"Epoch {epoch+1} | Train Loss: {total_loss / total_batches:.5f} | Val Loss: {avg_val_loss:.5f}")

    # print(f"Epoch {epoch+1} | Avg Loss: {total_loss / total_batches:.5f}")
    torch.save(model.state_dict(), "state/model.pt")

