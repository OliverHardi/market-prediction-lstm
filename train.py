import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import math
import numpy as np
import matplotlib.pyplot as plt

from modules import constants
from modules.generator import data_generator, StockDataset
from modules import lstm
# from modules.loss import quantile_loss

import matplotlib.pyplot as plt


num_lines = 1

plt.ion()
fig, ax = plt.subplots()
ax.axhline(0, linestyle='--', linewidth=1, color='gray')
histories = [[] for _ in range(num_lines)]
# line_labels = [f"Line {i+1}" for i in range(num_lines)]
line_labels = ['loss', 'pred'][:num_lines]
line_colors = ['tab:orange', 'tab:blue'][:num_lines]

lines = []
for label, color in zip(line_labels, line_colors):
    line, = ax.plot([], [], label=label, color=color)
    lines.append(line)

# ax.set_xlabel("Batch")
# ax.set_ylabel("Value")
# ax.legend()


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = lstm.LSTM(input_size=constants.NUM_FEATURES).to(device)
optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)

# loss_fn = nn.MSELoss()
# loss_fn = nn.L1Loss()
loss_fn = nn.BCEWithLogitsLoss()

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
# avg_ting2 = 0

# all_targets = []
# for _, y in train_dataloader:
#     all_targets.append(y)
# all_targets = torch.cat(all_targets)

# print("target mean:", all_targets.mean().item())
# print("target median:", all_targets.median().item())
# print("target std:", all_targets.std().item())
# print("percentage exactly zero (or |r|<1e-6):",
#       (all_targets.abs() < 1e-6).float().mean().item())

for epoch in range( constants.NUM_EPOCHS ):

    total_loss = 0
    total_batches = 0

    for X_batch, y_batch in train_dataloader:

        X_batch = X_batch.to(device)
        # y_batch = y_batch.to(device)
        y_batch = (y_batch > 0).float().to(device)


        optimizer.zero_grad()
        preds = model(X_batch).squeeze(-1)

        # normal loss     
        loss = loss_fn(preds, y_batch)

        #sign loss
        # sign_X = torch.sign(X_batch)
        # sign_y = torch.sign(y_batch)
        # alpha = 10.0
        # p_pos = torch.sigmoid(alpha * preds)
        # sign_true = (y_batch > 0).float()             # 1 if positive, 0 otherwise
        # sign_loss = torch.nn.functional.binary_cross_entropy(p_pos, sign_true)
        # sign_mismatch = (sign_X != sign_y).float()
        # sign_loss = sign_mismatch.mean()

        
        # lambda_sign = 0.25

        # loss = loss + lambda_sign * sign_loss

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
        avg_ting += loss.item()
        # avg_ting2 += preds[1].item()
        if(total_batches % 20 == 19):
            # histories[0].append(math.log(avg_ting/20))
            histories[0].append(avg_ting/20)
            lines[0].set_xdata(np.arange(len(histories[0])))
            lines[0].set_ydata(histories[0])
            avg_ting = 0

        #     histories[1].append(avg_ting2/20)
        #     lines[1].set_xdata(np.arange(len(histories[1])))
        #     lines[1].set_ydata(histories[1])
        #     avg_ting2 = 0

        
        # histories[0].append(loss.item())
        # lines[0].set_xdata(np.arange(len(histories[0])))
        # lines[0].set_ydata(histories[0])

        # histories[1].append(preds[1].item())
        # lines[1].set_xdata(np.arange(len(histories[1])))
        # lines[1].set_ydata(histories[1])

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

            preds_val = model(X_val).squeeze(-1)

            val_loss = loss_fn(preds_val, y_val)

            val_loss_total += val_loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss_total / val_batches
        
    print(f"Epoch {epoch+1} | Train Loss: {total_loss / total_batches:.5f} | Val Loss: {avg_val_loss:.5f}")

    # print(f"Epoch {epoch+1} | Avg Loss: {total_loss / total_batches:.5f}")
    torch.save(model.state_dict(), "state/model.pt")

