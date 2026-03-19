# train the model off of the training dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

import numpy as np
import matplotlib.pyplot as plt

from modules import constants
from modules.generator import data_generator, StockDataset
from modules import lstm

import matplotlib.pyplot as plt


num_lines = 1

plt.ion()
fig, ax = plt.subplots()
ax.axhline(0, linestyle='--', linewidth=1, color='gray')
histories = [[] for _ in range(num_lines)]
line_labels = ['loss', 'pred'][:num_lines]
line_colors = ['tab:orange', 'tab:blue'][:num_lines]

lines = []
for label, color in zip(line_labels, line_colors):
    line, = ax.plot([], [], label=label, color=color)
    lines.append(line)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = lstm.LSTM(input_size=constants.NUM_FEATURES).to(device)
optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)

loss_fn = nn.MSELoss()

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

avg = 0


for epoch in range( constants.NUM_EPOCHS ):

    total_loss = 0
    total_batches = 0

    for X_batch, y_batch in train_dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)


        optimizer.zero_grad()
        preds = model(X_batch).squeeze(-1)

        # normal loss with temperature scaling  
        temperature = 0.3;
        mse_loss = loss_fn(preds / temperature, y_batch)
        
        loss = mse_loss;

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        total_loss += loss.item()
        total_batches += 1


        percent = (total_batches / total_train_batches) * 100
        print(f"Batch {total_batches}/{total_train_batches}  ({percent:.2f}%) | Loss: {loss.item():.5f}", end="\r")
        
        avg += loss.item()
        if(total_batches % 20 == 19):
            histories[0].append(avg/20)
            lines[0].set_xdata(np.arange(len(histories[0])))
            lines[0].set_ydata(histories[0])
            avg = 0


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

    #save model after each epoch
    torch.save(model.state_dict(), "state/model_beta.pt")

