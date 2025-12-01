import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import matplotlib.pyplot as plt

from modules import constants
from modules.generator import data_generator, StockDataset
from modules import lstm
# from modules import loss

import matplotlib.pyplot as plt

def get_loss(outputs, y):
    p_logits = outputs[:, 0]     # raw logits (NOT sigmoid)
    e_pred   = outputs[:, 1]

    # targets
    p_true = y[:, 0]             # p_revert 0/1
    e_true = y[:, 1]             # e_return

    # binary cross entropy (with logits = more stable)
    loss_p = torch.nn.functional.binary_cross_entropy_with_logits(
        p_logits, p_true
    )

    # mask: only count e_return when p_revert = 1
    mask = p_true   # shape (batch,), 1 where revert, 0 otherwise

    mse_raw = (e_pred - e_true)**2
    mse_masked = (mse_raw * mask).sum() / (mask.sum() + 1e-6) * 100.0

    loss = loss_p + mse_masked

    return loss, loss_p.item(), mse_masked.item()

# init plot
# plt.ion()
# fig, ax = plt.subplots()
# loss_p_history = []
# loss_e_history = []
# line_p, = ax.plot([], [], label='BCE Loss', color='tab:blue')
# line_e, = ax.plot([], [], label='MSE Loss', color='tab:orange')
# ax.set_xlabel("Batch")
# ax.set_ylabel("Loss")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

pos_weight = torch.tensor([0.05]).to(device)  # tune based on your data
bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()

model = lstm.LSTM(input_size=constants.NUM_FEATURES).to(device)
optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)

dataset = StockDataset("data/AAPL_features.csv")
dataloader = DataLoader(dataset, batch_size=None)

for epoch in range( constants.NUM_EPOCHS ):

    total_loss = 0
    total_batches = 0

    for X_batch, y_batch in dataloader:

        # print(f"y_batch: {y_batch}")

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # actual stuff

        optimizer.zero_grad()

        outputs = model(X_batch)
        p_logits = outputs[:, 0]
        e_pred = outputs[:, 1]

        p_true = y_batch[:, 0]
        e_true = y_batch[:, 1]

        # loss_p = bce_loss(p_logits, p_true)
        # loss_e = mse_loss(e_pred, e_true) * 100.0

        # loss = loss_p + loss_e

        loss, loss_p, loss_e = get_loss(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        
        # print(f"Batch {total_batches} | Loss: {loss.item():.5f}", end='\r')
        # loss_p_history.append(loss_p)
        # loss_e_history.append(loss_e)
        # line_p.set_xdata(np.arange(len(loss_p_history)))
        # line_e.set_xdata(np.arange(len(loss_p_history)))
        # line_p.set_ydata(loss_p_history)
        # line_e.set_ydata(loss_e_history)
        # ax.relim()
        # ax.autoscale_view()
        # plt.pause(0.001)

        preds = (torch.sigmoid(outputs[:, 0]) > 0.5).float()
        p_true_f = p_true.float()
        pred_fraction = preds.mean().item()
        accuracy = (preds == p_true_f).float().mean().item()
        p_counts = np.bincount(y_batch[:,0].cpu().numpy().astype(int))

        print(f"Prediction fraction: {pred_fraction:.4f} | Accuracy: {accuracy:.4f}", end='\r')


    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / total_batches:.5f}")
