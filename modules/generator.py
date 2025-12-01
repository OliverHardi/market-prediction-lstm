import torch

import numpy as np
import pandas as pd
from modules import constants
from torch.utils.data import IterableDataset

import matplotlib.pyplot as plt

class StockDataset(IterableDataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def __iter__(self):
        for X_batch, y_batch in data_generator(self.csv_path):
            yield (
                torch.tensor(X_batch, dtype=torch.float32),
                torch.tensor(y_batch, dtype=torch.float32)
            )


def data_generator(
        csv_path,
        lookback=constants.LOOKBACK,
        batch_size=constants.BATCH_SIZE,
        chunk_size=10000,
        

):
    prev_tail = None
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, parse_dates=['datetime']):

        chunk.sort_values('datetime', inplace=True)

        if prev_tail is not None:
            chunk = pd.concat([prev_tail, chunk], ignore_index=True)

        prev_tail = chunk.iloc[-lookback:].copy()

        # window features
        window_cols = [
            "log_return",
            "ema_dist",
            "vwap_dist",
        ]
        window_values = chunk[window_cols].values
        # timestamp features
        ts_cols = [
            "rsi14",
            "macd_hist",
            "atr14",
            "rolling_std",
            "time_sin",
            "time_cos"
        ]
        ts_values = chunk[ts_cols].values

        closes = chunk['close'].values

        X_list = []
        y_list = []

        for session_date, sess_df in chunk.groupby('session'):
            sess_idx = sess_df.index.to_numpy()

            start_i = sess_idx[0] + constants.START_PAD
            end_i = sess_idx[-1] - constants.END_PAD - (lookback + constants.HORIZON_MAX)
            if start_i >= end_i:
                continue

            for i in range(start_i, end_i):
                window_start = i
                window_end = i + lookback

                if chunk['session'].iloc[window_end - 1] != session_date:
                    continue

                window_feat = window_values[window_start : window_end]

                ts_feat = ts_values[window_end - 1]
                ts_rep = np.tile(ts_feat, (lookback, 1))
                combined = np.concatenate([window_feat, ts_rep], axis=1)

                # anchor = closes[window_end - 1]

                future_start = window_end + constants.HORIZON_MIN
                future_end   = window_end + constants.HORIZON_MAX

                if chunk['session'].iloc[future_end - 1] != session_date:
                    continue

                future_prices = closes[future_start:future_end]

                if len(future_prices) == 0:
                    continue

                # p_revert
                anchor_idx = i + lookback - 1

                P0 = closes[anchor_idx]             # anchor/entry price
                M0 = chunk["sma30"].iloc[anchor_idx]  # or vwap or ema

                # domain of lookahead
                future_start = anchor_idx + constants.HORIZON_MIN
                future_end   = anchor_idx + constants.HORIZON_MAX
                # lookahead prices
                future_prices = closes[future_start:future_end]

                # whether price goes up by x% from anchor or more (non mean reversion strategy)
                # reverted = np.any(
                #     future_prices > P0 * (1 + constants.REVERT_FRACTION)
                # )

                # whether entry starts below the mean and later rises above the mean by x%
                reversion_up = np.any(
                    future_prices > M0 * (1 + constants.REVERT_FRACTION)
                ) and P0 < M0
                
                

                # optional condition for short entries
                # whether entry starts above the mean and later falls below the mean by x%
                revesion_down = np.any(
                    future_prices < M0 * (1 - constants.REVERT_FRACTION)
                ) and P0 > M0

                reverted = reversion_up or revesion_down

                # e_return
                # highest/lowest log return in lookahead window depending on reversion direction
                log_returns = np.log(future_prices / P0)
                e_return = np.max(log_returns) if reversion_up else np.min(log_returns)
                
                # whether it reverts past the mean 
                p_revert = 1.0 if reverted else 0.0 # add and


                X_list.append(combined)
                y_list.append([p_revert, e_return])

                # ------------------- PLOTTING -------------------
                if True and len(X_list) == batch_size:
                    plt.figure(figsize=(10, 4))

                    # training window close prices
                    train_prices = closes[window_start:window_end]
                    plt.plot(range(lookback), train_prices, color='green', label='Training window')

                    # inbetween window close prices
                    inbetween_prices = closes[window_end:future_start]
                    plt.plot(range(lookback, lookback + len(inbetween_prices)), inbetween_prices, color='gray', label='Inbetween window')

                    # lookahead window close prices
                    plt.plot(range(lookback + len(inbetween_prices), lookback + len(inbetween_prices) + len(future_prices)), future_prices, color='red', label='Lookahead window')

                    ema_window = chunk["sma30"].values[window_start:window_end + len(future_prices)]
                    plt.plot(range(len(ema_window)), ema_window, color='orange', linestyle='--', label='EMA20')

                    # e_return overlay
                    target_price = P0 * np.exp(e_return)
                    plt.axhline(target_price, color='blue', linestyle='--', label='e_return target')

                    # p_revert marker at end of training window
                    if p_revert == 1:
                        plt.scatter(lookback-1, train_prices[-1], color='green', s=100, marker='o', label='p_revert=1')
                    else:
                        plt.scatter(lookback-1, train_prices[-1], color='magenta', s=100, marker='x', label='p_revert=0')

                    plt.xlabel("Time step")
                    plt.ylabel("Close price")
                    plt.title("Batch visualization")
                    plt.legend()
                    plt.show()
                    plt.pause(0.01)
                    plt.close()
                # --------------------------------------------------


                if len(X_list) == batch_size:
                    yield (
                        np.array(X_list, dtype=np.float32),
                        np.array(y_list, dtype=np.float32)
                    )
                    X_list.clear()
                    y_list.clear()

        # yield extra
        if len(X_list) > 0:
            yield (
                np.array(X_list, dtype=np.float32),
                np.array(y_list, dtype=np.float32)
            )