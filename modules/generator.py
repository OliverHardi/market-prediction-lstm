# preprocesses data in chunks to create training batches for the model

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
            # "macd_hist",
            "atr14",
            # "rolling_std",
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

                anchor_price = closes[anchor_idx]             # anchor/entry price

                # domain of lookahead
                future_start = anchor_idx + constants.HORIZON_MIN
                future_end   = anchor_idx + constants.HORIZON_MAX
                # lookahead prices
                future_prices = closes[future_start:future_end]

                future_price = closes[anchor_idx + 30] # 30 minutes in the future


                c_val = (future_price - anchor_price) / anchor_price

                X_list.append(combined)
                y_list.append(c_val)


                if len(X_list) == batch_size:
                    yield (
                        np.array(X_list, dtype=np.float32),
                        np.array(y_list, dtype=np.float32)
                    )
                    X_list.clear()
                    y_list.clear()

        # yield extra/unused data at the end
        if len(X_list) > 0:
            yield (
                np.array(X_list, dtype=np.float32),
                np.array(y_list, dtype=np.float32)
            )