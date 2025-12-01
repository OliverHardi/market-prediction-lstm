import torch

import numpy as np
import pandas as pd
from modules import constants
from torch.utils.data import IterableDataset

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
            "vwap_dist"
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

        for i in range( len(chunk) - lookback - constants.HORIZON_MAX ):
            window_feat = window_values[i : i + lookback]
            ts_feat = ts_values[i + lookback - 1]
            ts_repeated = np.tile(ts_feat, (lookback, 1))
            combined = np.concatenate([window_feat, ts_repeated], axis=1) # X_window

            anchor_price = closes[i + lookback - 1]

            future_start = i + lookback + constants.HORIZON_MIN
            future_end = i + lookback + constants.HORIZON_MAX

            future_prices = closes[future_start : future_end]

            if len(future_prices) == 0:
                continue

            # p_revert -> did the price revert to the anchor price in the lookahead window?
            threshold = anchor_price * constants.REVERT_THRESHOLD

            diffs = np.abs(future_prices - anchor_price)
            reverted_anywhere = np.any(diffs <= threshold)

            p_revert = 1.0 if reverted_anywhere else 0.0

            # e_return -> max (log) return in lookahead window

            log_returns = np.log(future_prices / anchor_price)

            e_return = np.max(log_returns)

            #store
            X_list.append(combined)
            y_list.append([p_revert, e_return])

            if len(X_list) == batch_size:
                yield (
                    np.array(X_list, dtype=np.float32),
                    np.array(y_list, dtype=np.float32)
                )
                X_list = []
                y_list = []

        #yield remaining
        if len(X_list) > 0:
            yield (
                np.array(X_list, dtype=np.float32),
                np.array(y_list, dtype=np.float32)
            )