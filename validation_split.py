# splits the raw dataset into a training and validation set

import pandas as pd

csv_path = "data/AAPL_features.csv"
val_split = 0.15

df = pd.read_csv(csv_path, parse_dates=['datetime']).sort_values('datetime')
split_idx = int(len(df) * (1 - val_split))

df_train = df.iloc[:split_idx]
df_val   = df.iloc[split_idx:]

df_train.to_csv("data/AAPL_train.csv", index=False)
df_val.to_csv("data/AAPL_val.csv", index=False)