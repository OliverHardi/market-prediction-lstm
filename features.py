# the main preprocessing step, where raw data is turned into technical indicators and features for the model
# these features are also scaled and visualized at the end

import numpy as np
import pandas as pd
import numpy as np

import mplfinance as mpf

import talib
from sklearn.preprocessing import StandardScaler

TICKER = 'AAPL'

df = pd.read_csv(f'data/{TICKER}_1min_data.csv', parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

df['session'] = df.index.date

# mean reversion features:


# - log returns
# - EMA distance
# - VWAP distance


# - RSI
# - MACD


# - ATR
# - rolling standard deviation


# time of day sin/cos encoding

# extra stuff:
# EMA

# log returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# EMA distance
ema_list = []
for date, session_data in df.groupby('session'):
    close = session_data['close'].values
    ema = talib.EMA(close, timeperiod=20)
    ema_list.append(pd.Series(ema, index=session_data.index))

df['ema20'] = pd.concat(ema_list)
df['ema_dist'] = (df['close'] - df['ema20']) / df['ema20']

# VWAP distance
vwap_list = []

for date, session_data in df.groupby('session'):
    tp = (session_data['high'] + session_data['low'] + session_data['close']) / 3
    cum_pv = (tp * session_data['volume']).cumsum()
    cum_v = session_data['volume'].cumsum()
    vwap = cum_pv / cum_v
    vwap_list.append(vwap)

df['vwap'] = pd.concat(vwap_list)
df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']

# RSI
rsi_list = []

for date, session_data in df.groupby('session'):
    close = session_data['close'].values
    rsi = talib.RSI(close, timeperiod=14)
    rsi_list.append(pd.Series(rsi, index=session_data.index))

df['rsi14'] = pd.concat(rsi_list)

# MACD
macd_list = []

for date, session_data in df.groupby('session'):
    close = session_data['close'].values
    _, _, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_list.append(pd.Series(macd_hist, index=session_data.index))

df['macd_hist'] = pd.concat(macd_list)

# ATR
atr_list = []

for date, session_data in df.groupby('session'):
    high = session_data['high'].values
    low = session_data['low'].values
    close = session_data['close'].values
    atr = talib.ATR(high, low, close, timeperiod=14)
    atr_list.append(pd.Series(atr, index=session_data.index))

df['atr14'] = pd.concat(atr_list)

# rolling std
rolling_std_list = []
for date, session_data in df.groupby('session'):
    # min_periods=30 means the first 29 rows will be NaN
    std = session_data['close'].rolling(window=30, min_periods=30).std()
    rolling_std_list.append(std)

df['rolling_std'] = pd.concat(rolling_std_list)

sma_list = []
for date, session_data in df.groupby('session'):
    sma = session_data['close'].rolling(window=30, min_periods=30).mean()
    sma_list.append(sma)
df['sma30'] = pd.concat(sma_list)

# time of day sin/cos encoding
market_open = pd.to_datetime(df.index.date.astype(str) + ' 09:30')
market_close = pd.to_datetime(df.index.date.astype(str) + ' 16:00')

minutes_since_open = (df.index - market_open).total_seconds() / 60
session_length = (market_close - market_open).total_seconds() / 60

time_angle = 2 * np.pi * (minutes_since_open / session_length)

df['time_sin'] = np.sin(time_angle)
df['time_cos'] = np.cos(time_angle)


features_to_standardize = ['log_return', 'ema_dist', 'vwap_dist', 
                           'macd_hist', 'atr14', 'rolling_std']

scalers = {}

for feature in features_to_standardize:
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[feature].values.reshape(-1,1))
    # df[feature] = df[feature].fillna(0.0)
    scalers[feature] = scaler

df['rsi14'] = (df['rsi14'] - 50) / 50


# start_date = '2022-9-3'
# end_date   = '2025-9-3'
# df = df.loc[start_date:end_date]

# forward fill NaNs
df.ffill(inplace=True)
df.fillna(0.0, inplace=True)

adplt = [
    # mpf.make_addplot(df['time_sin'], panel=1, color='g', ylabel='Time Sin'),
    # mpf.make_addplot(df['time_cos'], panel=1, color='b', ylabel='Time Cos'),
    mpf.make_addplot(df['ema_dist'], panel=1, color='orange', ylabel='ema_dist', ylim=(-3, 3)),
    # mpf.make_addplot(df['vwap_dist'], panel=1, color='cyan', ylabel='vwap_dist'),       # -0.05 to 0.05
    # mpf.make_addplot(df['rsi14'], panel=1, color='purple', ylabel='RSI14'),             # 0 to 100 
    # mpf.make_addplot(df['macd_hist'], panel=1, color='magenta', ylabel='MACD Hist'),    # -1 to 1
    # mpf.make_addplot(df['atr14'], panel=1, color='brown', ylabel='ATR14'),
    mpf.make_addplot(df['rolling_std'], panel=1, color='gray', ylabel='Rolling Std', ylim=(-3, 3)),
]



mpf.plot(df,
   type='line',          # candle ohlc renko line pnf
   volume=False,
   style='yahoo',        # yahoo charles mike nightclouds
   ylabel='Price',
   ylabel_lower='Volume',
   figsize=(12, 6),
   addplot=adplt,
)

# write to new file with features, excluding, ema20, vwap
df.drop(columns=['ema20', 'vwap'], inplace=True)
df.drop(columns=['macd_hist', 'rolling_std'], inplace=True)
df.to_csv(f'data/{TICKER}_features.csv')
