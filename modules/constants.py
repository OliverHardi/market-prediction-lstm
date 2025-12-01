LOOKBACK = 60

START_PAD = 30
END_PAD = 15

HORIZON_MIN = 10
HORIZON_MAX = 45


# 10% of future returns fall below QUANTILE_LOW
QUANTILE_LOW = 0.1
# 50% of future returns fall below QUANTILE_MEAN
QUANTILE_MEAN = 0.5
# 90% of future returns fall below QUANTILE_HIGH
QUANTILE_HIGH = 0.9

REVERT_FRACTION = 0 * 0.01 # 0% above the mean

NUM_FEATURES = 9
BATCH_SIZE = 1
NUM_EPOCHS = 5

LEARNING_RATE = 1e-4 #possibly 1e-4

TICKERS = [
    'AAPL',
    'AMZN',
    'GOOG',
    'META',
    'MSFT',
    'NFLX',
    'NVDA',
    'ORCL',
    'TSLA',
]