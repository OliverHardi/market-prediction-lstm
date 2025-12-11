import pandas as pd
import mplfinance as mpf


df = pd.read_csv('data/ORCL_1min_data.csv', parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

start_date = '2023-8-1'
end_date   = '2023-8-1'
df = df.loc[start_date:end_date]



mpf.plot(df, 
         type='candle',          # candle ohlc renko line pnf
         volume=False,
         style='yahoo',        # yahoo charles mike nightclouds
         title='Stock Price Over Time',
         ylabel='Price',
         ylabel_lower='Volume',
         figsize=(12, 6))
