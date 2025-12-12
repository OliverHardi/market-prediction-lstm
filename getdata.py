import requests
import csv
from datetime import datetime, timedelta
import time


API_KEY = ""
SYMBOL = "NVDA"

url = f"https://api.twelvedata.com/time_series"

START_DATE = start_date = datetime.strptime("2020-03-25 09:30:00", "%Y-%m-%d %H:%M:%S")
NUM_REQUESTS = 400
# about 110 requests so we use a little extra

FILENAME = f"data/{SYMBOL}_1min_data.csv"

with open(FILENAME, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["datetime", "open", "high", "low", "close", "volume"])

    current_day = START_DATE.day

    for i in range(NUM_REQUESTS):

        final_day = current_day

        for j in range(12): # get last day in the thing
            final_day += 1
            # weekend check
            if (start_date + timedelta(days=final_day - current_day)).weekday() >= 5:
                final_day += 2

        # make sure it doesn't go into the future
        if (start_date + timedelta(days=final_day - START_DATE.day)) > datetime.now():
            print("reached current date, stopping")
            break

        start_time = (start_date + timedelta(days=current_day - START_DATE.day)).strftime('%Y-%m-%d %H:%M:%S')
        end_time = (start_date + timedelta(days=final_day - START_DATE.day) + timedelta(hours=6.5)).strftime('%Y-%m-%d %H:%M:%S')
        
        params = {
            "symbol": SYMBOL,
            "interval": "1min",
            "apikey": API_KEY,
            "start_date": start_time,
            "end_date": end_time,
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "values" in data:
                values = data["values"]
                # api returns newest first, reverse to oldest first
                for row in reversed(values):
                    writer.writerow([row["datetime"], row["open"], row["high"], row["low"], row["close"], row["volume"]])
                print(f"Written data from {start_time} to {end_time}")
            else:
                print("no data returned")
        else:
            print("error fetching data")
        
        current_day = final_day + 1

        # time.sleep(8) # 8 requests per minute limit
        for i in range(37):
            print('.', end='', flush=True)
            time.sleep(0.2)

