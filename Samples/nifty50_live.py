from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token
import pandas as pd
import datetime as dt
import time
import os

def collect_live_nifty50(interval_seconds=60):
    kite = KiteConnect(api_key=KiteAPI)
    kite.set_access_token(Kite_Access_Token)

    # Load indices to get NIFTY 50 token
    df_indices = pd.read_csv("indices.csv")
    nifty50_row = df_indices[df_indices["tradingsymbol"] == "NIFTY 50"]
    if nifty50_row.empty:
        raise Exception("❌ NIFTY 50 not found in indices.csv. Run get_instruments.py again.")
    nifty_token = int(nifty50_row.iloc[0]["instrument_token"])

    csv_file = "nifty50_live.csv"

    print("📡 Starting live data collection...")

    while True:
        now = dt.datetime.now()
        # Only run between 9:15 and 15:30 (Indian market hours)
        if now.time() >= dt.time(9, 15) and now.time() <= dt.time(15, 30):
            try:
                # Fetch last 5 minutes, take last candle
                data = kite.historical_data(
                    instrument_token=nifty_token,
                    from_date=now - dt.timedelta(minutes=5),
                    to_date=now,
                    interval="minute"
                )

                if data:
                    latest_candle = data[-1]  # last 1-minute candle
                    df = pd.DataFrame([latest_candle])

                    # Append to CSV
                    if os.path.exists(csv_file):
                        df.to_csv(csv_file, mode="a", header=False, index=False)
                    else:
                        df.to_csv(csv_file, index=False)

                    print(f"✅ {now.strftime('%H:%M:%S')} | Saved candle: {latest_candle}")

            except Exception as e:
                print(f"⚠️ Error fetching data: {e}")

            # Sleep until next minute
            time.sleep(interval_seconds)
        else:
            print("⏸️ Market closed. Waiting for next session...")
            time.sleep(300)  # sleep 5 minutes if market closed


if __name__ == "__main__":
    collect_live_nifty50()
