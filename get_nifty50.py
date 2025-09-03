from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token
import pandas as pd
import datetime as dt

def get_nifty50_data(
    date: dt.date = None,
    start_time: str = "09:15:00",
    end_time: str = "15:00:00",
    interval: str = "minute"
):
    """
    Fetch NIFTY 50 data for a specific trading day and time range.

    :param date: Date (default = today)
    :param start_time: Start time (default = 9:15 AM)
    :param end_time: End time (default = 3:00 PM)
    :param interval: Candle interval (default = minute)
    """
    kite = KiteConnect(api_key=KiteAPI)
    kite.set_access_token(Kite_Access_Token)

    # Load indices from saved file
    df_indices = pd.read_csv("indices.csv")
    nifty50_row = df_indices[df_indices["tradingsymbol"] == "NIFTY 50"]
    if nifty50_row.empty:
        raise Exception("❌ NIFTY 50 not found in indices.csv. Run get_instruments.py again.")
    nifty_token = int(nifty50_row.iloc[0]["instrument_token"])

    # Default = today
    if date is None:
        date = dt.datetime.now().date()

    # Build datetime range
    from_date = dt.datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
    to_date = dt.datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M:%S")

    print(f"✅ Fetching NIFTY 50 data for {date} from {start_time} to {end_time}")

    # Fetch historical data
    data = kite.historical_data(
        instrument_token=nifty_token,
        from_date=from_date,
        to_date=to_date,
        interval=interval
    )

    df = pd.DataFrame(data)
    df.to_csv("nifty50_data.csv", index=False)

    print(f"✅ Saved {len(df)} rows to nifty50_data.csv")
    return df


# ----------- Run Example ------------

if __name__ == "__main__":
    # Example: Today’s data from 9:15 to 3:00
    df = get_nifty50_data()

    # Example: Specific day (2025-09-02) from 9:15 to 3:00
    # df = get_nifty50_data(date=dt.date(2025, 9, 2))
