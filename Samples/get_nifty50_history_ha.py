# get_nifty50_history_ha.py
from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token
import pandas as pd
import datetime as dt
import os

# ---------------- CONFIG ----------------
# 👉 Set a specific date or leave as None for today's date.
# Example: CUSTOM_DATE = dt.date(2025, 9, 3)
CUSTOM_DATE = dt.date(2025, 9, 3)

DATA_DIR = "nifty_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Heikin Ashi / color config
DOJI_EPS_PCT = 0.0002  # ~0.02% of price for doji tolerance
# ----------------------------------------


def to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    """Ensure timestamps are in Asia/Kolkata."""
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
    # If naive, attach IST; if aware, convert to IST
    if ts.tzinfo is None:
        return ts.replace(tzinfo=IST)
    return ts.astimezone(IST)


def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: ['time','open','high','low','close'] (time in pandas datetime)
    Returns df with HA columns + colors.
    """
    df = df.sort_values("time").reset_index(drop=True).copy()

    # HA close
    df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    # HA open (iterative)
    ha_open = [(df.loc[0, "open"] + df.loc[0, "close"]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + df.loc[i - 1, "ha_close"]) / 2.0)
    df["ha_open"] = ha_open

    # HA high/low
    df["ha_high"] = df[["high", "ha_open", "ha_close"]].max(axis=1)
    df["ha_low"]  = df[["low",  "ha_open", "ha_close"]].min(axis=1)

    # Colors with relative doji tolerance
    tol = df["close"].abs() * DOJI_EPS_PCT

    # Normal candle color
    body = (df["close"] - df["open"]).abs()
    df["candle_color"] = "GREEN"
    df.loc[df["close"] < df["open"], "candle_color"] = "RED"
    df.loc[body <= tol, "candle_color"] = "DOJI"

    # HA color
    ha_body = (df["ha_close"] - df["ha_open"]).abs()
    df["ha_color"] = "GREEN"
    df.loc[df["ha_close"] < df["ha_open"], "ha_color"] = "RED"
    df.loc[ha_body <= tol, "ha_color"] = "DOJI"

    # Order columns nicely
    cols = [
        "time", "open", "high", "low", "close",
        "ha_open", "ha_high", "ha_low", "ha_close",
        "ha_color", "candle_color"
    ]
    return df[cols]


def main():
    # Init Kite
    kite = KiteConnect(api_key=KiteAPI)
    kite.set_access_token(Kite_Access_Token)

    # Get NIFTY 50 instrument token
    df_idx = pd.read_csv("nifty50_index.csv")
    row = df_idx[df_idx["tradingsymbol"] == "NIFTY 50"]
    if row.empty:
        raise Exception("❌ NIFTY 50 not found in nifty50_index.csv")
    token = int(row.iloc[0]["instrument_token"])

    # Date & time window (IST)
    target_date = CUSTOM_DATE if CUSTOM_DATE else dt.date.today()
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
    from_time = dt.datetime.combine(target_date, dt.time(9, 15), tzinfo=IST)
    to_time   = dt.datetime.combine(target_date, dt.time(15, 15), tzinfo=IST)

    print(f"📅 Fetching NIFTY 50 1m candles for {target_date} (09:15 → 15:15 IST)")

    # Fetch historical 1-minute data
    data = kite.historical_data(
        instrument_token=token,
        from_date=from_time,
        to_date=to_time,
        interval="minute"
    )

    if not data:
        print("⚠️ No data returned for the selected date/time window.")
        return

    df = pd.DataFrame(data)

    # Normalize/rename columns to expected schema
    # Zerodha returns 'date' (ISO string) — convert to pandas datetime
    if "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"])
        df.drop(columns=["date"], inplace=True)
    else:
        # fallback if it's already named differently
        if "time" not in df.columns:
            raise ValueError("Expected a 'date' or 'time' column in historical data.")

    # Ensure IST timestamps
    df["time"] = df["time"].apply(to_ist)

    # Keep only needed columns
    df = df[["time", "open", "high", "low", "close"]]

    # Compute Heikin Ashi
    df_ha_all = compute_heikin_ashi(df)

    # Save outputs
    base = os.path.join(DATA_DIR, f"{target_date}_nifty50_1min")
    combined_file = base + "_with_ha.csv"
    ha_only_file  = base + "_ha.csv"

    df_ha_all.to_csv(combined_file, index=False)
    df_ha_all[["time", "ha_open", "ha_high", "ha_low", "ha_close", "ha_color"]].to_csv(ha_only_file, index=False)

    print(f"✅ Saved combined OHLC+HA -> {combined_file}")
    print(f"✅ Saved HA-only         -> {ha_only_file}")
    print(df_ha_all.tail(5))


if __name__ == "__main__":
    main()