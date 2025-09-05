# get_history_ha.py
from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token
import pandas as pd
import datetime as dt
import os

# ========================= CONFIG =========================
# Choose the trading day (leave None for today)
CUSTOM_DATE = None  # e.g., dt.date(2025, 9, 3)

# ---- PICK ONE MODE: NIFTY or OPTION ----
# 1) Use NIFTY 50 index (uncomment this block)
# USE_NIFTY = True
# USE_OPTION = False

# 2) Use NIFTY Option (uncomment this block and fill details)
USE_NIFTY = False
USE_OPTION = True
OPTION_INSTRUMENT_TYPE = "CE"   # "CE" or "PE"
OPTION_STRIKE = 24800           # int/float
OPTION_EXPIRY = dt.date(2025, 9, 4)            # dt.date(2025, 9, 4)  # or None -> pick nearest expiry within next 14 days

# Output folder
DATA_DIR = "nifty_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Heikin Ashi / color config
DOJI_EPS_PCT = 0.0002  # ~0.02% tolerance for doji classification
# ==========================================================


def to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=IST)
    return ts.astimezone(IST)


def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects: time/open/high/low/close
    Adds: ha_open/ha_high/ha_low/ha_close + ha_color + candle_color
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

    # Colors (relative tolerance)
    tol = df["close"].abs() * DOJI_EPS_PCT

    body = (df["close"] - df["open"]).abs()
    df["candle_color"] = "GREEN"
    df.loc[df["close"] < df["open"], "candle_color"] = "RED"
    df.loc[body <= tol, "candle_color"] = "DOJI"

    ha_body = (df["ha_close"] - df["ha_open"]).abs()
    df["ha_color"] = "GREEN"
    df.loc[df["ha_close"] < df["ha_open"], "ha_color"] = "RED"
    df.loc[ha_body <= tol, "ha_color"] = "DOJI"

    cols = [
        "time", "open", "high", "low", "close",
        "ha_open", "ha_high", "ha_low", "ha_close",
        "ha_color", "candle_color"
    ]
    return df[cols]


def get_nifty_token(kite: KiteConnect) -> int:
    # Prefer your saved file if present, else fallback to instruments()
    if os.path.exists("nifty50_index.csv"):
        df_idx = pd.read_csv("nifty50_index.csv")
        row = df_idx[df_idx["tradingsymbol"] == "NIFTY 50"]
        if not row.empty:
            return int(row.iloc[0]["instrument_token"])

    instruments = kite.instruments()
    df = pd.DataFrame(instruments)
    row = df[(df["segment"] == "INDICES") & (df["exchange"] == "NSE") & (df["tradingsymbol"] == "NIFTY 50")]
    if row.empty:
        raise Exception("❌ NIFTY 50 token not found.")
    return int(row.iloc[0]["instrument_token"])


def get_option_token(
    kite: KiteConnect,
    option_type: str,  # "CE" or "PE"
    strike: float,
    expiry: dt.date | None
) -> tuple[int, pd.DataFrame]:
    """
    Find the NIFTY option token for given type/strike/expiry.
    If expiry is None, pick nearest expiry within next 14 days.
    Returns (token, df_row)
    """
    instruments = kite.instruments()
    df = pd.DataFrame(instruments)

    df_opt = df[(df["segment"] == "NFO-OPT") & (df["name"] == "NIFTY")].copy()
    if df_opt.empty:
        raise Exception("❌ No NIFTY options found in instruments().")

    df_opt["expiry"] = pd.to_datetime(df_opt["expiry"]).dt.date
    if expiry is None:
        today = dt.date.today()
        two_weeks = today + dt.timedelta(days=14)
        df_window = df_opt[(df_opt["expiry"] >= today) & (df_opt["expiry"] <= two_weeks)]
        if df_window.empty:
            # If nothing in 2 weeks, pick the nearest future expiry overall
            nearest = df_opt["expiry"].min()
        else:
            nearest = df_window["expiry"].min()
    else:
        nearest = expiry

    row = df_opt[
        (df_opt["instrument_type"] == option_type)
        & (df_opt["strike"] == float(strike))
        & (df_opt["expiry"] == nearest)
    ]
    if row.empty:
        # Help the user see candidates around the strike/expiry
        candidates = df_opt[df_opt["expiry"] == nearest].sort_values("strike").head(10)
        raise Exception(
            f"❌ Option not found: NIFTY {nearest} {strike} {option_type}. "
            f"Example nearest-expiry strikes:\n{candidates[['tradingsymbol','strike']].to_string(index=False)}"
        )

    token = int(row.iloc[0]["instrument_token"])
    return token, row.iloc[[0]]


def fetch_1min_history(kite: KiteConnect, token: int, target_date: dt.date) -> pd.DataFrame:
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
    start = dt.datetime.combine(target_date, dt.time(9, 15), tzinfo=IST)
    end   = dt.datetime.combine(target_date, dt.time(15, 15), tzinfo=IST)

    data = kite.historical_data(
        instrument_token=token,
        from_date=start,
        to_date=end,
        interval="minute"
    )
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # Zerodha returns 'date'
    if "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"]).apply(to_ist)
        df.drop(columns=["date"], inplace=True)
    else:
        if "time" not in df.columns:
            raise ValueError("Historical data missing 'date'/'time' column.")
        df["time"] = pd.to_datetime(df["time"]).apply(to_ist)

    return df[["time", "open", "high", "low", "close"]]


def main():
    # Init Kite
    kite = KiteConnect(api_key=KiteAPI)
    kite.set_access_token(Kite_Access_Token)

    # Date
    target_date = CUSTOM_DATE if CUSTOM_DATE else dt.date.today()

    # Decide mode (based on the config you uncommented above)
    if 'USE_NIFTY' in globals() and USE_NIFTY:
        # -------- NIFTY 50 mode --------
        token = get_nifty_token(kite)
        df = fetch_1min_history(kite, token, target_date)
        if df.empty:
            print("⚠️ No NIFTY data for the selected date.")
            return

        df_ha = compute_heikin_ashi(df)
        base = os.path.join(DATA_DIR, f"{target_date}_nifty50_1min")
        combined_file = base + "_with_ha.csv"
        ha_only_file  = base + "_ha.csv"

        df_ha.to_csv(combined_file, index=False)
        df_ha[["time", "ha_open", "ha_high", "ha_low", "ha_close", "ha_color"]].to_csv(ha_only_file, index=False)

        print(f"✅ NIFTY saved: {combined_file}")
        print(f"✅ NIFTY HA-only saved: {ha_only_file}")
        print(df_ha.tail(5))

    else:
        # -------- OPTION mode --------
        if 'USE_OPTION' not in globals() or not USE_OPTION:
            raise Exception("Please set USE_NIFTY=True or USE_OPTION=True in CONFIG.")

        if 'OPTION_INSTRUMENT_TYPE' not in globals() or 'OPTION_STRIKE' not in globals():
            raise Exception("For options, set OPTION_INSTRUMENT_TYPE ('CE'/'PE') and OPTION_STRIKE.")

        expiry = None
        if 'OPTION_EXPIRY' in globals():
            expiry = OPTION_EXPIRY

        token, opt_row = get_option_token(
            kite,
            option_type=OPTION_INSTRUMENT_TYPE,
            strike=OPTION_STRIKE,
            expiry=expiry
        )

        df = fetch_1min_history(kite, token, target_date)
        if df.empty:
            print("⚠️ No option data for the selected date.")
            return

        df_ha = compute_heikin_ashi(df)

        # Nice file name from tradingsymbol
        symbol = opt_row.iloc[0]["tradingsymbol"]
        base = os.path.join(DATA_DIR, f"{target_date}_{symbol}_1min")
        combined_file = base + "_with_ha.csv"
        ha_only_file  = base + "_ha.csv"

        df_ha.to_csv(combined_file, index=False)
        df_ha[["time", "ha_open", "ha_high", "ha_low", "ha_close", "ha_color"]].to_csv(ha_only_file, index=False)

        print(f"✅ OPTION saved: {combined_file}")
        print(f"✅ OPTION HA-only saved: {ha_only_file}")
        print(df_ha.tail(5))


if __name__ == "__main__":
    main()