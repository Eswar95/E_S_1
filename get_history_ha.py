# get_history_ha.py
from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token
import pandas as pd
import datetime as dt
import os
from typing import Optional, Tuple

# ========================= CONFIG =========================
# Choose the trading day (leave None for today)
CUSTOM_DATE: Optional[dt.date] = None  # e.g., dt.date(2025, 9, 3)

# ---- PICK ONE MODE: NIFTY or OPTION ----
# 1) Use NIFTY 50 index (uncomment this block)
# USE_NIFTY = True
# USE_OPTION = False

# 2) Use NIFTY Option (uncomment this block and fill details)
USE_NIFTY = False
USE_OPTION = True
OPTION_INSTRUMENT_TYPE = "CE"   # "CE" or "PE"
OPTION_STRIKE = 24800          # int/float
OPTION_EXPIRY: Optional[dt.date] = dt.date(2025, 9, 9)  # or None -> pick nearest expiry within next 14 days

# Output folder
DATA_DIR = "nifty_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Heikin Ashi / color config (kept in case you want tolerance later)
DOJI_EPS_PCT = 0.0002  # ~0.02% tolerance for doji classification (NOT used in current logic)
# ==========================================================


def to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    """Return timestamp as IST timezone-aware pandas.Timestamp."""
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
    # ensure we return a pandas.Timestamp with tzinfo set to IST
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)


def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects: columns time/open/high/low/close
    Adds: ha_open/ha_high/ha_low/ha_close + ha_color + candle_color
    This variant NEVER assigns "DOJI" — only "GREEN" or "RED".
    Ties (close == open) are treated as GREEN (non-bearish).
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "time", "open", "high", "low", "close",
                "ha_open", "ha_high", "ha_low", "ha_close",
                "ha_color", "candle_color"
            ]
        )

    df = df.sort_values("time").reset_index(drop=True).copy()

    # HA close
    df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    # HA open (iterative)
    ha_open = []
    # first HA open = (open0 + close0) / 2
    ha_open.append((df.loc[0, "open"] + df.loc[0, "close"]) / 2.0)
    for i in range(1, len(df)):
        prev_ha_open = ha_open[i - 1]
        prev_ha_close = df.loc[i - 1, "ha_close"]
        ha_open.append((prev_ha_open + prev_ha_close) / 2.0)
    df["ha_open"] = ha_open

    # HA high/low
    df["ha_high"] = df[["high", "ha_open", "ha_close"]].max(axis=1)
    df["ha_low"] = df[["low", "ha_open", "ha_close"]].min(axis=1)

    # Candle colors (no DOJI)
    # For normal candles: if close >= open -> GREEN, else RED
    df["candle_color"] = "GREEN"
    df.loc[df["close"] < df["open"], "candle_color"] = "RED"

    # For HA candles: if ha_close >= ha_open -> GREEN, else RED
    df["ha_color"] = "GREEN"
    df.loc[df["ha_close"] < df["ha_open"], "ha_color"] = "RED"

    cols = [
        "time", "open", "high", "low", "close",
        "ha_open", "ha_high", "ha_low", "ha_close",
        "ha_color", "candle_color"
    ]
    return df[cols]


def get_nifty_token(kite: KiteConnect) -> int:
    """Return instrument token for NIFTY 50 (tries cached CSV first)."""
    try:
        if os.path.exists("nifty50_index.csv"):
            df_idx = pd.read_csv("nifty50_index.csv")
            row = df_idx[df_idx["tradingsymbol"] == "NIFTY 50"]
            if not row.empty:
                return int(row.iloc[0]["instrument_token"])
    except Exception:
        # If the file exists but is malformed, fall back to instruments()
        pass

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
    expiry: Optional[dt.date]
) -> Tuple[int, pd.DataFrame]:
    """
    Find the NIFTY option token for given type/strike/expiry.
    If expiry is None, pick nearest expiry within next 14 days (or nearest future).
    Returns (token, df_row)
    """
    instruments = kite.instruments()
    df = pd.DataFrame(instruments)

    df_opt = df[(df["segment"] == "NFO-OPT") & (df["name"].str.upper() == "NIFTY")].copy()
    if df_opt.empty:
        raise Exception("❌ No NIFTY options found in instruments().")

    # normalize expiry column
    df_opt["expiry"] = pd.to_datetime(df_opt["expiry"]).dt.date

    if expiry is None:
        today = dt.date.today()
        two_weeks = today + dt.timedelta(days=14)
        df_window = df_opt[(df_opt["expiry"] >= today) & (df_opt["expiry"] <= two_weeks)]
        if df_window.empty:
            # If nothing in next 2 weeks, pick the nearest future expiry overall (>= today)
            future = df_opt[df_opt["expiry"] >= today]
            if future.empty:
                # fallback to minimum expiry in dataset
                nearest = df_opt["expiry"].min()
            else:
                nearest = future["expiry"].min()
        else:
            nearest = df_window["expiry"].min()
    else:
        nearest = expiry

    row = df_opt[
        (df_opt["instrument_type"] == option_type)
        & (df_opt["strike"].astype(float) == float(strike))
        & (df_opt["expiry"] == nearest)
    ]
    if row.empty:
        # Provide helpful candidates
        candidates = df_opt[df_opt["expiry"] == nearest].sort_values("strike").head(20)
        raise Exception(
            f"❌ Option not found: NIFTY {nearest} {strike} {option_type}.\n"
            f"Example nearest-expiry strikes:\n{candidates[['tradingsymbol','strike']].to_string(index=False)}"
        )

    token = int(row.iloc[0]["instrument_token"])
    return token, row.iloc[[0]]

# def get_option_token(
#     kite: KiteConnect,
#     option_type: str,  # "CE" or "PE"
#     strike: float,
#     expiry: dt.date | None
# ) -> tuple[int, pd.DataFrame]:
#     """
#     Robust NIFTY option lookup.

#     - Accepts expiry as None, dt.date, or string (YYYY-MM-DD).
#     - Normalizes instruments() output and searches for the best match.
#     - If exact match not found, raises Exception with helpful diagnostics:
#       nearest expiries available and sample strikes for the chosen expiry.
#     """
#     instruments = kite.instruments()
#     df = pd.DataFrame(instruments)

#     # Make sure we have the expected columns
#     for c in ["segment", "name", "tradingsymbol", "expiry", "strike", "instrument_type", "instrument_token"]:
#         if c not in df.columns:
#             df[c] = None

#     # Filter to options where name contains NIFTY (case-insensitive)
#     df_opt = df[df["segment"].notna() & df["segment"].str.contains("OPT", case=False, na=False)]
#     # Some feeds use 'name' or 'tradingsymbol' to indicate underlying — be flexible
#     df_opt = df_opt[
#         df_opt["name"].fillna("").str.upper().str.contains("NIFTY")
#         | df_opt["tradingsymbol"].fillna("").str.upper().str.contains("NIFTY")
#     ].copy()

#     if df_opt.empty:
#         # Provide broader hint: show some instruments snapshot to help debug
#         sample = df.head(20)[["tradingsymbol", "segment", "instrument_token", "expiry", "strike", "instrument_type"]]
#         raise Exception(
#             "❌ No NIFTY options found in instruments(). "
#             "This usually means instruments() returned no NFO-OPT rows containing 'NIFTY'.\n"
#             f"Sample first 20 instruments (for debugging):\n{sample.to_string(index=False)}"
#         )

#     # Normalize expiry -> date
#     df_opt["expiry"] = pd.to_datetime(df_opt["expiry"], errors="coerce").dt.date
#     # Normalize strike -> float
#     df_opt["strike"] = pd.to_numeric(df_opt["strike"], errors="coerce")
#     # Normalize instrument_type
#     df_opt["instrument_type"] = df_opt["instrument_type"].str.upper().fillna("")

#     # Resolve requested expiry
#     if expiry is None:
#         today = dt.date.today()
#         two_weeks = today + dt.timedelta(days=14)
#         window = df_opt[(df_opt["expiry"] >= today) & (df_opt["expiry"] <= two_weeks)]
#         if window.empty:
#             # fallback: nearest future expiry
#             future = df_opt[df_opt["expiry"] >= today]
#             if future.empty:
#                 nearest_expiry = df_opt["expiry"].min()
#             else:
#                 nearest_expiry = future["expiry"].min()
#         else:
#             nearest_expiry = window["expiry"].min()
#     else:
#         # allow expiry passed as string like "2025-09-04"
#         if isinstance(expiry, str):
#             try:
#                 nearest_expiry = pd.to_datetime(expiry).date()
#             except Exception:
#                 raise ValueError(f"expiry string could not be parsed: {expiry}")
#         elif isinstance(expiry, dt.datetime):
#             nearest_expiry = expiry.date()
#         else:
#             nearest_expiry = expiry

#     # Try exact match first
#     candidates = df_opt[
#         (df_opt["instrument_type"] == option_type.upper())
#         & (df_opt["strike"] == float(strike))
#         & (df_opt["expiry"] == nearest_expiry)
#     ]

#     if not candidates.empty:
#         token = int(candidates.iloc[0]["instrument_token"])
#         return token, candidates.iloc[[0]]

#     # If exact match not found, prepare diagnostics:
#     # 1) list available expiries (sorted)
#     expiries = sorted(df_opt["expiry"].dropna().unique())
#     # 2) list strikes available for nearest_expiry (if any)
#     strikes_for_expiry = df_opt[df_opt["expiry"] == nearest_expiry]["strike"].dropna().unique()
#     strikes_for_expiry = sorted(strikes_for_expiry) if len(strikes_for_expiry) else []

#     # 3) suggest closest strike(s) (if strikes exist overall)
#     all_strikes = df_opt["strike"].dropna().unique()
#     closest_strikes = []
#     if len(all_strikes):
#         absdiff = [(abs(s - float(strike)), s) for s in all_strikes]
#         absdiff.sort()
#         closest_strikes = [s for _, s in absdiff[:10]]

#     # Build helpful message
#     msg_lines = [
#         f"❌ Option not found: NIFTY {nearest_expiry} {strike} {option_type}.",
#         "",
#         "Available expiries (sample):",
#         ", ".join([str(e) for e in expiries[:10]]) if expiries else "  <none>",
#         "",
#         f"Available strikes for expiry {nearest_expiry} (sample up to 30):",
#         ", ".join(map(str, strikes_for_expiry[:30])) if strikes_for_expiry else "  <none>",
#         "",
#         "Closest strikes across all expiries (up to 10):",
#         ", ".join(map(str, closest_strikes)) if closest_strikes else "  <none>",
#         "",
#         "If you intended the nearest strike or nearest expiry, consider calling this function with expiry=None "
#         "or choose one of the strikes listed above. To debug the instruments() output, call debug_print_nifty_info(kite)."
#     ]
#     raise Exception("\n".join(msg_lines))

def fetch_1min_history(kite: KiteConnect, token: int, target_date: dt.date) -> pd.DataFrame:
    """
    Fetch minute data between 09:15 and 15:15 IST for the given date.
    Returns DataFrame with columns: time, open, high, low, close
    """
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
    start = dt.datetime.combine(target_date, dt.time(9, 15), tzinfo=IST)
    end = dt.datetime.combine(target_date, dt.time(15, 15), tzinfo=IST)

    data = kite.historical_data(
        instrument_token=token,
        from_date=start,
        to_date=end,
        interval="minute"
    )
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # Zerodha may return 'date' (mixed types) or 'time'
    if "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"]).apply(to_ist)
        df.drop(columns=["date"], inplace=True, errors="ignore")
    else:
        if "time" not in df.columns:
            raise ValueError("Historical data missing 'date'/'time' column.")
        df["time"] = pd.to_datetime(df["time"]).apply(to_ist)

    # Ensure numeric columns exist and are floats
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Historical data missing required column '{col}'.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN OHLC
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    return df[["time", "open", "high", "low", "close"]]


def main():
    try:
        kite = KiteConnect(api_key=KiteAPI)
        kite.set_access_token(Kite_Access_Token)
    except Exception as e:
        print("❌ Failed to initialize KiteConnect. Check API key and access token.")
        raise

    # Date
    target_date = CUSTOM_DATE if CUSTOM_DATE else dt.date.today()

    # Decide mode (based on the config you uncommented above)
    if 'USE_NIFTY' in globals() and USE_NIFTY:
        # -------- NIFTY 50 mode --------
        try:
            token = get_nifty_token(kite)
        except Exception as e:
            print(f"❌ Error finding NIFTY token: {e}")
            return

        df = fetch_1min_history(kite, token, target_date)
        if df.empty:
            print("⚠️ No NIFTY data for the selected date.")
            return

        df_ha = compute_heikin_ashi(df)
        base = os.path.join(DATA_DIR, f"{target_date}_nifty50_1min")
        combined_file = base + "_with_ha.csv"
        ha_only_file = base + "_ha.csv"

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

        try:
            token, opt_row = get_option_token(
                kite,
                option_type=OPTION_INSTRUMENT_TYPE,
                strike=OPTION_STRIKE,
                expiry=expiry
            )
        except Exception as e:
            print(f"❌ Error finding option token: {e}")
            return

        df = fetch_1min_history(kite, token, target_date)
        if df.empty:
            print("⚠️ No option data for the selected date.")
            return

        df_ha = compute_heikin_ashi(df)

        # Nice file name from tradingsymbol
        symbol = opt_row.iloc[0]["tradingsymbol"]
        base = os.path.join(DATA_DIR, f"{target_date}_{symbol}_1min")
        combined_file = base + "_with_ha.csv"
        ha_only_file = base + "_ha.csv"

        df_ha.to_csv(combined_file, index=False)
        df_ha[["time", "ha_open", "ha_high", "ha_low", "ha_close", "ha_color"]].to_csv(ha_only_file, index=False)

        print(f"✅ OPTION saved: {combined_file}")
        print(f"✅ OPTION HA-only saved: {ha_only_file}")
        print(df_ha.tail(5))


if __name__ == "__main__":
    main()