# hc_trade_helper.py
"""
Helper to turn a Heikin Ashi candle (from your daily HA CSV) into option entry suggestions.

What it does:
- Reads HA file for a target date (default = today)
- Picks the HA candle for a custom time (exact time or nearest earlier)
- Computes cal1 = ha_close - ha_open; uses PI = 22/7
- If cal1 <= 20: use form1 = cal1 * pi, else form2 = cal1 / pi
- Derives entry points from formulas described by you
- Finds ATM strike (nearest 100), generates candidate strikes (ATM +/- multiples of 100)
- Finds option instruments (nearest expiry relative to the target date) for those strikes
- Fetches option quotes and outputs paper-trade recommendations (entry, target, stop)
- DOES NOT place any real orders. Only suggestions printed & saved to CSV.

Assumptions / notes:
- Your HA CSV should be named like: nifty_data/YYYY-MM-DDminutes_heikin_ashi.csv
  (adjust INPUT_HA_FILE if your filename differs)
- Strike selection rounds spot to the nearest 100. Change logic if you want 50-step strikes etc.
- Targets/stops are in rupees per option premium (set TARGET_RUPEES / STOP_RUPEES)
- Quantity / LOT_SIZE are configurable
- Uses KiteConnect.quote and instruments() to resolve option tokens; ensure API limits considered.

Usage:
    python hc_trade_helper.py
"""

from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token
import pandas as pd
import datetime as dt
import os
import math
import sys

# ---------------- CONFIG ----------------
DATA_DIR = "nifty_data"

# file pattern: change if your filename is different
DATE_FOR_RUN = None  # e.g., dt.date(2025,9,4) or None for today
INPUT_HA_FILENAME_TEMPLATE = "{date}minutes_heikin_ashi.csv"  # placed in DATA_DIR
# Example produced earlier: 2025-09-04minutes_heikin_ashi.csv
# If your file name is different, change the template above.

# Time to check (string "HH:MM:SS") - change here to pick 09:15:00, 11:00:00, etc.
CUSTOM_TIME_STR = "09:15:00"

# Option selection params
STRIKE_ROUND = 100            # round spot to nearest 100 for ATM
STRIKE_OFFSETS = [-200, -100, 0, 100, 200]  # strikes to consider around ATM
EXPIRY_WINDOW_DAYS = 14      # pick nearest expiry >= target_date within this many days

# Paper trade sizing & rules
LOT_SIZE = 50                # contracts per lot (change to your lot size)
QTY_LOTS = 1                 # number of lots to trade (paper)
TARGET_RUPEES = 3.0          # profit target (Rs)
STOP_RUPEES = 1.0            # allowable loss (Rs)
PI = 22 / 7

# Output file
OUTPUT_SUGGESTIONS_CSV = os.path.join(DATA_DIR, "hc_trade_suggestions.csv")
# ---------------------------------------

kite = KiteConnect(api_key=KiteAPI)
kite.set_access_token(Kite_Access_Token)


def get_target_date():
    if DATE_FOR_RUN:
        return DATE_FOR_RUN
    return dt.date.today()


def build_ha_path(target_date):
    fname = INPUT_HA_FILENAME_TEMPLATE.format(date=target_date)
    return os.path.join(DATA_DIR, fname)


def load_ha_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"HA file not found: {path}")
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # required cols: time, ha_open, ha_high, ha_low, ha_close, ha_color (candle_color optional)
    for req in ["time", "ha_open", "ha_high", "ha_low", "ha_close", "ha_color"]:
        if req not in [c.lower() for c in df.columns]:
            # try to allow different casing
            # map lower-case columns
            lower_map = {c.lower(): c for c in df.columns}
            if req in lower_map:
                df = df.rename(columns={lower_map[req]: req})
            else:
                raise ValueError(f"Required column '{req}' not found in HA file. Columns: {df.columns.tolist()}")
    # ensure time is datetime
    df["time"] = pd.to_datetime(df["time"])
    # sort
    df = df.sort_values("time").reset_index(drop=True)
    return df


def pick_ha_row(df, time_str):
    # choose the exact time row if exists, else the last row <= time
    # time_str should be "HH:MM:SS"
    target_time = dt.datetime.strptime(time_str, "%H:%M:%S").time()
    # filter to same date
    # find rows where .time().strftime matches same date? we'll compare time only
    df["t_only"] = df["time"].dt.time
    eq = df[df["t_only"] == target_time]
    if not eq.empty:
        row = eq.iloc[-1]
        return row
    # else take last prior
    prior = df[df["t_only"] <= target_time]
    if prior.empty:
        # no prior rows (time too early); return first row
        return df.iloc[0]
    return prior.iloc[-1]


# --- option instrument helpers (uses instruments() table) ---
def instruments_df():
    inst = kite.instruments()
    return pd.DataFrame(inst)


def find_option_for_strike(target_date, option_type, strike, debug=False):
    """
    Find one option instrument (closest expiry >= target_date, same strike & type).
    Returns a pandas Series (row) or raises.
    """
    df = instruments_df()
    df_opt = df[(df["segment"] == "NFO-OPT") & (df["name"].str.upper() == "NIFTY")].copy()
    if df_opt.empty:
        raise RuntimeError("No NIFTY options in instruments() result")

    df_opt["expiry"] = pd.to_datetime(df_opt["expiry"]).dt.date
    df_opt["strike"] = df_opt["strike"].astype(float)
    # choose expiry relative to target_date
    # pick earliest expiry >= target_date if possible
    cand = df_opt[df_opt["expiry"] >= target_date]
    if cand.empty:
        # fallback to earliest available
        chosen_expiry = df_opt["expiry"].min()
    else:
        chosen_expiry = cand["expiry"].min()
    if debug:
        print("Chosen expiry:", chosen_expiry)
    pool = df_opt[(df_opt["expiry"] == chosen_expiry) & (df_opt["instrument_type"] == option_type)]
    if pool.empty:
        raise RuntimeError(f"No {option_type} options found for expiry {chosen_expiry}")
    # exact strike if present else nearest
    exact = pool[pool["strike"] == float(strike)]
    if not exact.empty:
        return exact.iloc[0]
    pool = pool.assign(dist=(pool["strike"] - float(strike)).abs()).sort_values("dist")
    row = pool.iloc[0]
    if debug:
        print("Using nearest strike:", row["strike"], row["tradingsymbol"])
    return row


def get_nifty_spot(token=None):
    """Fetch NIFTY spot price. If token not provided, try to get token from nifty50_index.csv"""
    if token is None:
        if os.path.exists("nifty50_index.csv"):
            df_idx = pd.read_csv("nifty50_index.csv")
            row = df_idx[df_idx["tradingsymbol"] == "NIFTY 50"]
            if not row.empty:
                token = int(row.iloc[0]["instrument_token"])
    if token is None:
        # fallback: fetch instruments and find
        inst = instruments_df()
        row = inst[(inst["segment"] == "INDICES") & (inst["exchange"] == "NSE") & (inst["tradingsymbol"] == "NIFTY 50")]
        if row.empty:
            raise RuntimeError("Cannot resolve NIFTY token to get spot")
        token = int(row.iloc[0]["instrument_token"])
    q = kite.quote([token])
    return float(q[str(token)]["last_price"])


def round_strike_to_bucket(spot, bucket=STRIKE_ROUND):
    """Round spot to nearest bucket (e.g., 100)"""
    return int(round(spot / bucket) * bucket)


def suggest_option_trades(ha_row, target_date):
    """
    Given a ha_row (Series with ha_open, ha_high, ha_low, ha_close, ha_color),
    compute entry formulas and suggest option strikes and entry/target/stop.
    """
    ha_open = float(ha_row["ha_open"])
    ha_high = float(ha_row["ha_high"])
    ha_low = float(ha_row["ha_low"])
    ha_close = float(ha_row["ha_close"])
    ha_color = str(ha_row.get("ha_color", "")).upper()

    cal1 = ha_close - ha_open
    print(f"HA at {ha_row['time']}: color={ha_color} cal1={cal1:.4f}")

    suggestions = []

    if cal1 <= 20:
        form1 = cal1 * PI
        entry_point1 = ha_close - cal1
        entry_point2 = ha_high - form1
        used_formula = "form1"
        computed = (entry_point1, entry_point2, form1)
    else:
        form2 = cal1 / PI
        entry_point3 = ha_close - form2
        entry_point4 = ha_high - form2
        used_formula = "form2"
        computed = (entry_point3, entry_point4, form2)

    # Determine side: if HA GREEN -> we look at CE buys (bullish), else HA RED -> PE buys (bearish)
    side = "BUY_CE" if ha_color == "GREEN" else "BUY_PE"

    # Find ATM
    spot = get_nifty_spot()
    atm = round_strike_to_bucket(spot, STRIKE_ROUND)
    candidate_strikes = [atm + off for off in STRIKE_OFFSETS]

    print(f"Spot {spot:.2f}, ATM ~ {atm}. Candidate strikes: {candidate_strikes}")

    # For each candidate strike, resolve option token and quote
    for strike in candidate_strikes:
        try:
            opt_row = find_option_for_strike(target_date, "CE" if ha_color == "GREEN" else "PE", strike)
        except Exception as e:
            print(f"Could not find option for strike {strike}: {e}")
            continue

        token = int(opt_row["instrument_token"])
        trad = opt_row["tradingsymbol"]

        # get quote
        try:
            q = kite.quote([token])[str(token)]
            last = q.get("last_price", None)
            ohlc = q.get("ohlc", {})
            open_p = ohlc.get("open")
            oi = q.get("oi")
            vol = q.get("volume")
        except Exception as e:
            print("Quote fetch failed for", token, e)
            last = None
            open_p = None
            oi = None
            vol = None

        # Decide entry price to use in suggestion:
        # We will use market last price if available. You could instead set entry_point derived earlier.
        entry_price = last if last is not None else (entry_point1 if cal1 <= 20 else entry_point3)

        # For paper trade: target and stop
        if entry_price is None:
            continue

        # target/stop directional: for buy target = +TARGET_RUPEES, stop = -STOP_RUPEES
        target_price = entry_price + TARGET_RUPEES
        stop_price = entry_price - STOP_RUPEES

        suggestion = {
            "time_checked": dt.datetime.now().isoformat(),
            "ha_time": ha_row["time"].isoformat(),
            "ha_color": ha_color,
            "cal1": cal1,
            "used_formula": used_formula,
            "computed_values": computed,
            "symbol": trad,
            "strike": strike,
            "option_token": token,
            "option_last": last,
            "option_open": open_p,
            "option_oi": oi,
            "option_vol": vol,
            "side": side,
            "entry_price": round(entry_price, 2),
            "target_price": round(target_price, 2),
            "stop_price": round(stop_price, 2),
            "qty_contracts": QTY_LOTS * LOT_SIZE,
        }
        suggestions.append(suggestion)

    return suggestions


def save_suggestions(suggestions):
    if not suggestions:
        print("No suggestions to save.")
        return
    df = pd.DataFrame(suggestions)
    write_header = not os.path.exists(OUTPUT_SUGGESTIONS_CSV)
    df.to_csv(OUTPUT_SUGGESTIONS_CSV, mode="a", header=write_header, index=False)
    print(f"Saved {len(suggestions)} suggestions -> {OUTPUT_SUGGESTIONS_CSV}")


def main():
    target_date = get_target_date()
    ha_path = build_ha_path(target_date)
    print("Loading HA file:", ha_path)
    df_ha = load_ha_df(ha_path)

    # pick HA row at custom time
    ha_row = pick_ha_row(df_ha, CUSTOM_TIME_STR)
    print("Selected HA row time:", ha_row["time"], "ha_color:", ha_row["ha_color"])

    # generate suggestions
    suggestions = suggest_option_trades(ha_row, target_date)
    # print nicely
    for s in suggestions:
        print("---- suggestion ----")
        print(f"{s['symbol']}: side={s['side']}, entry={s['entry_price']}, target={s['target_price']}, stop={s['stop_price']}, qty={s['qty_contracts']}")
        print(f"option last={s['option_last']}, oi={s['option_oi']}, vol={s['option_vol']}")
    # save to CSV
    save_suggestions(suggestions)

    


if __name__ == "__main__":
    main()

    
