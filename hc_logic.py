# hc_logic.py
import pandas as pd
import datetime as dt
import os

# ---- Config ----
DATA_DIR = "nifty_data"
DATE_STR = dt.date.today().strftime("%Y-%m-%d")
INPUT_MINUTE_FILE = os.path.join(DATA_DIR, f"{DATE_STR}minutes_timestamp.csv")
OUTPUT_HA_FILE   = os.path.join(DATA_DIR, f"{DATE_STR}minutes_heikin_ashi.csv")

ADD_NORMAL_CANDLE_COLOR = True   # also add regular candle GREEN/RED/DOJI
DOJI_EPS = 1e-6                  # tiny threshold to classify doji if needed


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert normal OHLC candles into Heikin Ashi candles.
    Input columns must include: ['time','open','high','low','close'].
    Returns a DataFrame with HA columns + color flags.
    """
    work = df.copy().reset_index(drop=True)

    # HA close
    work["ha_close"] = (work["open"] + work["high"] + work["low"] + work["close"]) / 4.0

    # HA open (iterative)
    ha_open = [(work.loc[0, "open"] + work.loc[0, "close"]) / 2.0]
    for i in range(1, len(work)):
        ha_open.append((ha_open[i-1] + work.loc[i-1, "ha_close"]) / 2.0)
    work["ha_open"] = ha_open

    # HA high/low
    work["ha_high"] = work[["high", "ha_open", "ha_close"]].max(axis=1)
    work["ha_low"]  = work[["low",  "ha_open", "ha_close"]].min(axis=1)

    # HA color
    def ha_color_row(row):
        if abs(row["ha_close"] - row["ha_open"]) <= DOJI_EPS:
            return "DOJI"
        return "GREEN" if row["ha_close"] > row["ha_open"] else "RED"

    work["ha_color"] = work.apply(ha_color_row, axis=1)

    # Optional: normal candle color too
    if ADD_NORMAL_CANDLE_COLOR:
        def norm_color_row(row):
            if abs(row["close"] - row["open"]) <= DOJI_EPS:
                return "DOJI"
            return "GREEN" if row["close"] > row["open"] else "RED"
        work["candle_color"] = work.apply(norm_color_row, axis=1)

    # Reorder columns nicely
    cols = ["time", "open", "high", "low", "close",
            "ha_open", "ha_high", "ha_low", "ha_close", "ha_color"]
    if ADD_NORMAL_CANDLE_COLOR:
        cols.append("candle_color")

    return work[cols]


def load_minute_file(path: str) -> pd.DataFrame:
    """
    Loads your minute CSV and normalizes column names.
    Expects at least time/open/high/low/close columns (time can be named date/datetime/timestamp).
    """
    df = pd.read_csv(path)

    # normalize time column name to 'time'
    if "time" not in df.columns:
        for cand in ["date", "datetime", "timestamp"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "time"})
                break

    # enforce dtypes
    df["time"] = pd.to_datetime(df["time"])
    # Some sources may have different column title cases; normalize
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)

    # ensure required columns exist
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    return df.sort_values("time").reset_index(drop=True)


def run():
    if not os.path.exists(INPUT_MINUTE_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_MINUTE_FILE}")

    df_min = load_minute_file(INPUT_MINUTE_FILE)
    df_ha = heikin_ashi(df_min)
    df_ha.to_csv(OUTPUT_HA_FILE, index=False)
    print(f"✅ Heikin Ashi with color saved -> {OUTPUT_HA_FILE}")
    # show last few for quick check
    print(df_ha.tail(5))


if __name__ == "__main__":
    run()
