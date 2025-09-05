# HA_FILE = os.path.join(DATA_DIR, "2025-09-05_nifty50_1min_ha.csv")
# OPTION_CE_FILE = os.path.join(DATA_DIR, "2025-09-05_NIFTY2590924800CE_1min_with_ha.csv")
# OPTION_PE_FILE = os.path.join(DATA_DIR, "2025-09-05_NIFTY2590924800PE_1min_with_ha.csv")
#!/usr/bin/env python3
"""
hc_trade_backtest_complete.py

Full backtest implementing:
- compute cal1 -> form1/form2 and named levels (entryA..D)
- check level hit against underlying OHLC (open/high/low/close) for that minute
- on hit: if ha_color GREEN -> CE (or RED -> PE). BUY_ONLY option available.
- enter at next minute's option last_price (within wait window)
- exit when option >= entry+3 (TARGET) or <= entry-1 (STOP). Target checked first.
- robust option file loading, per-trade CSV output, skip-reason breakdown.

Config at top.
"""
from __future__ import annotations
import os
import re
import pathlib
import math
import pandas as pd
import datetime as dt
from typing import Optional, Dict, Any, List

# ---------------- CONFIG ----------------
DATA_DIR = "nifty_data"

# HA file (must contain time and ha_* columns; ideally also include underlying open/high/low/close)
HA_FILE = os.path.join(DATA_DIR, "2025-09-05_nifty50_1min_ha.csv")
OPTION_CE_FILE = os.path.join(DATA_DIR, "2025-09-05_NIFTY2590924800CE_1min_with_ha.csv")
OPTION_PE_FILE = os.path.join(DATA_DIR, "2025-09-05_NIFTY2590924800PE_1min_with_ha.csv")

# If set (e.g. "09:15:00") run for that HA minute only; else None => run across day
SINGLE_HA_TIME: Optional[str] = None

# Only take one side (buy-only): set True to only take BUY_SIDE trades (CE or PE), False to take both sides
BUY_ONLY = True
BUY_SIDE = "CE"  # "CE" or "PE"

# Strike bucket + candidate offsets (adjust to your strike ladder)
STRIKE_ROUND = 100
STRIKE_OFFSETS = [-200, -100, 0, 100, 200]  # set [0] for ATM-only

# Entry waiting policy
MAX_ENTRY_WAIT_MINUTES = 3  # minutes to look for entry quote after the HA candle

# Position sizing
LOT_SIZE = 50
QTY_LOTS = 1

# Exit rules
TARGET_RUPEES = 3.0
STOP_RUPEES = 1.0

PI = 22 / 7  # per your formula

# Level priority when multiple named levels are reached
# For form1 branch: order = ["entryA", "entryB"], for form2: ["entryC","entryD"]
PRIORITY_ORDER_FORM1 = ["entryA", "entryB"]
PRIORITY_ORDER_FORM2 = ["entryC", "entryD"]

# Output
OUTPUT_CSV = os.path.join(DATA_DIR, "hc_backtest_complete_results.csv")
# ----------------------------------------

# ---- Utility / loaders ----
def ensure_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

def parse_time_col(df: pd.DataFrame, col_name: str = "time") -> pd.DataFrame:
    df = df.copy()
    df[col_name] = pd.to_datetime(df[col_name])
    return df

def load_ha(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower_map = {c.lower(): c for c in df.columns}
    required = ["time", "ha_open", "ha_high", "ha_low", "ha_close", "ha_color"]
    for r in required:
        if r not in lower_map:
            raise ValueError(f"HA file missing required column (case-insensitive): {r}. Found: {list(df.columns)}")
    # rename to canonical names
    df = df.rename(columns={lower_map[c]: c for c in lower_map})
    # parse times
    df = parse_time_col(df, "time")
    df = df.sort_values("time").reset_index(drop=True)
    return df

# Robust option loader (supports combined multi-symbol and per-symbol OHLC files)
def _infer_tradingsymbol_from_filename(path: str) -> Optional[str]:
    stem = pathlib.Path(path).stem.upper()
    # try to find typical fragments containing CE/PE and digits
    m = re.search(r"([A-Z]{2,}\d{0,6}[A-Z]{0,6}\d{3,6}(?:CE|PE))", stem)
    if m:
        return m.group(1)
    m2 = re.search(r"([A-Z0-9_]*?(?:CE|PE)[A-Z0-9_]*)", stem)
    if m2:
        cleaned = re.sub(r"[^A-Z0-9]", "", m2.group(1))
        return cleaned if cleaned else None
    if 3 <= len(stem) <= 40:
        return stem
    return None

def load_option_file(path: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns: time (datetime), tradingsymbol (str), last_price (float)
    Accepts:
    - Combined multi-symbol with tradingsymbol + last_price/close
    - Per-symbol OHLC file (will infer symbol from filename and use close as last_price)
    """
    df = pd.read_csv(path)
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    # time col
    time_col = None
    for cand in ("time", "date", "datetime", "timestamp"):
        if cand in lower_map:
            time_col = lower_map[cand]; break
    if time_col is None:
        raise ValueError(f"Option file {path} missing time-like column. Columns: {cols}")

    df[time_col] = pd.to_datetime(df[time_col])

    trad_col = None
    for cand in ("tradingsymbol", "tradingsymbols", "symbol", "instrument"):
        if cand in lower_map:
            trad_col = lower_map[cand]; break

    last_col = None
    for cand in ("last_price", "lastprice", "last", "close", "ltp"):
        if cand in lower_map:
            last_col = lower_map[cand]; break

    # Case: has trad & last -> straightforward
    if trad_col and last_col:
        df = df.rename(columns={time_col: "time", trad_col: "tradingsymbol", last_col: "last_price"})
        df = df[["time", "tradingsymbol", "last_price"]].copy()
        df["tradingsymbol"] = df["tradingsymbol"].astype(str)
        df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
        df = df.dropna(subset=["last_price"]).sort_values(["tradingsymbol", "time"]).reset_index(drop=True)
        return df

    # Case: no trad col but has close/last -> per-symbol OHLC style file: infer symbol from filename
    if (not trad_col) and last_col:
        inferred = _infer_tradingsymbol_from_filename(path)
        if not inferred:
            raise ValueError(f"Option file {path} has no tradingsymbol column and cannot infer symbol. Columns: {cols}")
        df = df.rename(columns={time_col: "time", last_col: "last_price"})
        df["tradingsymbol"] = inferred
        df = df[["time", "tradingsymbol", "last_price"]].copy()
        df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
        df = df.dropna(subset=["last_price"]).sort_values(["tradingsymbol", "time"]).reset_index(drop=True)
        return df

    # Case: trad exists but last missing: try map close
    if trad_col and (not last_col):
        if "close" in lower_map:
            df = df.rename(columns={time_col: "time", trad_col: "tradingsymbol", lower_map["close"]: "last_price"})
            df = df[["time", "tradingsymbol", "last_price"]].copy()
            df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
            df = df.dropna(subset=["last_price"]).sort_values(["tradingsymbol", "time"]).reset_index(drop=True)
            return df
        else:
            raise ValueError(f"Option file {path} has tradingsymbol but no close/last column. Columns: {cols}")

    raise ValueError(f"Option file {path} missing required fields. Columns: {cols}")

# ---- Strike / symbol helpers ----
_STRIKE_RE = re.compile(r"(\d{3,6})")
def extract_strike(trad: str) -> Optional[int]:
    m = _STRIKE_RE.search(trad.upper())
    return int(m.group(1)) if m else None

def find_symbol_for_strike(prices_df: pd.DataFrame, strike: int, after_time: pd.Timestamp) -> Optional[str]:
    s = str(int(strike))
    mask = prices_df["tradingsymbol"].str.contains(s, case=False, na=False)
    candidates = prices_df[mask]["tradingsymbol"].unique().tolist()
    if not candidates:
        return None
    best_sym = None
    best_ts = None
    for sym in candidates:
        df_sym = prices_df[prices_df["tradingsymbol"] == sym]
        after = df_sym[df_sym["time"] >= after_time]
        first_ts = after["time"].min() if not after.empty else df_sym["time"].max()
        if best_ts is None or (pd.notna(first_ts) and first_ts < best_ts):
            best_ts = first_ts
            best_sym = sym
    return best_sym

def get_entry_row_for_symbol(prices_df: pd.DataFrame, symbol: str, earliest_time: pd.Timestamp, max_wait_minutes: int) -> Optional[pd.Series]:
    df_sym = prices_df[prices_df["tradingsymbol"] == symbol]
    if df_sym.empty:
        return None
    window_end = earliest_time + pd.Timedelta(minutes=max_wait_minutes)
    cand = df_sym[(df_sym["time"] >= earliest_time) & (df_sym["time"] <= window_end)].sort_values("time")
    if cand.empty:
        return None
    return cand.iloc[0]

def simulate_trade_for_symbol(prices_df: pd.DataFrame, symbol: str, entry_time: pd.Timestamp, target: float, stop: float) -> Dict[str, Any]:
    df_sym = prices_df[prices_df["tradingsymbol"] == symbol]
    if df_sym.empty:
        return {"result": "NO_DATA", "symbol": symbol}
    df_after = df_sym[df_sym["time"] >= entry_time].sort_values("time")
    if df_after.empty:
        return {"result": "NO_QUOTE_AFTER_ENTRY", "symbol": symbol}
    entry_row = df_after.iloc[0]
    entry_price = float(entry_row["last_price"])
    target_price = entry_price + target
    stop_price = entry_price - stop
    for _, r in df_after.iterrows():
        p = float(r["last_price"])
        # Check target first
        if p >= target_price:
            pnl = round(target_price - entry_price, 2)
            return {
                "result": "TARGET",
                "symbol": symbol,
                "entry_time": entry_row["time"],
                "exit_time": r["time"],
                "entry_price": round(entry_price, 2),
                "exit_price": round(p, 2),
                "pnl_per_contract": pnl
            }
        if p <= stop_price:
            pnl = round(stop_price - entry_price, 2)
            return {
                "result": "STOP",
                "symbol": symbol,
                "entry_time": entry_row["time"],
                "exit_time": r["time"],
                "entry_price": round(entry_price, 2),
                "exit_price": round(p, 2),
                "pnl_per_contract": pnl
            }
    # unresolved
    last_p = float(df_after.iloc[-1]["last_price"])
    pnl = round(last_p - entry_price, 2)
    return {
        "result": "UNRESOLVED",
        "symbol": symbol,
        "entry_time": entry_row["time"],
        "exit_time": df_after.iloc[-1]["time"],
        "entry_price": round(entry_price, 2),
        "exit_price": round(last_p, 2),
        "pnl_per_contract": pnl
    }

# ---- Core backtest ----
def run_backtest():
    ensure_file_exists(HA_FILE)
    ensure_file_exists(OPTION_CE_FILE)
    ensure_file_exists(OPTION_PE_FILE)

    ha = load_ha(HA_FILE)
    prices_ce = load_option_file(OPTION_CE_FILE)
    prices_pe = load_option_file(OPTION_PE_FILE)

    # Optionally filter to a single HA time
    if SINGLE_HA_TIME:
        t_only = dt.datetime.strptime(SINGLE_HA_TIME, "%H:%M:%S").time()
        ha = ha[ha["time"].dt.time == t_only].reset_index(drop=True)
        if ha.empty:
            raise RuntimeError(f"No HA rows found at time {SINGLE_HA_TIME} in {HA_FILE}")

    results: List[Dict[str, Any]] = []
    skip_reasons: Dict[str, int] = {}

    for idx, row in ha.iterrows():
        T = row["time"]
        # get HA values (presence guaranteed)
        ha_open = float(row["ha_open"])
        ha_high = float(row["ha_high"])
        ha_low = float(row["ha_low"])
        ha_close = float(row["ha_close"])
        ha_color = str(row.get("ha_color", "")).upper()

        # prefer underlying OHLC if present
        if all(c in row.index for c in ("open", "high", "low", "close")):
            underlying_open = float(row["open"])
            underlying_high = float(row["high"])
            underlying_low = float(row["low"])
            underlying_close = float(row["close"])
        else:
            # fallback to HA-derived OHLC if underlying missing
            underlying_open = ha_open
            underlying_high = ha_high
            underlying_low = ha_low
            underlying_close = ha_close

        # compute cal1 & named levels
        cal1 = ha_close - ha_open
        form1 = None
        form2 = None
        named_levels: Dict[str, float] = {}
        used_formula = None

        if cal1 <= 20:
            form1 = cal1 * PI
            entryA = ha_close - cal1
            entryB = ha_high - form1
            named_levels = {"entryA": entryA, "entryB": entryB}
            used_formula = "form1"
            priority = PRIORITY_ORDER_FORM1
        else:
            form2 = cal1 / PI
            entryC = ha_close - form2
            entryD = ha_high - form2
            named_levels = {"entryC": entryC, "entryD": entryD}
            used_formula = "form2"
            priority = PRIORITY_ORDER_FORM2

        # trigger test uses underlying low/high
        reached_named: List[str] = []
        reached_values: List[float] = []
        for name, val in named_levels.items():
            if underlying_low <= val <= underlying_high:
                reached_named.append(name)
                reached_values.append(val)

        triggered = len(reached_named) > 0

        base = {
            "ha_time": T,
            "ha_color": ha_color,
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
            "underlying_open": underlying_open,
            "underlying_high": underlying_high,
            "underlying_low": underlying_low,
            "underlying_close": underlying_close,
            "cal1": cal1,
            "form1": form1,
            "form2": form2,
            "named_levels": named_levels,
            "used_formula": used_formula,
            "triggered": triggered,
            "reached_named_levels": reached_named,
            "reached_named_values": reached_values
        }

        if not triggered:
            base.update({"note": "NO_LEVEL_REACHED"})
            results.append(base)
            skip_reasons["NO_LEVEL_REACHED"] = skip_reasons.get("NO_LEVEL_REACHED", 0) + 1
            continue

        # BUY_ONLY filter
        if BUY_ONLY:
            if BUY_SIDE == "CE" and ha_color != "GREEN":
                base.update({"note": "SKIPPED_BY_BUY_ONLY"})
                results.append(base)
                skip_reasons["SKIPPED_BY_BUY_ONLY"] = skip_reasons.get("SKIPPED_BY_BUY_ONLY", 0) + 1
                continue
            if BUY_SIDE == "PE" and ha_color != "RED":
                base.update({"note": "SKIPPED_BY_BUY_ONLY"})
                results.append(base)
                skip_reasons["SKIPPED_BY_BUY_ONLY"] = skip_reasons.get("SKIPPED_BY_BUY_ONLY", 0) + 1
                continue

        # choose price dataframe by side
        if ha_color == "GREEN":
            prices_side = prices_ce
            side_label = "CE"
        elif ha_color == "RED":
            prices_side = prices_pe
            side_label = "PE"
        else:
            base.update({"note": "UNKNOWN_HA_COLOR"})
            results.append(base)
            skip_reasons["UNKNOWN_HA_COLOR"] = skip_reasons.get("UNKNOWN_HA_COLOR", 0) + 1
            continue

        # choose triggered level by priority (first in priority that exists in reached_named)
        chosen_trigger_level = None
        chosen_trigger_value = None
        for p in priority:
            if p in reached_named:
                chosen_trigger_level = p
                # find corresponding value
                chosen_trigger_value = named_levels.get(p)
                break
        # fallback: pick first reached if priority didn't match
        if chosen_trigger_level is None and reached_named:
            chosen_trigger_level = reached_named[0]
            chosen_trigger_value = named_levels.get(chosen_trigger_level)

        base["chosen_trigger_level"] = chosen_trigger_level
        base["chosen_trigger_value"] = chosen_trigger_value

        # ATM and candidate strikes
        atm = int(round(ha_close / STRIKE_ROUND) * STRIKE_ROUND)
        candidate_strikes = [atm + off for off in STRIKE_OFFSETS]

        T_next = T + pd.Timedelta(minutes=1)

        # For each candidate strike attempt to simulate
        any_sim_executed = False
        for strike in candidate_strikes:
            rec = dict(base)
            rec.update({"strike": strike, "side": side_label})

            sym = find_symbol_for_strike(prices_side, strike, T_next)
            if sym is None:
                rec.update({"symbol": None, "note": "NO_SYMBOL_FOR_STRIKE_IN_SIDE_FILE"})
                results.append(rec)
                skip_reasons["NO_SYMBOL_FOR_STRIKE"] = skip_reasons.get("NO_SYMBOL_FOR_STRIKE", 0) + 1
                continue

            entry_row = get_entry_row_for_symbol(prices_side, sym, T_next, MAX_ENTRY_WAIT_MINUTES)
            if entry_row is None:
                rec.update({"symbol": sym, "note": f"NO_ENTRY_QUOTE_WITHIN_{MAX_ENTRY_WAIT_MINUTES}m"})
                results.append(rec)
                skip_reasons["NO_ENTRY_QUOTE"] = skip_reasons.get("NO_ENTRY_QUOTE", 0) + 1
                continue

            any_sim_executed = True
            sim = simulate_trade_for_symbol(prices_side, sym, entry_row["time"], TARGET_RUPEES, STOP_RUPEES)
            pnl_per_contract = sim.get("pnl_per_contract")
            pnl_per_lot = pnl_per_contract * LOT_SIZE * QTY_LOTS if pnl_per_contract is not None else None

            rec.update({
                "symbol": sym,
                "entry_time": sim.get("entry_time"),
                "entry_price": sim.get("entry_price"),
                "exit_time": sim.get("exit_time"),
                "exit_price": sim.get("exit_price"),
                "result": sim.get("result"),
                "pnl_per_contract": sim.get("pnl_per_contract"),
                "pnl_per_lot": round(pnl_per_lot, 2) if pnl_per_lot is not None else None,
                "note": None
            })
            results.append(rec)

        if not any_sim_executed:
            skip_reasons["TRIGGER_BUT_NO_EXECUTION"] = skip_reasons.get("TRIGGER_BUT_NO_EXECUTION", 0) + 1

    # save results
    df_out = pd.DataFrame(results)
    # serialize complex fields for CSV readability
    if "named_levels" in df_out.columns:
        df_out["named_levels"] = df_out["named_levels"].apply(lambda x: str(x))
    if "reached_named_levels" in df_out.columns:
        df_out["reached_named_levels"] = df_out["reached_named_levels"].apply(lambda x: ",".join(x) if isinstance(x, list) else "")
    if "reached_named_values" in df_out.columns:
        df_out["reached_named_values"] = df_out["reached_named_values"].apply(lambda x: ",".join([f"{v:.3f}" for v in x]) if isinstance(x, list) else "")

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_out)} rows -> {OUTPUT_CSV}")

    # summary
    df_trades = df_out[df_out["triggered"] == True].copy()
    executed = df_trades[df_trades["result"].notna()]
    wins = executed[executed["result"] == "TARGET"]
    losses = executed[executed["result"] == "STOP"]
    unresolved = executed[executed["result"] == "UNRESOLVED"]

    total_trades = len(executed)
    wins_n = len(wins)
    losses_n = len(losses)
    unresolved_n = len(unresolved)
    total_pnl = executed["pnl_per_lot"].dropna().sum() if "pnl_per_lot" in executed else 0.0
    win_rate = (wins_n / (wins_n + losses_n)) if (wins_n + losses_n) > 0 else 0.0
    expectancy = (total_pnl / total_trades) if total_trades > 0 else 0.0

    print("=== BACKTEST SUMMARY ===")
    print(f"HA triggers (rows with triggered==True): {len(df_trades)}")
    print(f"Executed trades (with result): {total_trades}")
    print(f"Wins: {wins_n}, Losses: {losses_n}, Unresolved: {unresolved_n}")
    print(f"Win rate (wins / (wins+losses)): {win_rate:.2%}")
    print(f"Total P/L (₹): {total_pnl:.2f}")
    print(f"Expectancy per trade (₹): {expectancy:.2f}")

    print("\n--- Skip reasons breakdown ---")
    for k, v in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"{k}: {v}")

if __name__ == "__main__":
    run_backtest()