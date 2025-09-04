# peCE.py
from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token
import pandas as pd
import datetime as dt
import sys

def get_nifty_spot(kite: KiteConnect) -> float:
    """
    Get NIFTY 50 spot using indices.csv if present; otherwise fallback to instruments() lookup.
    """
    try:
        df_idx = pd.read_csv("indices.csv")
        row = df_idx[df_idx["tradingsymbol"] == "NIFTY 50"]
        if row.empty:
            raise FileNotFoundError("NIFTY 50 not in indices.csv")
        token = int(row.iloc[0]["instrument_token"])
    except Exception:
        # fallback: pull instruments and find NIFTY 50
        instruments = kite.instruments()
        df = pd.DataFrame(instruments)
        row = df[(df["segment"] == "INDICES") & (df["exchange"] == "NSE") & (df["tradingsymbol"] == "NIFTY 50")]
        if row.empty:
            raise RuntimeError("Could not locate NIFTY 50 instrument token")
        token = int(row.iloc[0]["instrument_token"])

    q = kite.quote([token])
    return q[str(token)]["last_price"]

def main():
    kite = KiteConnect(api_key=KiteAPI)
    kite.set_access_token(Kite_Access_Token)

    print("Fetching instruments...")
    instruments = kite.instruments()
    df = pd.DataFrame(instruments)

    # Filter NIFTY Options (NFO-OPT)
    df_opt = df[(df["segment"] == "NFO-OPT") & (df["name"] == "NIFTY")].copy()
    if df_opt.empty:
        print("No NIFTY options found.")
        sys.exit(1)

    # Nearest expiry within next 14 days
    today = dt.date.today()
    two_weeks = today + dt.timedelta(days=14)
    df_opt["expiry"] = pd.to_datetime(df_opt["expiry"]).dt.date
    df_near = df_opt[(df_opt["expiry"] >= today) & (df_opt["expiry"] <= two_weeks)].copy()
    if df_near.empty:
        print("No NIFTY options within next 2 weeks.")
        sys.exit(0)

    nearest_expiry = df_near["expiry"].min()
    df_nearest = df_near[df_near["expiry"] == nearest_expiry].copy()

    # Determine ATM using current NIFTY spot
    spot = get_nifty_spot(kite)
    df_nearest["strike_diff"] = (df_nearest["strike"] - spot).abs()

    # Split CE/PE and pick 5 closest-to-ATM
    df_ce = df_nearest[df_nearest["instrument_type"] == "CE"].sort_values("strike_diff").head(5).copy()
    df_pe = df_nearest[df_nearest["instrument_type"] == "PE"].sort_values("strike_diff").head(5).copy()

    if df_ce.empty and df_pe.empty:
        print("No CE/PE found for nearest expiry.")
        sys.exit(0)

    # Fetch live OHLC for selected tokens
    tokens = [int(t) for t in list(df_ce["instrument_token"]) + list(df_pe["instrument_token"])]
    quotes = kite.quote(tokens)

    def attach_ohlc(df_part: pd.DataFrame) -> pd.DataFrame:
        opens, closes, lasts = [], [], []
        for token in df_part["instrument_token"]:
            q = quotes[str(int(token))]
            opens.append(q["ohlc"]["open"])
            closes.append(q["ohlc"]["close"])  # prev close per Zerodha
            lasts.append(q["last_price"])
        out = df_part.copy()
        out["open"] = opens
        out["close"] = closes
        out["last_price"] = lasts
        return out

    df_ce_out = attach_ohlc(df_ce)[
        ["tradingsymbol", "expiry", "strike", "open", "close", "last_price", "instrument_token"]
    ].reset_index(drop=True)
    df_pe_out = attach_ohlc(df_pe)[
        ["tradingsymbol", "expiry", "strike", "open", "close", "last_price", "instrument_token"]
    ].reset_index(drop=True)

    # Save to separate CSVs
    ce_file = "nifty_ce_open_close.csv"
    pe_file = "nifty_pe_open_close.csv"
    df_ce_out.to_csv(ce_file, index=False)
    df_pe_out.to_csv(pe_file, index=False)

    print(f"✅ Nearest expiry: {nearest_expiry} | Spot: {spot:.2f}")
    print(f"✅ Saved CE -> {ce_file}")
    print(df_ce_out)
    print(f"✅ Saved PE -> {pe_file}")
    print(df_pe_out)

if __name__ == "__main__":
    main()
