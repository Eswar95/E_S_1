from kiteconnect import KiteConnect, KiteTicker
from config import KiteAPI, Kite_Access_Token, Kite_UserID
import pandas as pd
import datetime as dt
import os
import logging
import time

logging.basicConfig(level=logging.DEBUG)

# Initialize KiteConnect
kite = KiteConnect(api_key=KiteAPI)
kite.set_access_token(Kite_Access_Token)

# Get NIFTY 50 instrument token
df_indices = pd.read_csv("indices.csv")
nifty50_row = df_indices[df_indices["tradingsymbol"] == "NIFTY 50"]
if nifty50_row.empty:
    raise Exception("❌ NIFTY 50 not found in indices.csv. Run get_instruments.py again.")
nifty_token = int(nifty50_row.iloc[0]["instrument_token"])

csv_file = "nifty50_ws_live.csv"
current_candle = {}

def on_ticks(ws, ticks):
    global current_candle
    now = dt.datetime.now()
    minute = now.replace(second=0, microsecond=0)

    for tick in ticks:
        price = tick['last_price']

        if current_candle.get("time") != minute:
            if current_candle:
                save_candle(current_candle)
            current_candle = {
                "time": minute,
                "open": price,
                "high": price,
                "low": price,
                "close": price
            }
        else:
            current_candle["high"] = max(current_candle["high"], price)
            current_candle["low"] = min(current_candle["low"], price)
            current_candle["close"] = price

    print(f"🕒 {minute} | O:{current_candle['open']} H:{current_candle['high']} L:{current_candle['low']} C:{current_candle['close']}")

def save_candle(candle):
    df = pd.DataFrame([candle])
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)
    print(f"✅ Saved candle: {candle}")

def on_connect(ws, response):
    print("🔗 Connected to WebSocket, subscribing...")
    ws.subscribe([nifty_token])
    ws.set_mode(ws.MODE_FULL, [nifty_token])
    print(f"✅ Subscribed to NIFTY 50 ({nifty_token})")

def on_close(ws, code, reason):
    print(f"❌ Connection closed. Code={code}, Reason={reason}")

def on_error(ws, code, reason):
    print(f"⚠️ WebSocket error. Code={code}, Reason={reason}")

def on_noreconnect(ws):
    print("🚫 Will not reconnect")

def on_reconnect(ws, attempt_count):
    print(f"🔄 Reconnecting... Attempt {attempt_count}")

def run_ws():
    kws = KiteTicker(KiteAPI, Kite_Access_Token, Kite_UserID)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    kws.on_noreconnect = on_noreconnect
    kws.on_reconnect = on_reconnect
    kws.connect(threaded=True, disable_ssl_verification=False)

    # ✅ Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    run_ws()
