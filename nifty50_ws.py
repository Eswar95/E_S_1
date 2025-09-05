from kiteconnect import KiteConnect, KiteTicker
from config import KiteAPI, Kite_Access_Token, Kite_UserID
import pandas as pd
import datetime as dt
import os
import logging
import time
import pytz
import sys

logging.basicConfig(level=logging.DEBUG)

IST = pytz.timezone("Asia/Kolkata")
MARKET_CLOSE = dt.time(15, 15)   # ⏰ stop at 3:15 PM IST

# Initialize KiteConnect
kite = KiteConnect(api_key=KiteAPI)
kite.set_access_token(Kite_Access_Token)

# Get NIFTY 50 instrument token
df_indices = pd.read_csv("nifty50_index.csv")
nifty50_row = df_indices[df_indices["tradingsymbol"] == "NIFTY 50"]
if nifty50_row.empty:
    raise Exception("❌ NIFTY 50 not found in indices.csv. Run get_instruments.py again.")
nifty_token = int(nifty50_row.iloc[0]["instrument_token"])

folder = "nifty_data"
os.makedirs(folder, exist_ok=True)

today_str = dt.date.today().strftime("%Y-%m-%d")
candle_csv = os.path.join(folder, f"{today_str}minutes_timestamp.csv")
tick_csv   = os.path.join(folder, f"{today_str}seconds_time_stamp.csv")

current_candle = {}
stopped = False  # prevent double-close/logging after shutdown


def on_ticks(ws, ticks):
    """Handle incoming ticks from WebSocket"""
    global current_candle, stopped
    if stopped:
        return

    tick_batch = []
    for tick in ticks:
        # Tick time (UTC -> IST)
        tick_time = tick.get("timestamp", dt.datetime.utcnow()).astimezone(IST)
        # ⏹ If we've crossed 3:15 PM, flush & stop
        if tick_time.time() >= MARKET_CLOSE and not stopped:
            if current_candle:
                save_candle(current_candle)
                print("🧾 Final candle saved at/after 15:15 IST.")
            stopped = True
            print("⏹ Market close (15:15 IST) reached. Closing WebSocket...")
            try:
                ws.close()   # gracefully close the socket
            finally:
                # small delay to let close propagate, then exit
                time.sleep(1)
                sys.exit(0)
            return

        # ---- Aggregate into 1-min candle ----
        minute = tick_time.replace(second=0, microsecond=0)
        price = tick["last_price"]

        if current_candle.get("time") != minute:
            if current_candle:
                save_candle(current_candle)
            current_candle = {
                "time": minute,
                "open": price,
                "high": price,
                "low":  price,
                "close": price
            }
        else:
            current_candle["high"]  = max(current_candle["high"], price)
            current_candle["low"]   = min(current_candle["low"],  price)
            current_candle["close"] = price

        # ---- Save raw tick with candle context ----
        tick_batch.append({
            "date": minute.date(),
            "time": tick_time,
            "open":  current_candle["open"],
            "high":  current_candle["high"],
            "low":   current_candle["low"],
            "close": current_candle["close"],
            "last_price": price,
            "volume": tick.get("volume"),
            "oi":     tick.get("oi")
        })

        print(f"🕒 {minute} | O:{current_candle['open']} H:{current_candle['high']} "
              f"L:{current_candle['low']} C:{current_candle['close']} | Last:{price}")

    if tick_batch:
        save_ticks(tick_batch)


def save_ticks(rows):
    df = pd.DataFrame(rows)
    if os.path.exists(tick_csv):
        df.to_csv(tick_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(tick_csv, index=False)
    print(f"✅ Saved {len(rows)} tick(s)")


def save_candle(candle):
    df = pd.DataFrame([candle])
    if os.path.exists(candle_csv):
        df.to_csv(candle_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(candle_csv, index=False)
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

    # Keep main thread alive; also hard-stop if loop time passes 15:15 (backup)
    while True:
        now_ist = dt.datetime.now(IST).time()
        if now_ist >= MARKET_CLOSE:
            print("⏹ Main loop sees 15:15 IST. Exiting...")
            break
        time.sleep(1)


if __name__ == "__main__":
    run_ws()