from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token
import pandas as pd
import datetime as dt

# Initialize Kite
kite = KiteConnect(api_key=KiteAPI)
kite.set_access_token(Kite_Access_Token)

print("Fetching instruments...")
instruments = kite.instruments()
df = pd.DataFrame(instruments)

# ---------------- Get NIFTY 50 index ----------------
df_indices = df[(df["segment"] == "INDICES") & (df["exchange"] == "NSE")]
nifty50 = df_indices[df_indices["tradingsymbol"] == "NIFTY 50"]

# ---------------- Get NIFTY options ----------------
df_options = df[(df["segment"] == "NFO-OPT") & (df["name"] == "NIFTY")]

# Filter for expiry within next 14 days
today = dt.date.today()
two_weeks = today + dt.timedelta(days=14)
df_options["expiry"] = pd.to_datetime(df_options["expiry"]).dt.date
df_near_expiry = df_options[(df_options["expiry"] >= today) & (df_options["expiry"] <= two_weeks)]

# Split CE and PE
df_ce = df_near_expiry[df_near_expiry["instrument_type"] == "CE"].sort_values(by="strike").head(5)
df_pe = df_near_expiry[df_near_expiry["instrument_type"] == "PE"].sort_values(by="strike").head(5)

# ---------------- Save results ----------------
nifty50.to_csv("nifty50_index.csv", index=False)
df_ce.to_csv("nifty50_top5_CE.csv", index=False)
df_pe.to_csv("nifty50_top5_PE.csv", index=False)

print("✅ Saved:")
print(" - NIFTY 50 index -> nifty50_index.csv")
print(" - Top 5 CE -> nifty50_top5_CE.csv")
print(" - Top 5 PE -> nifty50_top5_PE.csv")

print("\nNIFTY 50 Index:")
print(nifty50[["instrument_token", "tradingsymbol", "name"]])

print("\nTop 5 CE (nearest expiry):")
print(df_ce[["instrument_token", "tradingsymbol", "expiry", "strike"]])

print("\nTop 5 PE (nearest expiry):")
print(df_pe[["instrument_token", "tradingsymbol", "expiry", "strike"]])
