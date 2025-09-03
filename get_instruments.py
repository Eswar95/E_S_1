from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token
import pandas as pd

# Initialize Kite
kite = KiteConnect(api_key=KiteAPI)
kite.set_access_token(Kite_Access_Token)

# Fetch all instruments
print("Fetching instruments...")
instruments = kite.instruments()

# Convert to DataFrame
df = pd.DataFrame(instruments)

# Filter only indices
df_indices = df[(df["segment"] == "INDICES") & (df["exchange"] == "NSE")]

# Save to CSV
df_indices.to_csv("indices.csv", index=False)

print(f"✅ Saved {len(df_indices)} index instruments to indices.csv")
print(df_indices[["instrument_token", "tradingsymbol", "name"]])
