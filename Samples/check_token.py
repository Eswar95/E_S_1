from kiteconnect import KiteConnect
from config import KiteAPI, Kite_Access_Token

kite = KiteConnect(api_key=KiteAPI)
kite.set_access_token(Kite_Access_Token)

try:
    profile = kite.profile()
    print("✅ Token is valid! User profile:", profile["user_name"], "| User ID:", profile["user_id"])
except Exception as e:
    print("❌ Token test failed:", e)
