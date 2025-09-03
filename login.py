from kiteconnect import KiteConnect
from config import KiteAPI, KiteSecret
import webbrowser
import re

# Initialize Kite
kite = KiteConnect(api_key=KiteAPI)

# Print login URL
print("Login URL:", kite.login_url())

# Open login URL in browser
webbrowser.open(kite.login_url())

# Ask user to paste the request token
request_token = input("Enter the request token: ")
print("Request token received:", request_token)

# Generate access token
data = kite.generate_session(request_token, api_secret=KiteSecret)
access_token = data["access_token"]
print("Access token generated:", access_token)

# Function to update or insert variable in config.py
def update_config_var(var_name, value):
    with open("config.py", "r") as f:
        config_content = f.read()

    # Replace existing variable
    if var_name in config_content:
        config_content = re.sub(
            rf'{var_name}\s*=\s*".*"', 
            f'{var_name} = "{value}"', 
            config_content
        )
    else:
        # Append if not found
        config_content += f'\n{var_name} = "{value}"\n'

    with open("config.py", "w") as f:
        f.write(config_content)

# Update request token and access token in config.py
update_config_var("Kite_Request_Token", request_token)
update_config_var("Kite_Access_Token", access_token)

print("✅ config.py updated with new Kite_Request_Token and Kite_Access_Token")
