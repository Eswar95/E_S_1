import pandas as pd

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert normal OHLC candles into Heikin Ashi candles.
    
    Input DataFrame columns:
        ['date', 'open', 'high', 'low', 'close']
    Output DataFrame columns:
        ['date', 'ha_open', 'ha_high', 'ha_low', 'ha_close']
    """

    ha_df = df.copy()

    # Calculate HA_Close
    ha_df['ha_close'] = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4

    # Initialize HA_Open
    ha_open = [(ha_df['open'][0] + ha_df['close'][0]) / 2]

    # Calculate HA_Open iteratively
    for i in range(1, len(ha_df)):
        ha_open.append((ha_open[i-1] + ha_df['ha_close'][i-1]) / 2)

    ha_df['ha_open'] = ha_open

    # HA_High & HA_Low
    ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
    ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

    # Keep only HA columns
    ha_df = ha_df[['date', 'ha_open', 'ha_high', 'ha_low', 'ha_close']]

    return ha_df


# --------- Example Test ---------
if __name__ == "__main__":
    # Example candles (Open, High, Low, Close)
    data = {
        "date": ["2025-09-01 09:15", "2025-09-01 09:16", "2025-09-01 09:17"],
        "open": [100, 102, 101],
        "high": [103, 104, 105],
        "low": [99, 100, 100],
        "close": [102, 101, 104]
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    ha = heikin_ashi(df)
    print(ha)
