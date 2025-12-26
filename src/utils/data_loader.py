import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_ohlcv(symbol, timeframe, since, limit=1000, max_retries=3):
    """
    Fetch OHLCV data from Binance using CCXT with retry logic and pagination.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    all_ohlcv = []
    current_since = since
    
    # Calculate end time (now)
    now = exchange.milliseconds()
    
    print(f"Fetching {symbol} {timeframe} data from {datetime.fromtimestamp(since/1000)}...")
    
    while current_since < now:
        retries = 0
        while retries < max_retries:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
                if not ohlcv:
                    break
                
                # Check if we got new data
                if all_ohlcv and ohlcv[0][0] == all_ohlcv[-1][0]:
                     # If the first timestamp of new batch matches last of collected, it's a duplicate or stuck.
                     # Move past the last collected point
                     current_since = all_ohlcv[-1][0] + 1
                     continue

                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1 
                
                # Progress update
                last_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                print(f"  Fetched up to {last_date}")
                
                break # Success, break retry loop
            except Exception as e:
                print(f"  Error fetching data: {e}. Retrying in {2**retries} seconds...")
                time.sleep(2**retries)
                retries += 1
        
        if retries == max_retries:
            print("  Max retries reached. Stopping fetch.")
            break
            
        if not ohlcv:
             break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def save_data(df, symbol, timeframe, output_dir="data"):
    """
    Save DataFrame to CSV.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Clean symbol for filename (e.g., BTC/USDT -> BTC_USDT)
    clean_symbol = symbol.replace('/', '_')
    filename = f"{clean_symbol}_{timeframe}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")

def main():
    symbol = 'BTC/USDT'
    timeframes = ['5m', '15m', '1h', '4h']
    
    # Approx 5 years ago in milliseconds
    five_years_ago = datetime.now() - timedelta(days=5*365)
    since_timestamp = int(five_years_ago.timestamp() * 1000)
    
    for tf in timeframes:
        try:
            print(f"\n--- Processing {tf} ---")
            df = fetch_ohlcv(symbol, tf, since_timestamp)
            if not df.empty:
                save_data(df, symbol, tf)
            else:
                print(f"No data found for {tf}")
        except Exception as e:
            print(f"Failed to process {tf}: {e}")

if __name__ == "__main__":
    main()
