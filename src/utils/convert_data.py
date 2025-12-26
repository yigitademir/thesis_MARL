import os
import pandas as pd
import glob

def convert_csv_to_parquet(data_dir="data"):
    """
    Convert all .csv files in data_dir to .parquet
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Starting conversion...")
    
    for csv_path in csv_files:
        try:
            parquet_path = csv_path.replace(".csv", ".parquet")
            
            # Read CSV
            print(f"Reading {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Ensure timestamp logic is consistent if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Save Parquet
            print(f"Saving to {parquet_path}...")
            df.to_parquet(parquet_path, engine='pyarrow', index=False)
            
            # Compare sizes
            csv_size = os.path.getsize(csv_path) / (1024*1024)
            pq_size = os.path.getsize(parquet_path) / (1024*1024)
            print(f"Done. Size: {csv_size:.2f}MB -> {pq_size:.2f}MB (Compression: {csv_size/pq_size:.1f}x)")
            
        except Exception as e:
            print(f"Failed to convert {csv_path}: {e}")

if __name__ == "__main__":
    convert_csv_to_parquet()
