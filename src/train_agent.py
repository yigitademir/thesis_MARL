import argparse
import os
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.trading_env import TradingEnv
from src.features.indicators import add_indicators
from src.agents.base_agent import BaseAgent

def load_and_process_data(filepath):
    # Check for parquet alternative if filepath ends with csv
    if filepath.endswith('.csv'):
        parquet_path = filepath.replace('.csv', '.parquet')
        if os.path.exists(parquet_path):
            print(f"Loading data from {parquet_path} (Parquet)...")
            df = pd.read_parquet(parquet_path)
            # Timestamps are typically preserved in Parquet, but ensure datetime
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                 df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
             print(f"Loading data from {filepath} (CSV)...")
             df = pd.read_csv(filepath)
             df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        # Default fallback
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp just in case
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print("Adding indicators...")
    df = add_indicators(df)
    print(f"Data shape after feature engineering: {df.shape}")
    return df

def main(args):
    # Paths
    data_path = f"data/{args.symbol.replace('/', '_')}_{args.timeframe}.csv"
    models_dir = f"models/{args.symbol.replace('/', '_')}_{args.timeframe}"
    log_dir = f"logs/{args.symbol.replace('/', '_')}_{args.timeframe}"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load Data
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
        
    df = load_and_process_data(data_path)
    
    # Split Data (70% Train, 15% Val, 15% Test)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    
    print(f"Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create Environments
    # We wrap in Monitor to track stats for EvalCallback
    train_env = DummyVecEnv([lambda: Monitor(TradingEnv(train_df), filename=None)])
    val_env = DummyVecEnv([lambda: Monitor(TradingEnv(val_df), filename=None)])
    
    # Agent
    print(f"Initializing PPO Agent for {args.timeframe}...")
    agent = BaseAgent(name=f"ppo_{args.timeframe}", env=train_env, tensorboard_log=log_dir)
    
    # Callback for Evaluation/Saving Best Model
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=max(1000, 5000), # Check every 5000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Train
    print(f"Starting training for {args.timesteps} timesteps...")
    agent.train(total_timesteps=args.timesteps, save_path=f"{models_dir}/last_model")
    
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Asset symbol")
    parser.add_argument("--timeframe", type=str, required=True, choices=['5m', '15m', '1h', '4h'], help="Timeframe")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    
    args = parser.parse_args()
    main(args)
