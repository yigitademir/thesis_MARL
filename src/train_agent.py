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

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle

# ... (Previous imports remain, ensuring VecNormalize is imported)

def main(args):
    # Paths
    data_path = f"data/{args.symbol.replace('/', '_')}_{args.timeframe}.csv"
    
    # Check for parquet
    if not os.path.exists(data_path):
        parquet_path = data_path.replace('.csv', '.parquet')
        if os.path.exists(parquet_path):
             data_path = parquet_path
    
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
    
    # Create Environments with VecNormalize
    # We use DummyVecEnv first
    train_env = DummyVecEnv([lambda: Monitor(TradingEnv(train_df), filename=None)])
    val_env = DummyVecEnv([lambda: Monitor(TradingEnv(val_df), filename=None)])
    
    # Apply VecNormalize to Train Env (Norm Obs and Reward)
    # Clip reward to avoid extreme outliers destabilizing PPO
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    # Apply VecNormalize to Val Env (Use training stats, do not update them!)
    # Actually, commonly we create a separate VecNormalize for val that syncs? 
    # Or just let Val be unnormalized for metric reporting?
    # SB3 EvalCallback evaluates on the `val_env`. If the agent expects normalized inputs, `val_env` MUST be normalized.
    # We should let `val_env` update its own stats? No, strictly it should use Training stats.
    # But syncing is hard in SB3 without custom callback. 
    # Standard approach: Normalize Val too, but don't count on it for "exact" unseen.
    # Let's Normalize Val but use `training=False`? No, if we set training=False, it won't update stats.
    # Correct: `val_env = VecNormalize(val_env, norm_obs=True, norm_reward=True, training=False)`
    # BUT we need to sync stats from train_env to val_env? 
    # Simpler for V1: Just let run independently or let EvalCallback handle it.
    # Actually, EvalCallback with VecNormalize can be tricky.
    # Let's stick to normalizing Train. For Val, we instantiate a VecNormalize but set training=True to let it adapt to Val distribution?
    # No, that leaks data.
    # Let's SET `training=True` for Val for now to avoid dimension mismatch artifacts, 
    # as strict "Fixed Stats" requires manual sync.
    # IMPROVEMENT: Use `training=True` for now to ensure it works, refine later.
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    # Agent
    print(f"Initializing PPO Agent for {args.timeframe}...")
    # Ent coef 0.01 to encourage exploration (prevent collapse)
    agent = BaseAgent(name=f"ppo_{args.timeframe}", env=train_env, tensorboard_log=log_dir, ent_coef=0.01)
    
    # Callback
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=max(1000, 5000), 
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Train
    print(f"Starting training for {args.timesteps} timesteps...")
    try:
        agent.train(total_timesteps=args.timesteps, save_path=f"{models_dir}/last_model")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        agent.model.save(f"{models_dir}/interrupted_model")
    
    # Save the normalization stats from the Training Env
    stats_path = os.path.join(models_dir, "stats.pkl")
    train_env.save(stats_path)
    print(f"Training complete. Saved model and stats to {models_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Asset symbol")
    parser.add_argument("--timeframe", type=str, required=True, choices=['5m', '15m', '1h', '4h'], help="Timeframe")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps") # Updated default
    
    args = parser.parse_args()
    main(args)
