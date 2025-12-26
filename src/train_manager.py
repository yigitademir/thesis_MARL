import os
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.env.trading_env import TradingEnv
from src.env.coordination_env import CoordinationEnv
from src.agents.manager_agent import ManagerAgent
from src.train_agent import load_and_process_data

def load_sub_agents(models_dir_base="models"):
    """
    Load trained models (5m, 15m, 1h, 4h) and their data/stats.
    """
    timeframes = ['5m', '15m', '1h', '4h']
    sub_agents = []
    
    for tf in timeframes:
        print(f"Loading {tf} agent...")
        model_path = f"{models_dir_base}/BTC_USDT_{tf}/last_model"
        stats_path = f"{models_dir_base}/BTC_USDT_{tf}/stats.pkl"
        data_path = f"data/BTC_USDT_{tf}.parquet"
        
        # Load Model
        if not os.path.exists(model_path + ".zip"):
            print(f"Warning: Model {model_path} not found. Skipping.")
            continue
            
        model = PPO.load(model_path)
        
        # Load Stats (VecNormalize)
        stats = None
        if os.path.exists(stats_path):
            # We need a dummy env to load stats into
            dummy_env = DummyVecEnv([lambda: TradingEnv(pd.DataFrame({'close': [100]}))]) # Minimal dummy
            try:
                vec_norm = VecNormalize.load(stats_path, dummy_env)
                stats = vec_norm
                print(f"  Loaded normalization stats.")
            except Exception as e:
                print(f"  Failed to load stats: {e}")
        
        # Load Data
        df = load_and_process_data(data_path) 
        # Note: load_and_process_data adds indicators.
        # This matches what the model was trained on (before normalization).
        
        sub_agents.append({
            'name': tf,
            'model': model,
            'stats': stats,
            'df': df
        })
        
    return sub_agents

def main(args):
    # Log Dir
    log_dir = "logs/Manager"
    models_dir = "models/Manager"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Sub-Agents
    sub_agents = load_sub_agents()
    if not sub_agents:
        print("No sub-agents found. Aborting.")
        return

    # 2. Prepare Base Environment (The Manager trades on 5m timeframe usually? Or 1h?)
    # Thesis proposal: Manager acts on 5m timeframe to be responsive?
    # Or 1h for stability?
    # Let's say Manager makes decisions every 5m.
    data_path = f"data/BTC_USDT_{args.timeframe}.parquet"
    df = load_and_process_data(data_path)
    
    # Split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    
    # 3. Create Coordination Envs
    # We pass the sub_agents list.
    def make_env(data):
        base = TradingEnv(data)
        return CoordinationEnv(base, sub_agents)
        
    train_env = DummyVecEnv([lambda: Monitor(make_env(train_df), filename=None)])
    
    # Normalize Manager's inputs (The meta-obs)
    # Important: The sub-agent probs (0-1) don't need much normalization, 
    # but the base market data part DOES.
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    # 4. Agent
    print("Initializing Manager Agent...")
    agent = ManagerAgent("Manager_PPO", train_env, tensorboard_log=log_dir, ent_coef=0.01)
    
    # 5. Train
    print(f"Starting Manager training for {args.timesteps} steps...")
    try:
        agent.train(total_timesteps=args.timesteps, save_path=f"{models_dir}/last_model")
    except KeyboardInterrupt:
        pass
        
    # Save Stats
    train_env.save(f"{models_dir}/stats.pkl")
    print("Manager Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="5m", help="Base timeframe for Manager")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps")
    args = parser.parse_args()
    main(args)
