import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from stable_baselines3.common.env_checker import check_env
from src.env.trading_env import TradingEnv
from src.features.indicators import add_indicators

def create_mock_data():
    # Create dummy OHLCV data
    dates = pd.date_range(start='2021-01-01', periods=200, freq='1h')
    data = {
        'timestamp': dates,
        'open': np.random.rand(200) * 100 + 10000,
        'high': np.random.rand(200) * 100 + 10100,
        'low': np.random.rand(200) * 100 + 9900,
        'close': np.random.rand(200) * 100 + 10050,
        'volume': np.random.rand(200) * 1000
    }
    df = pd.DataFrame(data)
    return df

def test_env():
    print("Generating mock data...")
    df = create_mock_data()
    
    print("Adding indicators...")
    try:
        df = add_indicators(df)
        print("Indicators added successfully.")
        print(f"Feature columns: {[c for c in df.columns if c != 'timestamp']}")
    except Exception as e:
        print(f"Failed to add indicators: {e}")
        return

    print("Initializing Environment...")
    env = TradingEnv(df)
    
    print("Checking Environment compliance...")
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return
        
    print("Running random agent loop...")
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Balance: {info['portfolio_value']:.2f}")
        if terminated or truncated:
            obs, info = env.reset()
            
    print("Verification complete.")

if __name__ == "__main__":
    test_env()
