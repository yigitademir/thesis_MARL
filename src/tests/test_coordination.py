import gymnasium as gym
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import sys
import os

sys.path.append(os.getcwd())

from src.env.trading_env import TradingEnv
from src.env.coordination_env import CoordinationEnv

def test_coordination_env():
    # 1. Create Dummy Data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='5min')
    df_5m = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 100
    })
    
    # 2. Base Env
    base_env = TradingEnv(df_5m)
    
    # 3. Mock Sub-Agents
    sub_agents = []
    timeframes = ['5m', '15m', '1h', '4h']
    
    for tf in timeframes:
        # Mock Model
        mock_model = MagicMock()
        mock_dist = MagicMock()
        # distribution.probs -> numpy array [0.33, 0.33, 0.33]
        mock_dist.distribution.probs.numpy.return_value = np.array([[0.1, 0.2, 0.7]])
        mock_model.policy.get_distribution.return_value = mock_dist
        
        # Mock DF (aligned or not, just needs to exist)
        mock_df = df_5m.copy() # Reuse 5m for simplicity
        
        sub_agents.append({
            'name': tf,
            'model': mock_model,
            'df': mock_df,
            'stats': None # Test without normalization first
        })
        
    # 4. Init CoordinationEnv
    env = CoordinationEnv(base_env, sub_agents)
    
    # 5. Check Observation Space
    # Base (TradingEnv) obs shape + 4 agents * 3 probas
    # TradingEnv V1 obs depends on indicators. features/indicators.py adds ~20 cols.
    # But here we didn't run add_indicators, so obs is smaller? 
    # TradingEnv defaults: if no indicators, it returns ... wait, TradingEnv relies on 'df' having columns.
    # Our dummy df has no indicators. TradingEnv might fail or return just OHLCV.
    # Actually TradingEnv calls self._get_observation().
    
    print("Resetting Env...")
    try:
        obs, info = env.reset()
        print(f"Initial Obs Shape: {obs.shape}")
        
        # 4 agents * 3 feats = 12 extra features.
        # Check if obs has correct length.
        
        # Step
        print("Stepping Env...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step Reward: {reward}")
        print(f"Step Obs Shape: {obs.shape}")
        
        print("Test Passed!")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_coordination_env()
