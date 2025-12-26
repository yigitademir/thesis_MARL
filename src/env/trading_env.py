import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Tuple, Dict, Any

class TradingEnv(gym.Env):
    """
    A custom trading environment for Bitcoin trading.
    Compatible with OpenAI Gym / Gymnasium.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, fee: float = 0.001, window_size: int = 10):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.fee = fee
        self.window_size = window_size
        
        # Action Space: 0=Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: 
        # Feature columns (exclude timestamp)
        self.feature_columns = [c for c in df.columns if c != 'timestamp']
        self.shape = (window_size, len(self.feature_columns))
        
        # Using float32 for observations
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32
        )
        
        # Internal State
        self.current_step = window_size
        self.balance = initial_balance
        self.position = 0 # 0=None, 1=Long, -1=Short
        self.entry_price = 0.0
        self.portfolio_value = initial_balance
        self.history = []
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Random start option could be added here for training
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.history = []
        
        return self._get_observation(), {}
        
    def _get_observation(self) -> np.ndarray:
        # Return window_size records ending at current_step-1? 
        # Actually usually it is [current_step-window : current_step]
        # Make sure we don't look ahead. current_step is the index of the "current" candle we just finished or about to trade?
        # Let's assume current_step is the index of the LATEST available candle.
        obs = self.df.iloc[self.current_step - self.window_size + 1 : self.current_step + 1][self.feature_columns].values
        return obs.astype(np.float32)
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Execute action based on Close price of current_step
        current_price = self.df.iloc[self.current_step]['close']
        prev_portfolio_value = self.portfolio_value
        
        # 0: Hold, 1: Long, 2: Short
        if action == 1: # Long
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.portfolio_value * self.fee
            elif self.position == -1: # Switch Short to Long
                # Close Short
                pnl = (self.entry_price - current_price) / self.entry_price
                self.balance *= (1 + pnl)
                self.balance -= self.portfolio_value * self.fee
                # Open Long
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.portfolio_value * self.fee
                
        elif action == 2: # Short
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.portfolio_value * self.fee
            elif self.position == 1: # Switch Long to Short
                # Close Long
                pnl = (current_price - self.entry_price) / self.entry_price
                self.balance *= (1 + pnl)
                self.balance -= self.portfolio_value * self.fee
                # Open Short
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.portfolio_value * self.fee
        
        # Update Portfolio Value
        if self.position == 1:
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            self.portfolio_value = self.balance * (1 + unrealized_pnl_pct)
        elif self.position == -1:
            unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
            self.portfolio_value = self.balance * (1 + unrealized_pnl_pct)
        else:
            self.portfolio_value = self.balance
            
        # Reward
        reward = self.portfolio_value - prev_portfolio_value
        
        # Advance Step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        next_obs = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'price': current_price
        }
        
        return next_obs, reward, terminated, truncated, info
