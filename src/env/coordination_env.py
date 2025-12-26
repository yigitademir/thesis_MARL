import gymnasium as gym
import numpy as np
import pandas as pd
from typing import List, Dict

class CoordinationEnv(gym.Env):
    """
    Coordination Environment for the Manager Agent.
    Wraps the base TradingEnv and 4 Sub-Agents.
    
    Observation Space:
    [
      ...Raw Market Obs (from 5m env)..., 
      Agent_5m_Action, Agent_5m_Conf,
      Agent_15m_Action, Agent_15m_Conf,
      ...
    ]
    
    Action Space:
    Discrete [Hold, Long, Short] (Same as base)
    """
    def __init__(self, base_env, sub_agents: List[Dict]):
        """
        Args:
            base_env: The granular (5m) TradingEnv.
            sub_agents: List of dicts [{'name': '5m', 'model': model, 'df': dataframe}, ...]
        """
        super().__init__()
        self.base_env = base_env
        self.sub_agents = sub_agents
        
        # Pre-process DataFrames for fast lookup
        # We need to map 5m timestamp -> Index in 15m/1h/4h DF
        print("CoordinationEnv: Aligning dataframes...")
        self.data_maps = {}
        base_timestamps = self.base_env.df['timestamp']
        
        for agent in self.sub_agents:
            tf = agent['name'] # e.g. '1h'
            if tf == '5m':
                continue # Base env is 5m
                
            agent_df = agent['df']
            # Create a map: timestamp -> index
            # This is slow if loop. Use searchsorted.
            # Assume both sorted.
            
            # Find the index in agent_df where timestamp <= base_timestamp
            # Actually, for 5m at 10:05, we want the latest closed 1h candle? 
            # If 1h candle corresponds to 10:00-11:00, it closes at 11:00.
            # At 10:05, we only have data up to 10:00 (closed). 
            # So we look for timestamp == 10:00.
            
            # Using searchsorted(side='right') - 1 gives <=
            # But we need strict equality? No, we need "most recent closed".
            
            # Efficient implementation:
            # Reindex agent_df to base_timestamps, ffill?
            # 1. Set index to timestamp
            # 2. Reindex with method='ffill'
            
            params_df = agent_df.set_index('timestamp')
            # Only keep columns needed for observation? 
            # The agent model needs processed features. 
            # agent_df should ALREADY have indicators/features.
            
            aligned_df = params_df.reindex(base_timestamps, method='ffill')
            self.data_maps[tf] = aligned_df.reset_index(drop=True)
            
        print("CoordinationEnv: Alignment complete.")
        
        # Extend Observation Space: 3 probs per agent (0-2)
        base_shape = self.base_env.observation_space.shape[0]
        extra_features = len(sub_agents) * 3 
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(base_shape + extra_features,),
            dtype=np.float32
        )
        self.action_space = self.base_env.action_space
        
    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        meta_obs = self._augment_obs(obs)
        return meta_obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        meta_obs = self._augment_obs(obs)
        return meta_obs, reward, terminated, truncated, info
        
    def _augment_obs(self, base_obs):
        agent_feats = []
        current_step = self.base_env.current_step
        
        for agent in self.sub_agents:
            model = agent['model']
            tf = agent['name']
            
            if tf == '5m':
                # Use base_obs directly!
                # But is base_obs exactly what 5m agent expects?
                # Yes, base_env IS the 5m env.
                obs_input = base_obs
            else:
                # Lookup aligned row
                # The data_maps contain the FULL aligned dataframe.
                # We need the row at 'current_step'.
                # But 'base_obs' is normalized by the environment?
                # Does `agent_df` contain NORMALIZED features? 
                # CRITICAL: models expect NORMALIZED input if they were trained with VecNormalize.
                # If we just load CSV and add indicators, it's NOT normalized.
                # We need to apply the SAME normalization stats used during training.
                
                # Complexity: Training CoordinationEnv requires pre-normalized data 
                # OR loading the VecNormalize statistics for EACH sub-agent.
                
                # For V1 Prototype: Assume models are trained WITHOUT VecNormalize 
                # OR we apply rudimentary scaling here.
                # Wait, I just verified "Refined Agents" use VecNormalize.
                # So we MUST load their stats.
                
                # This is getting complicated for a single file edit.
                # Ideal: The `sub_agents` dict should contain a `predict_fn` that handles this.
                
                # Let's delegate:
                # We assume `agent['predict_fn']` exists and takes (timestamp) or index.
                # But to keep it contained:
                # We will perform inference. 
                # If normalized, we need to normalize this row.
                
                # For now, let's just grab row. We'll handle normalization locally 
                # or assume the calling script sets up the 'df' to be pre-normalized?
                # No, that's impossible.
                
                # Implementation:
                # Get row from Aligned DF
                row = self.data_maps[tf].iloc[current_step]
                
                # Drop non-feature cols
                # We assume the DF has 'timestamp', 'date', maybe 'open', 'high', 'low', 'close', 'volume'?
                # The model trained on features.indicators output.
                # Use drop with ignore.
                vals = row.drop(['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume'], errors='ignore').values.astype(np.float32)
                
                # Normalize if stats available
                if 'stats' in agent and agent['stats'] is not None:
                    # vec_norm.obs_rms is a RunningMeanStd object
                    # obs = (obs - mean) / sqrt(var + epsilon)
                    # Clip obs too?
                    vec_norm = agent['stats']
                    mean = vec_norm.obs_rms.mean
                    var = vec_norm.obs_rms.var
                    epsilon = 1e-8
                    
                    vals = (vals - mean) / np.sqrt(var + epsilon)
                    # Clip (default 10)
                    vals = np.clip(vals, -10.0, 10.0)
                
                obs_input = vals
            
            # Predict Probabilities
            import torch
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_input).float().unsqueeze(0) # Batch dim
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.numpy()[0] # [p_hold, p_long, p_short]
                
            agent_feats.extend(probs)
        
        # Ensure dimensionality match
        if len(base_obs.shape) > 1:
            base_obs = base_obs.flatten()
            
        meta_obs = np.concatenate([base_obs, np.array(agent_feats)], axis=0)
        return meta_obs.astype(np.float32)
