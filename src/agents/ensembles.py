import numpy as np

class BaseEnsemble:
    def __init__(self, models):
        """
        Args:
            models (dict): Dictionary of {name: model}
        """
        self.models = models
        self.agent_names = list(models.keys())
        
    def predict(self, obs):
        raise NotImplementedError

class VotingEnsemble(BaseEnsemble):
    def predict(self, obs, deterministic=True):
        """
        Simple Average (Soft Voting) of probabilities.
        """
        probs = []
        for name, model in self.models.items():
            # Get action distribution (proba)
            # SB3 PPO predict doesn't return proba directly easily without accessing policy
            # For simplicity in V1, let's use Hard Voting (Mode of actions)
            action, _ = model.predict(obs, deterministic=deterministic)
            probs.append(action)
            
        # Hard Voting
        # Assuming discrete actions [0, 1, 2]
        # logic: 0=Hold, 1=Long, 2=Short
        # We need to be careful with handling scalars vs arrays
        if isinstance(probs[0], np.ndarray):
             # Vectorized input
             stacked = np.stack(probs, axis=1) # (n_envs, n_agents)
             # Mode along axis 1
             from scipy.stats import mode
             final_actions, _ = mode(stacked, axis=1)
             return final_actions.flatten(), None
        else:
             # Single scalar
             return max(set(probs), key=probs.count), None

class AssetWeightedEnsemble(BaseEnsemble):
    def __init__(self, models, initial_capital=10000.0):
        super().__init__(models)
        self.portfolios = {name: initial_capital for name in self.models.keys()}
        self.initial_capital = initial_capital
        # We need to track 'position' and 'entry_price' for each sub-agent 
        # to calculate their virtual PnL at every step.
        # This requires the ensemble to receive 'price' updates.
        self.positions = {name: 0 for name in self.models.keys()} # 0, 1, -1 (Side, not amount? Or full amount?) 
        # Simplified: Assume each agent invests 100% of capital in its signal.
        self.entry_prices = {name: 0.0 for name in self.models.keys()}
        
    def update_portfolios(self, current_price):
        """
        Update portfolio values based on price change and current positions.
        """
        # This needs to be called externally before predict? 
        # Or predict needs to accept 'price'.
        # For now, let's assume we pass a context dict or handle it logic elsewhere.
        pass

    def predict(self, obs, current_price=None, prev_price=None):
        """
        Weighted Vote based on current portfolio values.
        Args:
            obs: Observation for agents
            current_price: Needed to update virtual portfolios
            prev_price: Needed to calc PnL since last step?
        """
        # 1. Update Portfolios if prices available
        if current_price is not None and prev_price is not None:
            for name in self.agent_names:
                pos = self.positions[name]
                if pos != 0:
                    # Calc PnL
                    # Long: (curr - entry) / entry * capital ? 
                    # Simpler: % change
                    # pct_change = (current_price - prev_price) / prev_price
                    # pnl = self.portfolios[name] * pct_change * (1 if pos==1 else -1)
                    # self.portfolios[name] += pnl
                    # Better: Mark-to-Market
                    pass

        # 2. Calculate Weights
        total_value = sum(self.portfolios.values())
        weights = {name: val / total_value for name, val in self.portfolios.items()}
        
        # 3. Weighted Vote
        # Soft Voting difficult with just actions.
        # Let's map actions to vectors: Hold=[0,0], Long=[1,0], Short=[0,1]?
        # Or: 0->0, 1->1, 2->-1. Then weighted sum.
        # Result > 0.33 -> Long, < -0.33 -> Short, else Hold.
        
        score = 0.0
        for name, model in self.models.items():
            action, _ = model.predict(obs, deterministic=True)
            
            # Action Mapping for voting: 0=Hold, 1=Long, 2=Short
            # Let's map to sign: 1->1, 2->-1, 0->0
            val = 0
            if action == 1: val = 1
            elif action == 2: val = -1
            
            score += val * weights[name]
            
            # Update virtual position for next step (Simple execution assumption)
            # Assuming agent gets filled at current_price
            if current_price:
                 self.positions[name] = val # 1, -1, 0
                 
        # Design Decision: Thresholds for ensemble
        final_action = 0
        if score > 0.33: final_action = 1
        elif score < -0.33: final_action = 2
        
        return final_action, weights
