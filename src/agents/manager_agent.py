from stable_baselines3 import PPO
from src.agents.base_agent import BaseAgent

class ManagerAgent(BaseAgent):
    """
    The Manager Agent (Meta-Learner).
    Inherits from BaseAgent but tailored for the CoordinationEnv.
    """
    def __init__(self, name, env, tensorboard_log=None, ent_coef=0.0):
        super().__init__(name, env, tensorboard_log, ent_coef)
        # We might want specific hyperparameters for the Manager
        # e.g., lower learning rate as it's fine-tuning?
        self.model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=tensorboard_log,
            ent_coef=ent_coef,
            learning_rate=0.0001, # Slower learning for stability
            gamma=0.99
        )
