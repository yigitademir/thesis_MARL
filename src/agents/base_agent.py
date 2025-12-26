from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

class BaseAgent:
    """
    Base Agent class focusing on a specific timeframe.
    Wraps Stable Baselines3 PPO.
    """
    def __init__(self, name, env, tensorboard_log=None, ent_coef=0.0):
        self.name = name
        self.env = env
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            ent_coef=ent_coef, # Entropy Coefficient for exploration
            device='auto'
        )

    def train(self, total_timesteps: int = 100000, save_path: str = None):
        """
        Train the agent.
        """
        callbacks = []
        if save_path:
            # Save checkpoints
            checkpoint_callback = CheckpointCallback(
                save_freq=max(10000, total_timesteps // 10),
                save_path=os.path.dirname(save_path),
                name_prefix=self.name
            )
            callbacks.append(checkpoint_callback)
            
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
        
        if save_path:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
            
    def load(self, path: str):
        """
        Load a trained model.
        """
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            self.model = PPO.load(path, env=self.env)
            print(f"Model loaded from {path}")
        else:
            print(f"Model path {path} not found.")

    def predict(self, state, deterministic=True):
        return self.model.predict(state, deterministic=deterministic)
