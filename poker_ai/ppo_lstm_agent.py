from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from poker_env import PokerEnv
# from stable_baselines3.common.torch_layers import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import logging
import time
import os
import shutil

# configure logging
# logging.basicConfig(
#     filename='train.log', 
#     level=logging.INFO, 
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
log_dir = './log'
model_dir = './model'
os.makedirs(log_dir, exist_ok=True)
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'train.log')
with open(log_path, 'w') :
    pass


agent_logger = logging.getLogger("agent")
agent_logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
agent_logger.addHandler(fh)

class LogCallback(BaseCallback):
    def __init__(self, total_steps, model,verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()
        self.total_steps = total_steps
        self.model = model
        self.episode_cnt = 0

    def _on_step(self) -> bool:
        reward = self.locals.get('rewards', [0])[0]
        done = self.locals.get('dones', [False])[0]

        if done and self.episode_cnt% 100 == 0:
            # log
            elapsed_time = time.time() - self.start_time
            percent = self.num_timesteps / self.total_steps * 100
            agent_logger.info(f"Step: {self.num_timesteps}, Reward: {reward}, "
                              f"Elapsed Time: {elapsed_time:.2f}s, "
                              f"Progress: {percent:.2f}%")
            # save model every 100 episodes
            self.model.save(os.path.join(model_dir, f"ppo_lstm_poker_{self.episode_cnt}.zip"))
            agent_logger.info(f"Model saved at episode {self.episode_cnt}")

        if done:
            self.episode_cnt += 1

        return True



env = DummyVecEnv([lambda: PokerEnv()])

model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=[128, 128])#, lstm_hidden_size=128)
)
log_callback = LogCallback(total_steps=1_000_000, model=model)
model.learn(total_timesteps=1_000_000, callback=log_callback)
model.save("ppo_lstm_poker")