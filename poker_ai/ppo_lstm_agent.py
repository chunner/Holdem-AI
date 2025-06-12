from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from poker_ai.poker_env import PokerEnv
from stable_baselines3.common.torch_layers import RecurrentActorCriticPolicy

env = make_vec_env('PokerEnv', n_envs=1)

model = PPO(
    policy=RecurrentActorCriticPolicy,
    env=env,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=[128, 128], lstm_hidden_size=128)
)
model.learn(total_timesteps=1_000_000)
model.save("ppo_lstm_poker")