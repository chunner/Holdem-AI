import numpy as np
import gymnasium as gym
from gymnasium import spaces
from poker_ai.utils import get_card_coding, sendJson, recvJson, get_action, action_to_actionstr
import socket
import logging
import os
from sb3_contrib import RecurrentPPO
from poker.ia.env import Env
from poker.ia.action import IaAction
NUM_PLAYERS = 6
NUM_HELPERS = NUM_PLAYERS - 1  # Number of helper agents
INIT_MONEY = 1000
OPP_UPDATE_FREQ = 500




log_dir = './log'
model_dir = './model'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'poker_env_6.log')
with open(log_path, 'w'):
    pass

env_logger = logging.getLogger('env')
env_logger.setLevel(logging.INFO)
fh=logging.FileHandler(log_path)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
env_logger.addHandler(fh)


class PokerEnv(gym.Env):
    def __init__(self):
        super(PokerEnv, self).__init__()
        # Action Space: 0: Fold, 1: Check/Call, 2: Raise 50% Pot, 3: Raise 100% Pot, 4: All In
        self.action_space = spaces.Discrete(5)

        # Observation Space:1(position) +  52 (private cards) + 52 (public cards) + 6 (player money) + 6 * 5 * 4 (history)
        self.obs_shape = 1 + 52 + 52 + NUM_PLAYERS + NUM_PLAYERS * 5 * 4
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype=np.float32)

        # hisory obs
        self.history_len = NUM_PLAYERS * 4 # 4 actions per player
        self.history_action = []

        # Initialize clients
        self.train_client = "train_client"
        self.helper_clients = ["helper_client_" + str(i) for i in range(NUM_HELPERS)]
        self.train_pos = 0
        self.helper_pos = list(range(1, NUM_PLAYERS))
        self.players = [self.train_client] + self.helper_clients

        self.train_data = None
        self.helper_data = None
        self.episode_cnt  = 0
        self.helper_models = [None] * NUM_HELPERS
        self.env = None
        self.helper_lstm_states = [None] * NUM_HELPERS  # LSTM states for helper agents 

    def reset(self, seed=None, option=None):
        """
        Reset the environment for a new episode.
        """
        super().reset(seed=seed)
        env_logger.info("Resetting environment for a new episode.")
        self.train_pos = (1 + self.train_pos) % NUM_PLAYERS
        for i in range(NUM_HELPERS):
            self.helper_pos[i] = (self.helper_pos[i]+ 1) % NUM_PLAYERS
        self.players.insert(0, self.players.pop())
        env_logger.info(f"Training position: {self.train_pos}, Helper positions: {self.helper_pos}")
        env_logger.info(f"Players: {self.players}")

        self.env = Env(self.players)
        self.env.reset()

        if (self.episode_cnt % OPP_UPDATE_FREQ == 0):
            self.update_opponent_models()
        self.train_data = self.env.get_state(self.train_pos).json()
        obs = self._get_obs(self.train_data)
        return obs,{}

    def step(self, action):
        """
        Execute one step in the environment.
        Args:
            action (int): Action to take.
        Returns:
            obs (np.ndarray): Observation after taking the action.
            reward (float): Reward received after taking the action.
            done (bool): Whether the episode is done.
            info (dict): Additional information.
        """
        self.train_data = self.env.get_state(self.train_pos).json()

        while self.train_data['action_position'] != self.train_pos:
            action_pos = self.train_data['action_position']
            self.helper_data = self.env.get_state(action_pos).json()
            action_raw = self.helper_get_action(action_pos, self.helper_data)
            action_pased = IaAction.parse(action_raw)
            self.env.new_action(action_pased)
            if not self.env.is_over() and self.env.is_stage_over():
                self.env.new_stage()

            self.train_data = self.env.get_state(self.train_pos).json()

            done = self.env.is_over()
            if done:
                result = self.env.result.json()
                env_logger.info(f"Episode {self.episode_cnt} finished. Result: {result}")
                self.episode_cnt += 1
                obs = self._get_obs(result)
                reward = self._calculate_reward(result)
                return obs, reward, done,False, {}

        if self.train_data['action_position'] == self.train_pos:
            action_raw = action_to_actionstr(action, self.train_data)
            action_parsed = IaAction.parse(action_raw)
            self.env.new_action(action_parsed)

            if not self.env.is_over() and self.env.is_stage_over():
                self.env.new_stage()
            
            self.train_data = self.env.get_state(self.train_pos).json()
            done = self.env.is_over()
            obs = self._get_obs(self.train_data)
            reward = self._calculate_reward(self.train_data)
            if done:
                result = self.env.result.json()
                env_logger.info(f"Episode {self.episode_cnt} finished. Result: {result}")
                self.episode_cnt += 1
            

                obs = self._get_obs(result)
                reward = self._calculate_reward(result)
            return obs, reward, done,False, {}

    def _calculate_reward(self, state):
        """
        Calculate the reward based on the game state.
        Args:
            state (dict): Game state data.
        Returns:
            float: Reward for the agent.
        """
        if state['info'] == 'result':
            win_money = state['players'][self.train_pos]['win_money']
            reward = win_money / INIT_MONEY  # Normalize reward based on initial money
            return reward
        else:
            return 0.0

    

    
    def update_opponent_models(self):
        """
        Load the helper agent models.
        """
        import random
        import glob
        model_files = glob.glob(os.path.join(model_dir, 'ppo_poker_*.zip'))
        if model_files:
            for i in range(NUM_HELPERS):
                model_file = random.choice(model_files)
                self.helper_models[i] = RecurrentPPO.load(model_file)
                env_logger.info(f"Loaded helper model from {model_file} for agent {i}")
        else:
            self.helper_models = [None] * NUM_HELPERS

    def helper_get_action(self, action_pos, data):
        """
        Get action from the helper agent.
        Args:
            action_pos (int): Position of the helper agent.
            data (dict): Game state data.
        Returns:
            str: Action string for the helper agent.
        """
        idx = action_pos if action_pos < self.train_pos else action_pos - 1
        model = self.helper_models[idx]
        if model is None:
            return get_action(data)  # Fallback to default action if no model is loaded
        else:
            obs = self._get_obs(data)
            action, self.helper_lstm_states[idx] = model.predict(obs, state=self.helper_lstm_states[idx], deterministic=True)
            return action_to_actionstr(action, data)

    def _get_obs(self, state):
        """
        Convert the game state into an observation vector.
        Args:
            data (dict): Game state data.
        """
        if (state['info'] == 'result'):
            return np.zeros(self.obs_shape, dtype=np.float32)
        
        position_obs = np.array([self.train_pos], dtype=np.float32)  # Current player's position
        private_cards_obs = get_card_coding(state['private_card'])
        public_cards_obs = get_card_coding(state['public_card'])
        player_money_obs = np.array([p['money_left'] / INIT_MONEY for p in state['players']], dtype=np.float32)
        self.get_history(state)
        history_obs = list(self.history_action)
        while len(history_obs) < self.history_len:
            history_obs.append(np.zeros(5, dtype=np.float32))
        history_obs = np.array(history_obs[-self.history_len:], dtype=np.float32).flatten()

        return np.concatenate([
            position_obs,
            private_cards_obs,
            public_cards_obs,
            player_money_obs,
            history_obs
        ], dtype=np.float32)
        
    def get_history(self, state):
        """
        Update the history of actions taken in the game.
        Args:
            state (dict): Game state data.
        """
        self.history_action.clear()
        action_history = state['action_history']
        for turn in reversed(action_history):
            for item in reversed(turn):
                action_str = item['action']
                if (action_str == 'fold'):
                    self.record_history(0, 0, item['position'])
                elif (action_str == 'check' or action_str == 'call'):
                    self.record_history(1, 0, item['position'])
                elif (action_str.startswith('r')):
                    amount = int(action_str[1:])
                    self.record_history(2, amount, item['position'])
                if (self.history_len == len(self.history_action)):
                    return
    def record_history(self, action, amount, actor):
        """
        Args: 
            action (int): The action taken by the actor.
            actor (int): The player who took the action.
        History:
            action (3 floats) + amount (1 floats) + actor (1 float)
        """
        action_his = [1.0, 0.0, 0.0] if action == 0 else [0.0, 1.0, 0.0] if action == 1 else [0.0, 0.0, 1.0]
        amount_his = [amount / INIT_MONEY]
        actor_his = [actor]
        history_vec = np.array(action_his + amount_his + actor_his, dtype=np.float32)
        self.history_action.append(history_vec)