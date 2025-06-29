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

NUM_PLAYERS = 2
INIT_MONEY = 20000
OPP_UPDATA_FREQ = 500 # episode  

room_number = NUM_PLAYERS
game_number = 1000

# logging.basicConfig(
#     filename='poker_env.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# log_dir = './log'
model_dir = './model'
# os.makedirs(log_dir, exist_ok=True)
# os.makedirs(log_dir, exist_ok=True)
# log_path = os.path.join(log_dir, 'poker_env.log')
# with open(log_path, 'w'):
#     pass

# env_logger = logging.getLogger('env')
# env_logger.setLevel(logging.INFO)
# fh=logging.FileHandler(log_path)
# fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# env_logger.addHandler(fh)

class PokerEnv(gym.Env):
    def __init__(self):
        super(PokerEnv, self).__init__()
        # Action Space:  0:Fold, 1:Check/Call, 2:Raise 20%, 3:Raise 40%, 4:Raise 60%, 5:Raise 80%, 6:All-In
        self.action_space = spaces.Discrete(7)

        # Observation Space: 52 (private cards) + 52 (public cards) + 2 (player money) + 45 (history)
        self.obs_shape = 151
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype=np.float32)

        # hisory obs
        self.history_len = 9
        self.history_action = []

        # Initialize clients
        self.train_client = "train_client"
        self.helper_client = "helper_client"
        self.env = Env([self.train_client, self.helper_client])
        self.train_data = None
        self.helper_data = None
        self.episode_cnt  = 0
        self.helper_model = None
        self.train_pos = None

        self.helper_lstm_state = None

    def reset(self, seed=None, option=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        # env_logger.info(f'Resetting Poker Environment, episode:{self.episode_cnt}')
        # Reset clients and data
        self.env.reset()
        
        if (self.episode_cnt % OPP_UPDATA_FREQ == 0):
            # env_logger.info('Updating opponent model...')
            self.load_helper()
        
        self.train_data = self.env.get_state(0).json()
        # env_logger.info(f"Train client data: {self.train_data}")
        self.helper_data = self.env.get_state(1).json()
        obs = self._get_obs(self.train_data)
        return obs, {}

    def step(self, action):
        """
        Execute the action in the environment.
        Args:
            action(int): The action to be executed.
        """
        # helper client action
        self.train_data = self.env.get_state(0).json()
        self.helper_data = self.env.get_state(1).json()

        while self.helper_data['position'] == self.helper_data['action_position']:
            self.train_pos = self.train_data['position']
            self.helper_pos = self.helper_data['position']

            action_raw = self.helper_get_action(self.helper_data)
            action_parsed = IaAction.parse(action_raw)
            self.env.new_action(action_parsed)
            if not self.env.is_over() and self.env.is_stage_over():
                self.env.new_stage()

            # env_logger.info(f"Game state: {self.helper_data}")
            # env_logger.info(f"Helper client action: {action_raw}")
            # sendJson(self.helper_client, {'action': action_str, 'info': 'action'})

            self.train_data = self.env.get_state(0).json()
            self.helper_data = self.env.get_state(1).json()
            # env_logger.info(f'Train recieved data: {self.train_data}')

            done = self.env.is_over()
            if (done):
                result = self.env.result.json()
                # env_logger.info('win money: {},\tyour card: {},\topp card: {},\t\tpublic card: {}'.format(
                #     result['players'][0]['win_money'],
                #     result['player_card'][0],
                #     result['player_card'][1], 
                #     result['public_card']))
                # env_logger.info('Game Over, result: {}'.format(result))
                self.episode_cnt += 1

                obs = self._get_obs(result)
                reward = self._calculate_reward(result)
                return obs, reward, done, False,{}

        # train client action
        self.train_pos = self.train_data['position']
        self.helper_pos = self.helper_data['position']
        if self.train_data['position'] == self.train_data['action_position']:
            action_raw = action_to_actionstr(action, self.train_data)
            action_parsed = IaAction.parse(action_raw)
            self.env.new_action(action_parsed)
            if not self.env.is_over() and self.env.is_stage_over():
                self.env.new_stage()

            # env_logger.info(f"game state: {self.train_data}")
            # env_logger.info(f"Train client action: {action_raw}")

            self.helper_data = self.env.get_state(1).json()
            self.train_data = self.env.get_state(0).json()
            # env_logger.info(f'Train recieved data: {self.train_data}')

            done = self.env.is_over()
            obs = self._get_obs(self.train_data)
            reward = self._calculate_reward(self.train_data)
            if (done):
                result = self.env.result.json()
                # env_logger.info('Game Over, result: {}'.format(result))
                self.episode_cnt += 1

                obs = self._get_obs(result)
                reward = self._calculate_reward(result)

            return obs, reward, done, False, {}


            


    def _get_obs(self, state):
        """
        Convert the state to an observation vector.
        Observation Space: 52 (private cards) + 52 (public cards) + 2 (player money) + 45 (history)
        """
        if (state['info'] == 'result'):
            return np.zeros(self.obs_shape, dtype=np.float32)

        # state['info] = 'state
        private_cards_obs = get_card_coding(state['private_card']) 
        public_cards_obs = get_card_coding(state['public_card'])
        player_money_obs = np.array([p['money_left'] / INIT_MONEY for p in state['players']], dtype=np.float32)
        self.get_history(state)
        history_obs = list(self.history_action)
        while len(history_obs) < self.history_len:
            history_obs.append(np.zeros(5, dtype=np.float32))
        history_obs = np.array(history_obs[-self.history_len:], dtype=np.float32).flatten()

        return np.concatenate([
            private_cards_obs,
            public_cards_obs,
            player_money_obs,
            history_obs
        ]).astype(np.float32)

    def _calculate_reward(self, state):
        """
        Calculate the reward based on the game state.
        """
        if state['info'] == 'result':
            player_money = state['players'][0]['win_money']
            # normalize to [-1, 1]
            reward = (player_money)/ (INIT_MONEY * NUM_PLAYERS)
            return reward
        else:
            return 0.0


    def load_helper(self):
        """
        Load the helper model for the environment.
        """
        import glob
        import random
        model_files = glob.glob(os.path.join(model_dir, 'ppo_poker_*.zip'))
        model_files = sorted(model_files, key=os.path.getctime)[-10:]
        if (model_files):
            latest_model = random.choice(model_files)
            # env_logger.info(f"Loading helper model from {latest_model}")
            self.helper_model = RecurrentPPO.load(latest_model, env=self)
        else:
            self.helper_model = None

    def helper_get_action(self, state):
        """
        Get the action from the helper model.
        """
        if (self.helper_model is None):
            return get_action(state)
        else:
            obs = self._get_obs(state)
            action, self.helper_lstm_state = self.helper_model.predict(obs, state=self.helper_lstm_state, deterministic=True)
            return action_to_actionstr(action, state)

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


    def get_history(self, data):
        """
        Get the history of actions taken in the game.
        """
        self.history_action = []
        action_history = data['action_history']
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

# import json

# def update_model_pool(model_path, winrate, pool_path='model_pool.json', top_n=10):
#     """
#     Update the model pool with the new model.
#     Args:
#         model_path (str): Path to the new model.
#         winrate (float): Win rate of the new model.
#         pool_path (str): Path to the model pool file.
#         top_n (int): Number of top models to keep in the pool.
#     """
#     if os.path.exists(pool_path):
#         with open(pool_path, 'r') as f:
#             model_pool = json.load(f)
#     else:
#         model_pool = []

#     model_pool.append({'model_path': model_path, 'winrate': winrate})
#     model_pool = sorted(model_pool, key=lambda x: x['winrate'], reverse=True)[:top_n]

#     with open(pool_path, 'w') as f:
#         json.dump(model_pool, f, indent=4)