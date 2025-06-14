import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils import get_card_coding, sendJson, recvJson, get_action
import socket
import logging
import os
from stable_baselines3 import PPO

NUM_PLAYERS = 2
INIT_MONEY = 20000
OPP_UPDATA_FREQ = 500 # episode  

server_ip = "127.0.0.1"
server_port = 8888
room_number = NUM_PLAYERS
game_number = 1000

# logging.basicConfig(
#     filename='poker_env.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
log_dir = './log'
model_dir = './model'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'poker_env.log')
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

        # Observation Space: 52 (private cards) + 52 (public cards) + 2 (player money) + 45 (history)
        self.obs_shape = 151
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype=np.float32)

        # hisory obs
        self.history_len = 9
        self.history_action = []

        # Initialize clients
        self.train_client = None
        self.helper_client = None
        self.train_data = None
        self.helper_data = None
        self.episode_cnt  = 0
        self.helper_model = None
        self.train_pos = None

    def reset(self, seed=None, option=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        env_logger.info(f'Resetting Poker Environment, episode:{self.episode_cnt}')
        if (self.train_client == None or self.train_client.fileno() == -1):
            self.connect_to_server()
        else:
            sendJson(self.train_client, {'info': 'ready', 'status': 'start'})
            sendJson(self.helper_client, {'info': 'ready', 'status': 'start'})
            self.train_data = recvJson(self.train_client)
            # env_logger.info(f'Train recieved data: {self.train_data}')
            self.helper_data = recvJson(self.helper_client)
        
        if (self.episode_cnt % OPP_UPDATA_FREQ == 0):
            env_logger.info('Updating opponent model...')
            self.load_helper()
        obs = self._get_obs(self.train_data)
        return obs, {}

    def step(self, action):
        """
        Execute the action in the environment.
        Args:
            action(int): The action to be executed. 0 - 4
        """
        # helper client action
        while self.helper_data['position'] == self.helper_data['action_position']:
            self.train_pos = self.train_data['position']
            self.helper_pos = self.helper_data['position']

            action_str = self.helper_get_action(self.helper_data)

            env_logger.info(f"Helper client action: {action_str}")
            sendJson(self.helper_client, {'action': action_str, 'info': 'action'})

            self.helper_data = recvJson(self.helper_client)
            self.train_data = recvJson(self.train_client)
            # env_logger.info(f'Train recieved data: {self.train_data}')

            done = (self.train_data['info'] == 'result')
            if (done):
                env_logger.info('win money: {},\tyour card: {},\topp card: {},\t\tpublic card: {}'.format(
                    self.train_data['players'][self.train_pos]['win_money'], 
                    self.train_data['player_card'][self.train_pos],
                    self.train_data['player_card'][1 - self.train_pos], 
                    self.train_data['public_card']))
                self.episode_cnt += 1
                if (self.episode_cnt % game_number == 0):
                    env_logger.info('Game Turn over, closing clients...')
                    self.train_client.close()
                    self.helper_client.close()
                obs = self._get_obs(self.train_data)
                reward = self._calculate_reward(self.train_data)
                return obs, reward, done, False,{}

        # train client action
        self.train_pos = self.train_data['position']
        self.helper_pos = self.helper_data['position']
        if self.train_data['position'] == self.train_data['action_position']:
            action_str = self.action_to_actionstr(action, self.train_data)
            env_logger.info(f"Train client action: {action_str}")
            sendJson(self.train_client, {'action': action_str, 'info': 'action'})

            self.helper_data = recvJson(self.helper_client)
            self.train_data = recvJson(self.train_client)
            # env_logger.info(f'Train recieved data: {self.train_data}')

            obs = self._get_obs(self.train_data)
            reward = self._calculate_reward(self.train_data)
            done = (self.train_data['info'] == 'result')
            if (done):
                env_logger.info('win money: {},\tyour card: {},\topp card: {},\t\tpublic card: {}'.format(
                    self.train_data['players'][self.train_pos]['win_money'], 
                    self.train_data['player_card'][self.train_pos],
                    self.train_data['player_card'][1 - self.train_pos], 
                    self.train_data['public_card']))
                self.episode_cnt += 1
                if (self.episode_cnt % game_number == 0):
                    env_logger.info('Game Turn over, closing clients...')
                    self.train_client.close()
                    self.helper_client.close()
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
            player_money = state['players'][self.train_pos]['win_money']
            opponent_money = state['players'][1 - self.train_pos]['win_money']
            # normalize to [-1, 1]
            reward = (player_money - opponent_money)/ (INIT_MONEY * NUM_PLAYERS)
            return reward
        else:
            return 0.0


    def connect_to_server(self):
        """
        Connect to the poker server.
        """
        self.train_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.train_client.connect((server_ip, server_port))
        message = dict(info='connect',
                       name="train",
                       room_number=room_number,
                       game_number=game_number)
        sendJson(self.train_client, message)

        self.helper_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.helper_client.connect((server_ip, server_port))
        message = dict(info='connect',
                       name="helper",
                       room_number=room_number,
                       game_number=game_number)
        sendJson(self.helper_client, message)

        self.train_data = recvJson(self.train_client)
        self.helper_data = recvJson(self.helper_client)
        # env_logger.info(f'Train recieved data: {self.train_data}')

    def load_helper(self):
        """
        Load the helper model for the environment.
        """
        import glob
        import random
        model_files = glob.glob(os.path.join(model_dir, 'ppo_poker_*.zip'))
        if (model_files):
            # latest_model = max(model_files, key=os.path.getctime)
            latest_model = random.choice(model_files)
            env_logger.info(f"Loading helper model from {latest_model}")
            self.helper_model = PPO.load(latest_model, env=self)
        else:
            self.helper_model = None

    def helper_get_action(self, state):
        """
        Get the action from the helper model.
        """
        if (self.helper_model is None):
            self.record_history(1,0, 1) # assume check/call
            return get_action(state)
        else:
            obs = self._get_obs(state)
            action,_ = self.helper_model.predict(obs, deterministic=True)
            return self.action_to_actionstr(action, state)

    def record_history(self, action, amount, actor):
        """
        Args: 
            action (int): The action taken by the actor.
            actor (int): The player who took the action.
        """
        action_his = [1.0, 0.0, 0.0] if action == 0 else [0.0, 1.0, 0.0] if action == 1 else [0.0, 0.0, 1.0]
        amount_his = [amount / INIT_MONEY]
        actor_his = [actor]
        history_vec = np.array(action_his + amount_his + actor_his, dtype=np.float32)
        self.history_action.append(history_vec)
        if len(self.history_action) > self.history_len:
            self.history_action.pop(0)

    def action_to_actionstr(self,action, state):
        """
        Args:
            action (int): Action index (0: fold, 1: check/call, 2: raise small, 3: raise big, 4: all-in)
            state (dict): Current game state containing legal actions and raise range
        Returns:
            str: Action string representation
        """
        pos = state['position']
        money_left = state['players'][pos]['money_left']
        legal = state['legal_actions']
        raise_range = state['raise_range']

        # 默认值
        action_type = action
        amount = 0

        if action == 0:
            action_str = 'fold'
        elif action == 1:
            action_str = 'check' if 'check' in legal else 'call'
        elif action == 2:
            if 'raise' not in legal or money_left == 0:
                action_type = 1
                action_str = 'check' if 'check' in legal else 'call'
            else:
                amount = min(raise_range[0] * 1.5, raise_range[1])
                action_str = 'r' + str(int(amount))
        elif action == 3:
            if 'raise' not in legal or money_left == 0:
                action_type = 1
                action_str = 'check' if 'check' in legal else 'call'
            else:
                amount = min(raise_range[0] * 2, raise_range[1])
                action_str = 'r' + str(int(amount))
        elif action == 4:
            if 'raise' not in legal or money_left == 0:
                action_type = 1
                action_str = 'check' if 'check' in legal else 'call'
            else:
                amount = raise_range[1]
                action_str = 'r' + str(int(amount))
        else:
            raise ValueError(f"Unknown action: {action}")

        self.record_history(action_type, amount, pos)
        return action_str