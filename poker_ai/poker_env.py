import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils import get_card_coding, sendJson, recvJson, get_action
import socket
import logging
import os

NUM_PLAYERS = 2
INIT_MONEY = 20000

server_ip = "127.0.0.1"
server_port = 8888
room_number = NUM_PLAYERS
game_number = 2

# logging.basicConfig(
#     filename='poker_env.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
log_dir = './log'
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

        # Initialize clients
        self.train_client = None
        self.helper_client = None
        self.train_data = None
        self.helper_data = None
        self.episode_cnt  = 0

    def reset(self, seed=None, option=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        if (self.train_client == None or self.train_client.fileno() == -1):
            self.connect_to_server()
        else:
            sendJson(self.train_client, {'info': 'ready', 'status': 'start'})
            sendJson(self.helper_client, {'info': 'ready', 'status': 'start'})
            self.train_data = recvJson(self.train_client)
            self.helper_data = recvJson(self.helper_client)

        obs = self._get_obs(self.train_data)
        return obs, {}

    def step(self, action):
        """
        Execute the action in the environment.
        Args:
            action(int): The action to be executed. 0 - 4
        """
        self.train_pos = self.train_data['position']
        self.helper_pos = self.helper_data['position']
        # train client action
        if self.train_data['position'] == self.train_data['action_position']:
            if action == 0:
                action_str = 'fold'
            elif action == 1:
                action_str = 'check' if 'check' in self.train_data['legal_actions'] else 'call'
            elif action == 2:
                raise_size = min(self.train_data['raise_range'][0] * 1.5, self.train_data['raise_range'][1])
                action_str = 'r' + str(int(raise_size))
            elif action == 3:
                raise_size = min(self.train_data['raise_range'][0] * 2, self.train_data['raise_range'][1])
                action_str = 'r' + str(int(raise_size))
            elif action == 4:
                raise_size = self.train_data['players'][self.train_pos]['money_left']
                action_str = 'r' + str(int(raise_size))
            env_logger.info(f"Train client action: {action_str}")
            sendJson(self.train_client, {'action': action_str, 'info': 'action'})

            self.helper_data = recvJson(self.helper_client)
            self.train_data = recvJson(self.train_client)
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

        # helper client action
        if self.helper_data['position'] == self.helper_data['action_position']:
            action_str = get_action(self.helper_data)
            env_logger.info(f"Helper client action: {action_str}")
            sendJson(self.helper_client, {'action': action_str, 'info': 'action'})

            self.helper_data = recvJson(self.helper_client)
            self.train_data = recvJson(self.train_client)
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
            return obs, reward, done, False,{}
            


    def _get_obs(self, state):
        """
        Convert the state to an observation vector.
        """
        if (state['info'] == 'result'):
            return np.zeros(self.obs_shape, dtype=np.float32)

        # state['info] = 'state
        private_cards_obs = get_card_coding(state['private_card']) 
        public_cards_obs = get_card_coding(state['public_card'])
        player_money_obs = np.array([p['money_left'] / INIT_MONEY for p in state['players']], dtype=np.float32)
        history_obs = np.zeros(45, dtype=np.float32)  # Placeholder for history

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
            reward = player_money - opponent_money
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