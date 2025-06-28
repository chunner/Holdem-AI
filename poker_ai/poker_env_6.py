import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils import get_card_coding, sendJson, recvJson, get_action, action_to_actionstr
import socket
import logging
import os
from stable_baselines3 import PPO

NUM_PLAYERS = 6
NUM_HELPERS = NUM_PLAYERS - 1  # Number of helper agents
INIT_MONEY = 1000
OPP_UPDATE_FREQ = 500

server_ip = "127.0.0.1"
server_port = 8888
room_number = NUM_PLAYERS
game_number = 1000


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


class PokerEnv_6(gym.Env):
    def __init__(self):
        super(PokerEnv_6, self).__init__()
        # Action Space: 0: Fold, 1: Check/Call, 2: Raise 50% Pot, 3: Raise 100% Pot, 4: All In
        self.action_space = spaces.Discrete(5)

        # Observation Space: 52 * 6 (private cards) + 52 (public cards) + 6 (player money) + 6 * 5 * 4 (history)
        self.obs_shape = 52 * NUM_PLAYERS + 52 + NUM_PLAYERS + NUM_PLAYERS * 5 * 4
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype=np.float32)

        # hisory obs
        self.history_len = NUM_PLAYERS * 4 # 4 actions per player
        self.history_action = []

        # Initialize clients
        self.train_client = None
        self.helper_clients = [None] * NUM_HELPERS
        self.train_data = None
        self.helper_data = [None] * NUM_HELPERS
        self.episode_cnt  = 0
        self.helper_models = [None] * NUM_HELPERS
        self.train_pos = None

    def reset(self, seed=None, option=None):
        """
        Reset the environment for a new episode.
        """
        super().reset(seed=seed, option=option)
        env_logger.info("Resetting environment for a new episode.")

        # Connect to the server
        if (self.train_client == None or self.train_client.fileno()==-1):
            self.connect_to_server()
        else:
            sendJson(self.train_client, {'info': 'ready', 'status': 'start'})
            for client in self.helper_clients:
                sendJson(client, {'info': 'ready', 'status' : 'start'} )
            self.recv_data()

        if (self.episode_cnt % OPP_UPDATE_FREQ == 0):
            self.update_opponent_models()

        obs = self._get_obs(self.train_data)
        return obs,{}

    def connect_to_server(self):
        """
        Connect to the poker server and initialize clients.
        """
        self.train_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.train_client.connect((server_ip, server_port))
        message = dict(info='connect',
                       name='train_agent',
                       room_number=room_number,
                       game_number=game_number)
        sendJson(self.train_client, message)

        for i in range(NUM_HELPERS):
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((server_ip, server_port))
            self.helper_clients[i] = client
            message = dict(info='connect',
                           name=f'helper_agent_{i}',
                           room_number=room_number,
                           game_number=game_number)
            sendJson(client, message)
        
        self.recv_data()

    def recv_data(self):
        """
        Receive data from the server.
        """
        self.train_data = recvJson(self.train_client)
        for i in range(NUM_HELPERS):
            self.helper_data[i] = recvJson(self.helper_clients[i])
    
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
                self.helper_models[i] = PPO.load(model_file)
                env_logger.info(f"Loaded helper model from {model_file} for agent {i}")
        else:
            self.helper_models = [None] * NUM_HELPERS

    def _get_obs(self, state):
        """
        Convert the game state into an observation vector.
        Args:
            data (dict): Game state data.
        """
        if (state['info'] == 'result'):
            return np.zeros(self.obs_shape, dtype=np.float32)
        
        privata