import numpy as np
import gym
from gym import spaces
from utils import get_card_coding
import sys
import json
import socket
import struct
from client import sendJson, recvJson, get_action

NUM_PLAYERS = 2
INIT_MONEY = 20000

server_ip = "127.0.0.1"
server_port = 2333
room_number = NUM_PLAYERS
game_number = 2



class PokerEnv(gym.Env):
    def __init__(self):
        super(PokerEnv, self).__init__()
        # Action Space: 0: Fold, 1: Check/Call, 2: Raise 50% Pot, 3: Raise 100% Pot, 4: All In
        self.action_space = spaces.Discrete(5)

        # Observation Space: 52 (private cards) + 52 (public cards) + 2 (player money) + 45 (history)
        self.obs_shape = 151
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype=np.float32)


    def reset(self):
        """
        Reset the environment to the initial state.
        """
        try:
            sendJson(self.train_client, {'info': 'ready', 'status': 'start'})
            sendJson(self.helper_client, {'info': 'ready', 'status': 'start'})
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            print("Connection error. Reconnecting...")
            self.train_client.close()
            self.helper_client.close()
            self.connect_to_server()

        obs = self._get_obs(self.train_data)
        return obs

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
            sendJson(self.train_client, {'action': action_str, 'info': 'action'})

            self.helper_data = recvJson(self.helper_client)
            self.train_data = recvJson(self.train_client)
            obs = self._get_obs(self.train_data)
            reward = self._calculate_reward(self.train_data)
            done = (self.train_data['info'] == 'result')
            return obs, reward, done, {}

        # helper client action
        if self.helper_data['position'] == self.helper_data['action_position']:
            action_str = get_action(self.helper_data)
            sendJson(self.helper_client, {'action': action_str, 'info': 'action'})

            self.helper_data = recvJson(self.helper_client)
            self.train_data = recvJson(self.train_client)
            obs = self._get_obs(self.train_data)
            reward = self._calculate_reward(self.train_data)
            done = (self.train_data['info'] == 'result')
            return obs, reward, done, {}
            


    def _get_obs(self, state):
        """
        Convert the state to an observation vector.
        """
        private_cards_obs = get_card_coding(state['private_cards'][0])  # Assuming player 0 is the agent
        public_cards_obs = get_card_coding(state['public_cards'])
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