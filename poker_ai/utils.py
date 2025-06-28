import numpy as np
import json
import struct

def get_card_coding(cards):
    """
    Get the one hot encoding of the cards value
    Args:
        cards (list): List of card string (e.g., ['2s', '3d'])
    """
    VALUES = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
        'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
    }
    SUITS = ['s', 'h', 'd', 'c']
    coding = np.zeros(52, dtype=np.float32)
    for card in cards:
        value, suit = card[0], card[1]
        if value in VALUES and suit in SUITS:
            index = VALUES[value] * 4 + SUITS.index(suit)
            coding[index] = 1.0
    return coding

def get_action(data):
    if 'call' in data['legal_actions']:
        action = 'call'
    else:
        action = 'check'
    return action


def sendJson(request, jsonData):
    data = json.dumps(jsonData).encode()
    request.send(struct.pack('i', len(data)))
    request.sendall(data)


def recvJson(request):
    data = request.recv(4)
    length = struct.unpack('i', data)[0]
    data = request.recv(length).decode()
    while len(data) != length:
        data = data + request.recv(length - len(data)).decode()
    data = json.loads(data)
    return data


def action_to_actionstr(action, state):
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
    return action_str
