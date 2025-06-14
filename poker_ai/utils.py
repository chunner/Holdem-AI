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



