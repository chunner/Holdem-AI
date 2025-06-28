import sys
import json
import struct
import socket

server_ip = "127.0.0.1"                 # 德州扑克平台地址
server_port = 8888                      # 德州扑克平台开放端口
room_number = int(sys.argv[1])          # 一局游戏人数
name = sys.argv[2]                      # 当前程序的 AI 名字
game_number = int(sys.argv[3])          # 最大对局数量


model_path = './poker_ai/model/ppo_poker_final.zip'
from sb3_contrib import RecurrentPPO
from poker_ai.poker_env import PokerEnv
from poker_ai.utils import action_to_actionstr

print("Loading model from:", model_path)
model  = RecurrentPPO.load(model_path)
print("Model loaded successfully.")
env = PokerEnv()
print("Environment initialized.")

lstm_state = None
def get_action(data):
    obs = env._get_obs(data)
    action, lstm_state = model.predict(obs, state=lstm_state, deterministic=True)
    return action_to_actionstr(action, data)




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


if __name__ == "__main__":
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))
    message = dict(info='connect',
                   name=name,
                   room_number=room_number,
                   game_number=game_number)
    print('send data: {}'.format(message))
    sendJson(client, message)
    print('connect to server: {}, port: {}'.format(server_ip, server_port))
    while True:
        data = recvJson(client)
        print('receive data: {}'.format(data))
        if data['info'] == 'state':
            if data['position'] == data['action_position']:
                position = data['position']
                action = get_action(data)
                print('action: {}'.format(action))
                sendJson(client, {'action': action, 'info': 'action'})
        elif data['info'] == 'result':
            print('win money: {},\tyour card: {},\topp card: {},\t\tpublic card: {}'.format(
                data['players'][position]['win_money'], data['player_card'][position],
                data['player_card'][1 - position], data['public_card']))
            lstm_state = None
            sendJson(client, {'info': 'ready', 'status': 'start'})
        else:
            print(data)
            break
    client.close()
