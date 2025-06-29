import sys
import json
import struct
import socket
from strategy import PokerStrategy

server_ip = "127.0.0.1"                 # 德州扑克平台地址
server_port =  8888                     # 德州扑克平台开放端口
room_number = int(sys.argv[1])          # 一局游戏人数
name = sys.argv[2]                      # 当前程序的 AI 名字
game_number = int(sys.argv[3])          # 最大对局数量

strategy = PokerStrategy()
def get_action(data):
    players_info = []
    for p in data['players']:
        # 计算贡献值 = 初始筹码 - 剩余筹码
        contribution = p['total_money'] - p['money_left']
        players_info.append({
            'position': p['position'],
            'money_left': p['money_left'],
            'contribution': contribution
        })
    # 只允许访问Agent应该看到的信息
    allowed_data = {
        'position': data['position'],
        'legal_actions': data['legal_actions'],
        'private_card': data['private_card'],
        'public_card': data['public_card'],
        'players': players_info,
        'raise_range': data.get('raise_range', [])
    }
    
    # 调用策略决策
    action = strategy.decide_action(allowed_data)
    print("\n" + "="*50)
    print(f"玩家: {name}")
    print(f"位置: {data['position']}")
    print(f"手牌: {data['private_card']}")
    print(f"公共牌: {data['public_card']}")
    print(f"剩余筹码: {data['players'][data['position']]['money_left']}")
    print(f"当前底池: {sum(p['contribution'] for p in players_info)}")
    print(f"可选动作: {data['legal_actions']}")
    if data.get('raise_range'):
        print(f"加注范围: {data['raise_range'][0]} - {data['raise_range'][1]}")
    print(f"决策: {action}")
    print("="*50)
    
    return action


def sendJson(request, jsonData):
    data = json.dumps(jsonData).encode()
    request.send(struct.pack('i', len(data)))
    request.sendall(data)


def recvJson(request):
    data = request.recv(4)
    if not data or len(data) < 4:
        return None  # 连接已关闭
    
    try:
        length = struct.unpack('i', data)[0]
    except struct.error:
        return None
    
    # 接收完整数据
    chunks = []
    bytes_received = 0
    while bytes_received < length:
        chunk = request.recv(min(length - bytes_received, 2048))
        if not chunk:
            break
        chunks.append(chunk)
        bytes_received += len(chunk)
    
    # 合并数据
    data = b''.join(chunks).decode('utf-8')
    if len(data) != length:
        return None
    
    return json.loads(data)


if __name__ == "__main__":
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))
    message = dict(info='connect',
                   name=name,
                   room_number=room_number,
                   game_number=game_number)
    sendJson(client, message)
    while True:
        data = recvJson(client)
        if data is None:  # 添加空数据检查
            print("服务器关闭连接")
            break
        
        if data['info'] == 'state':
            if data['position'] == data['action_position']:
                position = data['position']
                action = get_action(data)
                sendJson(client, {'action': action, 'info': 'action'})
        elif data['info'] == 'result':
            # 找到当前玩家的位置
            my_pos = None
            for i, p in enumerate(data['players']):
                if p['name'] == name:
                    my_pos = i
                    break
                
            if my_pos is not None:
                my_cards = data['player_card'][my_pos]  
                print("\n" + "="*50)
                print(f"{name}的最终结果")
                print(f"手牌: {my_cards}")
                print(f"公共牌: {data['public_card']}")
                print(f"收支: {data['players'][my_pos]['win_money']} chips")
                print("="*50)
            sendJson(client, {'info': 'ready', 'status': 'start'})
        else:
            print(data)
            break
    client.close()
