from poker.host.host import Host

if __name__ == '__main__':
    host = Host(('127.0.0.1', 8888))
    host.run()
