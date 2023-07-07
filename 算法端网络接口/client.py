import socket

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 50001  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        # 接收输入
        # 发送数据
        a = input("请输入：")
        if a == "exit":
            break
        a = a.encode()
        s.sendall(a)
        data = s.recv(1024).decode()
        print(f"Received {data!r}")
