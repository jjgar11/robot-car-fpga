import struct
from socket import socket
from multiprocessing.connection import Connection


PORT = 50007


def robot_loop(pipe: Connection):
    # open connection and wait for laptop to connect
    s = socket()
    s.bind(("", PORT))
    s.listen(1)
    conn, addr = s.accept()
    while True:
        # wait for sign detection to send data over pipe
        if pipe.poll(1):
            data = pipe.recv()
            # send data to laptop
            payload = struct.pack("!II", *data)
            conn.sendall(payload)


def laptop_loop(pipe: Connection, robot_ip):
    sockets = []
    # connect to robot
    s = socket()
    s.connect((robot_ip, PORT))
    sockets.append(s)
    while True:
        # wait for message from robot
        data = s.recv(8)
        x, y = struct.unpack("!II", data)
        # send data to main process
        pipe.send((x, y))
