'''
@author: Zheng Dunyuan
@company: Graymatics
@contact: dunyuan@graymatics.com
@description: scripts for triggering the process of the Face Recognition
@time: 2019/01/30
@platform: python3.5
'''
import socket
import os
import sys
import struct
import time
HOST = "localhost"
PORT = 9999 # Port for sending the files
HEAD_STRUCT = "128sq"


def send_exit():
    sock_text = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock_text.connect((HOST, PORT))
        data = b"exit"
        sock_text.send(data)
    except socket.error as e:
        print("Socket error: %s" % str(e))
    finally:
        sock_text.close()

def send_clear():
    print("Ready?")
    sock_text = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock_text.connect((HOST, PORT))
        data = b"clear"
        sock_text.send(data)
    except socket.error as e:
        print("Socket error: %s" % str(e))
    finally:
        sock_text.close()

if __name__ == '__main__':
    send_clear()
    send_exit()
