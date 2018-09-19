import zmq

subscribers = 3
# context of zmq
context = zmq.Context()
# socket type PUB
socket = context.socket(zmq.PUB)

port = "5555"
#socket.bind("tcp://*:5555")
socket.bind("tcp://*:" + port)

while True:
    str = input("Enter your commands: ")
    socket.send_string(str)