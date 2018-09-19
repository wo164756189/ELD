import zmq
import util
import demo
import base64
import cv2
import time

def main():
    #  Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    syncclient = context.socket(zmq.REQ)

    #ip = "localhost"
    ip = "192.168.21.103"
    portSUB = "5561"
    portREQ = "5562"
    # socket.connect("tcp://localhost:5555")
    socket.connect("tcp://" + ip + ":" + portSUB)
    syncclient.connect("tcp://" + ip + ":" + portREQ)
    print("Connected to server")

    # receive message started with ''
    socket.setsockopt_string(zmq.SUBSCRIBE, '')

    # initilize parameters + Env
    p = util.Parameter()
    camera, labels, products = util.initCamEnv("labels.pkl")

    while True:
        str = socket.recv_string()
        if str == "CAP":
            str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            byte_time = str_time.encode(encoding='utf-8')
            # product = "NV230C"
            product = "147GHN"
            #product = "182CST"
            print("[INFO] Product: " + product)
            #str_status = demo.detection(p, camera, labels, products, product)
            str_status, str_statusCode = demo.detectionCam(p, camera, labels, products, product)
            byte_status = str_status.encode(encoding='utf-8')
            byte_statusCode = str_statusCode.encode(encoding='utf-8')
            f = open("result.jpg", 'rb')
            bytes = bytearray(f.read())
            byte_img = base64.b64encode(bytes)
            f.close()
            info = byte_time + b"#***#" + b"1" + b"#***#" + byte_statusCode + b"#***#"+ byte_status + b"#***#" + byte_img
            print(byte_time + b"#***#" + b"1" + b"#***#" + byte_statusCode + b"#***#"+ byte_status)
            syncclient.send(info)
            syncclient.recv()
            continue
        elif str == "img":
            str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            byte_time = str_time.encode(encoding='utf-8')
            product = "NV230C"
            #product = "147GHN"
            #product = "182CST"
            print("[INFO] Product: " + product)
            #str_status = demo.detection(p, camera, labels, products, product)
            img = cv2.imread('test (19).jpg', 1)
            str_status, str_statusCode = demo.detectionImg(p, img, labels, products, product)
            byte_status = str_status.encode(encoding='utf-8')
            byte_statusCode = str_statusCode.encode(encoding='utf-8')
            f = open("result.jpg", 'rb')
            bytes = bytearray(f.read())
            byte_img = base64.b64encode(bytes)
            f.close()
            info = byte_time + b"#***#" + b"1" + b"#***#" + byte_statusCode + b"#***#"+ byte_status + b"#***#" + byte_img
            print(byte_time + b"#***#" + b"1" + b"#***#" + byte_statusCode + b"#***#"+ byte_status)
            syncclient.send(info)
            syncclient.recv()
            continue
        elif str == "para":
            print("change parameters")
        elif str == "exit":
            break
        else:
            print("Other Commands: " + str)

if __name__ == '__main__':
    main()


