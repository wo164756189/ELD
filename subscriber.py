import zmq
import util
#import demo
import base64
import cv2
import time

def main():
    #  Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.REP)

    port = "5555"
    # socket.connect("tcp://localhost:5555")
    socket.bind("tcp://*:" + port)
    print("Connected to server")

    # initilize parameters + Env
    p = util.Parameter()
    #features, products, positions = util.initImgEnv("labels.pkl")
    #img = cv2.imread('test3.bmp', 1)
    camera, features, products, positions = util.initCamEnv("labels.pkl")

    while True:
        recv = socket.recv_string().split("#***#")
        if recv[0] == "cap":
            str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            byte_time = str_time.encode(encoding='utf-8')
            #product = "NV230C"
            #product = "147GHN"
            #product = "182CST"
            product = recv[2]
            print("[INFO] Product: " + product)
            #str_status = util.detectionImg(p, img, features, products, positions, product)
            #str_status, QRCodes = util.detectionImgQR(p, img, features, products, positions, product)
            #str_status = util.detectionCam(p, camera, features, products, positions, product)
            str_status, QRCodes = util.detectionCamQR(p, camera, features, products, positions, product)
            if recv[1]==QRCodes[0] and len(QRCodes)>0:
            #if recv[1]==QRCodes[0]:
                str_status="Matching"
            else:
                str_status="Not matching"
            byte_status = str_status.encode(encoding='utf-8')
            f = open("result.jpg", 'rb')
            bytes = bytearray(f.read())
            byte_img = base64.b64encode(bytes)
            f.close()
            #info = byte_time + b"#***#" + b"1" + b"#***#" + byte_statusCode + b"#***#"+ byte_status + b"#***#" + byte_img
            info = byte_status + b"#***#" + byte_img
            #print(byte_time + b"#***#" + b"1" + b"#***#" + byte_statusCode + b"#***#"+ byte_status)
            #info = byte_status
            print(byte_status)
            socket.send(info)
            continue
        elif recv[0] == "img":
            str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            byte_time = str_time.encode(encoding='utf-8')
            product = "NV230C"
            #product = "147GHN"
            #product = "182CST"
            print("[INFO] Product: " + product)
            #str_status = demo.detection(p, camera, labels, products, product)
            img = cv2.imread('test (19).jpg', 1)
            str_status, str_statusCode = util.detectionImg(p, img, labels, products, product)
            byte_status = str_status.encode(encoding='utf-8')
            byte_statusCode = str_statusCode.encode(encoding='utf-8')
            f = open("result.jpg", 'rb')
            bytes = bytearray(f.read())
            byte_img = base64.b64encode(bytes)
            f.close()
            #info = byte_time + b"#***#" + b"1" + b"#***#" + byte_statusCode + b"#***#"+ byte_status + b"#***#" + byte_img
            #print(byte_time + b"#***#" + b"1" + b"#***#" + byte_statusCode + b"#***#"+ byte_status)
            info = byte_status
            print(byte_status)
            socket.send(info)
            continue
        elif recv[0] == "get_tem_width":
            socket.send(str(p.raw_template_width).encode(encoding='utf-8'))
            print("sending threshold to client")
        elif recv[0] == "get_tem_height":
            socket.send(str(p.raw_template_height).encode(encoding='utf-8'))
            print("sending threshold to client")
        elif recv[0] == "get_img_resize":
            socket.send(str(p.resize_img).encode(encoding='utf-8'))
            print("sending threshold to client")
        elif recv[0] == "get_tem_resize":
            socket.send(str(p.resize_tem).encode(encoding='utf-8'))
            print("sending threshold to client")
        elif recv[0] == "get_crop_resize":
            socket.send(str(p.resize_crop).encode(encoding='utf-8'))
            print("sending threshold to client")
        elif recv[0] == "get_first_match":
            socket.send(str(p.first_match_count).encode(encoding='utf-8'))
            print("sending threshold to client")
        elif recv[0] == "get_second_match":
            socket.send(str(p.second_match_count).encode(encoding='utf-8'))
            print("sending threshold to client")
        elif recv[0] == "get_threshold":
            socket.send(str(p.score_thresh).encode(encoding='utf-8'))
            print("sending threshold to client")
        elif recv[0] == "get_min_size":
            socket.send(str(p.min_size).encode(encoding='utf-8'))
            print("sending threshold to client")
#############################    set    #########################################
        elif recv[0] == "set_tem_width":
            try:
                p.raw_template_width = int(recv[1])
                socket.send(str(p.raw_template_width).encode(encoding='utf-8'))
                print("changing para in client")
            except:
                socket.send_str("failed")
        elif recv[0] == "set_tem_height":
            try:
                p.raw_template_height = int(recv[1])
                socket.send(str(p.raw_template_height).encode(encoding='utf-8'))
                print("changing para in client")
            except:
                socket.send_str("failed")
        elif recv[0] == "set_img_resize":
            try:
                p.resize_img = float(recv[1])
                socket.send(str(p.resize_img).encode(encoding='utf-8'))
                print("changing para in client")
            except:
                socket.send_str("failed")
        elif recv[0] == "set_tem_resize":
            try:
                p.resize_tem = float(recv[1])
                socket.send(str(p.resize_tem).encode(encoding='utf-8'))
                print("changing para in client")
            except:
                socket.send_str("failed")
        elif recv[0] == "set_crop_resize":
            try:
                p.resize_crop = float(recv[1])
                socket.send(str(p.resize_crop).encode(encoding='utf-8'))
                print("changing para in client")
            except:
                socket.send_str("failed")
        elif recv[0] == "set_first_match":
            try:
                p.first_match_count = int(recv[1])
                socket.send(str(p.first_match_count).encode(encoding='utf-8'))
                print("changing para in client")
            except:
                socket.send_str("failed")
        elif recv[0] == "set_second_match":
            try:
                p.second_match_count = int(recv[1])
                socket.send(str(p.second_match_count).encode(encoding='utf-8'))
                print("changing para in client")
            except:
                socket.send_str("failed")
        elif recv[0] == "set_threshold":
            try:
                p.score_thresh = int(recv[1])
                socket.send(str(p.score_thresh).encode(encoding='utf-8'))
                print("changing para in client")
            except:
                socket.send_str("failed")
        elif recv[0] == "set_min_size":
            try:
                p.min_size = int(recv[1])
                socket.send(str(p.min_size).encode(encoding='utf-8'))
                print("changing para in client")
            except:
                socket.send_str("failed")
        elif recv[0] == "set":
            print("change parameters")
        elif recv[0] == "exit":
            break
        else:
            print("Other Commands: " + recv[0])

if __name__ == '__main__':
    main()


