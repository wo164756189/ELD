import cv2
import numpy as np
import util
#import pytesseract
#from pyzbar import pyzbar




if __name__ == '__main__':
    # initilize parameters
    p = util.Parameter()

    #features, products, positions = util.initImgEnv("labels.pkl")
    camera, features, products, positions = util.initCamEnv("labels.pkl")

    #product = "NV230C"
    product = "147GHN"
    #product = "182CST"
    print("[INFO] Product: " + product)

    # img = cv2.imread('test3.bmp', 1)
    img = cv2.imread('test3.bmp', 1)

    #str_status = util.detectionImg(p, img, features, products, positions, product)
    str_status = util.detectionCam(p, camera, features, products, positions, product)
    #str_status, QRCodes = util.detectionImgQR(p, img, features, products, positions, product)
    #str_status, QRCodes = detectionCamQR(p, camera, features, products, positions, product)

    print("[INFO] Detection ends")


