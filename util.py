import pypylon.pylon as py
import cv2
import numpy as np
import pickle
from imutils import paths
import os
from pyzbar import pyzbar

# matching
def bruteForce(descriptor1, descriptor2, factor):
    bf = cv2.BFMatcher()
    try:
        matches = bf.knnMatch(descriptor1, descriptor2, k=2)  # k best matches; k=2 -> 2 DMatch in each list
                                                              # queryIdx for keypoint index in descriptor1; trainIdx for keypoint index in descriptor2
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < factor * n.distance:
                good.append(m)
    except:
        print("[INFO] Fail to perform bruteForce matching.")
        exit()
    return good

# RootSIFT
class RootSIFT:
    def __init__(self):
        # initialize the SIFT feature extractor
        self.sift = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, eps=1e-7):
        # compute SIFT descriptors
        kps, descs = self.sift.detectAndCompute(image, None)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            print("[ERROR] keypoints = 0; Can not generate product feature.")
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        # descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)

        # return a tuple of the keypoints and descriptors
        return kps, descs

# feature
def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        i = i + 1
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

# get score
def getScore(points, width, height):
    score = np.int32(1000000 * len(points) / width / height)
    return  score

# parameters
class Parameter:
    def __init__(self, template_width = 3024, template_height = 4032, resize_img = 0.1, resize_crop = 0.2,
                 resize_tem=0.2, first_matching_level = 0.8, second_matching_level = 0.8, first_match_count = 40,
                 score_thresh=160, second_match_count=150, min_size = 100, stretch_h = 500, stretch_w = 300):
        self.raw_template_width = template_width
        self.raw_template_height = template_height
        self.template_width = int(template_width * resize_tem)
        self.template_height = int(template_height * resize_tem)
        self.resize_img = resize_img
        self.resize_crop = resize_crop
        self.resize_tem = resize_tem
        self.first_matching_level = first_matching_level
        self.second_matching_level = second_matching_level
        self.first_match_count = first_match_count
        self.second_match_count = second_match_count
        self.score_thresh = score_thresh
        self.min_size = min_size
        self.stretch_h = stretch_h
        self.stretch_w = stretch_w

# basler camera
def initilizeCamera():
    first_device = py.TlFactory.GetInstance().CreateFirstDevice()
    instant_camera = py.InstantCamera(first_device)
    instant_camera.Open()
    instant_camera.PixelFormat = "YUV422_YUYV_Packed"
    return instant_camera

def getImgFromCamera(camera, timeDelay=4000, type = cv2.COLOR_YUV2BGR_YUYV):
    grab_result = camera.GrabOne(timeDelay)
    image = grab_result.Array
    img = cv2.cvtColor(image, type)
    return img

# label + feature data
def initLabel(file):
    # Get feature from file
    # data = pickle.loads(open("labels.pkl", "rb").read())
    data = pickle.loads(open(file, "rb").read())
    features = data["feature"]
    products = data["product"]
    print("[INFO] Label data loaded.")
    return features, products

# label + feature + position data
def initLabelEx(file):
    # Get feature from file
    # data = pickle.loads(open("labels.pkl", "rb").read())
    data = pickle.loads(open(file, "rb").read())
    features = data["feature"]
    products = data["product"]
    positions = data["position"]
    print("[INFO] Label data loaded.")
    return features, products, positions

# Initilize env: init camera + init feature

def initCamEnv(file):
    camera = initilizeCamera()
    features, products, positions = initLabelEx(file)
    return camera, features, products, positions

def initImgEnv(file):
    features, products, positions = initLabelEx(file)
    return features, products, positions

# encode features
def encodeFeature(resize_factor = 0.2):
    print("[INFO] extracting label feature ...")
    imagePaths = list(paths.list_images("dataset"))
    # initialize the list of known encodings and known names
    features = []
    products = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing label {}/{}".format(i + 1, len(imagePaths)))
        product = imagePath.split(os.path.sep)[-1].split('.')[-2]
        products.append(product)
        img = cv2.imread(imagePath, 1)  # color image
        img_row, img_col, img_channel = img.shape
        # resize image to reduce processing time
        resize_img = cv2.resize(img, (int(img_col * resize_factor), int(img_row * resize_factor)),
                                interpolation=cv2.INTER_AREA)
        rs = RootSIFT()
        kp, des = rs.compute(resize_img)
        feature = pickle_keypoints(kp, des)
        features.append(feature)

    if features.__len__() == products.__len__():
        # dump the label features + products + positions to disk
        print("[INFO] saving feature data ...")
        data = {"feature": features, "product": products}
        pickle.dump(data, open("labels.pkl", "wb"))
        print("[INFO] Generate template database sucessfully!")
    else:
        print("[Error] Failed to generate feature file. ")

def encodeFeaturePos(resize_factor = 0.2):
    print("[INFO] extracting label feature ...")
    imagePaths = list(paths.list_images("dataset"))
    # initialize the list of known encodings and known names
    features = []
    products = []
    positions = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        product = imagePath.split(os.path.sep)[-1].split('.')[-2]
        products.append(product)
        # search for matching points
        for line in open("./dataset/pos.txt"):
            line_info = line.split(',')
            product_info = line_info[0]
            if product_info == product:
                pul = (int(line_info[1]), int(line_info[2]))  # point of upper left
                pbd = (int(line_info[3]), int(line_info[4]))  # point of bottom right
                position = (pul, pbd)
                positions.append(position)

                print("[INFO] processing label {}/{}".format(i + 1, len(imagePaths)))
                img = cv2.imread(imagePath, 1)  # color image
                img = img[pul[1]: pbd[1], pul[0]: pbd[0]]
                img_row, img_col, img_channel = img.shape
                # resize image to reduce processing time
                resize_img = cv2.resize(img, (int(img_col * resize_factor), int(img_row * resize_factor)),
                                        interpolation=cv2.INTER_AREA)
                rs = RootSIFT()
                kp, des = rs.compute(resize_img)
                feature = pickle_keypoints(kp, des)
                features.append(feature)
    if positions.__len__() == products.__len__():
        print("[INFO] Grasp position successfully")
        # dump the label features + products + positions to disk
        print("[INFO] saving feature data ...")
        data = {"feature": features, "product": products, "position": positions}
        pickle.dump(data, open("labels.pkl", "wb"))
        print("[INFO] Generate template database sucessfully!")
    else:
        print("[Error] Failed to grasp position. ")

def getProduct(features, products, product):
    try:
        index = products.index(product)
        label = features[index]
        kp_tmp, des_tmp = unpickle_keypoints(label)
        print("[INFO] Product is included in .pkl file")
    except:
        print("[ERROR] Product is not included in .pkl file!")
        exit()
    return kp_tmp, des_tmp

def getProductEx(features, products, positions, product):
    try:
        index = products.index(product)
        label = features[index]
        position = positions[index]
        kp_tmp, des_tmp = unpickle_keypoints(label)
        print("[INFO] Product is included in .pkl file")
    except:
        print("[ERROR] Product is not included in .pkl file!")
        exit()
    return position, kp_tmp, des_tmp

# QR code reader
def codeReader(img_path):
    image = cv2.imread(img_path)
    QRCodes = []
    # find the barcodes in the image and decode each of the barcodes
    barcodes = pyzbar.decode(image)

    # loop over the detected barcodes
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        # draw the barcode data and barcode type on the image
        # text = "{} ({})".format(barcodeData, barcodeType)
        text = "{}".format(barcodeData)
        #print("[INFO] " + text)
        QRCodes.append(text)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
    cv2.imshow("QR code image", image)
    return QRCodes

# Detection
# RootSift + Load Feature + ROI detection + BF matcher + single captured image
# Dependency: rootsift.py + feature.py

def detectionCam(p, camera, features, products, positions, product):
    # get product feature & position
    position, kp_tmp, des_tmp = getProductEx(features, products, positions, product)
    # Grab a single camera image
    captured_img = getImgFromCamera(camera)
    # Resize captured image: default 0.1
    cap_rows, cap_cols, cap_channels = captured_img.shape
    cap_scaled = cv2.resize(captured_img, (int(cap_cols * p.resize_img), int(cap_rows * p.resize_img)),
                            interpolation=cv2.INTER_AREA)
    # find the keypoints and descriptors with RootSIFT
    rs = RootSIFT()
    kp_cap_scaled, des_cap_scaled = rs.compute(cap_scaled)
    # first BF matching
    first_matching_points = bruteForce(des_tmp, des_cap_scaled, p.first_matching_level)
    print("[INFO] Good matching pairs of first match: " + str(len(first_matching_points)))
    # first_match_count: default = 60
    if len(first_matching_points) >= p.first_match_count:
        src_pts = np.float32([kp_tmp[m.queryIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_cap_scaled[m.trainIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except:
            print("[WARNING] Fail to perform homography.")

        h = int((position[1][1] - position[0][1] + 1) * p.resize_crop)
        w = int((position[1][0] - position[0][0] + 1) * p.resize_crop)
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        max_index = np.max(np.int32(dst / p.resize_img), axis=0)
        min_index = np.min(np.int32(dst / p.resize_img), axis=0)
        #  top-left point:       (x,y) -> (min_index[0, 0], min_index[0, 1])
        #  bottom-right point:   (x,y) -> (max_index[0, 0], max_index[0, 1])
        width = abs(min_index[0, 1] - max_index[0, 1])
        height = abs(min_index[0, 0] - max_index[0, 0])
        if min_index[0, 0] >= 0 and max_index[0, 0] < cap_cols and min_index[0, 1] >= 0 and max_index[
            0, 1] < cap_rows and width >= p.min_size and height >= p.min_size:
            # height from min_index[0, 1] to max_index[0, 1];
            # width  from min_index[0, 0] to max_index[0, 0];
            cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]

            crop_rows, crop_cols, crop_channels = cap_crop.shape
            cap_crop = cv2.resize(cap_crop, (int(crop_cols * p.resize_crop), int(crop_rows * p.resize_crop)),
                                  interpolation=cv2.INTER_AREA)
            #cv2.imshow("cropped image", cap_crop)

            dst_resize = np.float32(dst / p.resize_img)
            stretch_points = np.float32([[0, 0], [0, p.stretch_h], [p.stretch_w, p.stretch_h], [p.stretch_w, 0]])
            M_perspective = cv2.getPerspectiveTransform(dst_resize, stretch_points)
            img_perspective = cv2.warpPerspective(captured_img, M_perspective, (p.stretch_w, p.stretch_h))
            cv2.imshow("stretch image", img_perspective)
            print("[INFO] resolution of croped image: " + str(cap_crop.shape[0]) + " * " + str(cap_crop.shape[1]))
            kp_cap_crop, des_cap_crop = rs.compute(cap_crop)

            try:
                second_matching_points = bruteForce(des_tmp, des_cap_crop, p.second_matching_level)
            except:
                #exit()
                print("[INFO] Not matching: second matching failed.")
                status = "Not matching: second matching failed"
                captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                        interpolation=cv2.INTER_AREA)
                cv2.imwrite('result.jpg', captured_img)
                return status

            # img2 = cv.polylines(cap_scaled,[np.int32(dst)],True,255,3, cv.LINE_AA)
            print("[INFO] Good matching pairs of second match: " + str(len(second_matching_points)))
            score = getScore(second_matching_points, width, height)
            print("[INFO] score = " + str(score))
            if score >= 1000:
                status = "Not matching: too large score"
                print("[INFO] " + status)
            elif score >= p.score_thresh and len(second_matching_points) >= p.second_match_count:
                print("[INFO] Matching!!!")
                status = "Matching"
                captured_img = cv2.polylines(captured_img, [np.int32(dst / p.resize_img)], True, (0,255,0), 13, cv2.LINE_AA)
                captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                        interpolation=cv2.INTER_AREA)
                cv2.imshow("results", captured_img)
                cv2.imwrite('result.jpg', captured_img)
            else:
                print("[INFO] Not matching: <threshold")
                status = "Not matching: score <threshold"
                captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                        interpolation=cv2.INTER_AREA)
                cv2.imshow("results", captured_img)
                cv2.imwrite('result.jpg', captured_img)
        elif (width < p.min_size and width > 0) or (height < p.min_size and height > 0):
            print("[INFO] Not matching: too small size !")
            status = "Not matching: ROI too small"
            #cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]
            # print("resolution of croped image= " + str(max_index[0, 0] - min_index[0, 0]) + "*" + str(max_index[0, 1] - min_index[0, 1]) + "(h * w)")
            # cv2.imshow("cropped image", cap_crop)
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imshow("results", captured_img)
            cv2.imwrite('result.jpg', captured_img)
        else:
            print("[INFO] Not matching: ROI out of range !")
            status = "Not matching: ROI out of range"
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imshow("results", captured_img)
            cv2.imwrite('result.jpg', captured_img)
    else:
        print("[INFO] Not matching: first matching pairs < required !")
        status = "Not matching: first matching pairs < required"
        captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                  interpolation=cv2.INTER_AREA)
        cv2.imshow("results", captured_img)
        cv2.imwrite('result.jpg', captured_img)
    #cv2.waitKey(2000)
    cv2.waitKey()
    return status

def detectionImg(p, captured_img, features, products, positions, product):
    # get product feature & position
    position, kp_tmp, des_tmp = getProductEx(features, products, positions, product)
    # Resize captured image: default 0.1
    cap_rows, cap_cols, cap_channels = captured_img.shape
    cap_scaled = cv2.resize(captured_img, (int(cap_cols * p.resize_img), int(cap_rows * p.resize_img)),
                            interpolation=cv2.INTER_AREA)
    # find the keypoints and descriptors with RootSIFT
    rs = RootSIFT()
    kp_cap_scaled, des_cap_scaled = rs.compute(cap_scaled)
    # first BF matching
    first_matching_points = bruteForce(des_tmp, des_cap_scaled, p.first_matching_level)
    print("[INFO] Good matching pairs of first match: " + str(len(first_matching_points)))
    # first_match_count: default = 60
    if len(first_matching_points) >= p.first_match_count:
        src_pts = np.float32([kp_tmp[m.queryIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_cap_scaled[m.trainIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except:
            print("[WARNING] Fail to perform homography.")
        #matchesMask = mask.ravel().tolist()
        # h, w = template_img.shape
        h = int((position[1][1] - position[0][1] + 1) * p.resize_crop)
        w = int((position[1][0] - position[0][0] + 1) * p.resize_crop)
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        max_index = np.max(np.int32(dst / p.resize_img), axis=0)
        min_index = np.min(np.int32(dst / p.resize_img), axis=0)
        #  top-left point:       (x,y) -> (min_index[0, 0], min_index[0, 1])
        #  bottom-right point:   (x,y) -> (max_index[0, 0], max_index[0, 1])
        width = abs(min_index[0, 1] - max_index[0, 1])
        height = abs(min_index[0, 0] - max_index[0, 0])
        if min_index[0, 0] >= 0 and max_index[0, 0] < cap_cols and min_index[0, 1] >= 0 and max_index[
            0, 1] < cap_rows and width >= p.min_size and height >= p.min_size:
            # height from min_index[0, 1] to max_index[0, 1];
            # width  from min_index[0, 0] to max_index[0, 0];
            cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]

            crop_rows, crop_cols, crop_channels = cap_crop.shape
            cap_crop = cv2.resize(cap_crop, (int(crop_cols * p.resize_crop), int(crop_rows * p.resize_crop)),
                                  interpolation=cv2.INTER_AREA)
            #cv2.imshow("cropped image", cap_crop)

            dst_resize = np.float32(dst / p.resize_img)
            stretch_points = np.float32([[0, 0], [0, p.stretch_h], [p.stretch_w, p.stretch_h], [p.stretch_w, 0]])
            M_perspective = cv2.getPerspectiveTransform(dst_resize, stretch_points)
            img_perspective = cv2.warpPerspective(captured_img, M_perspective, (p.stretch_w, p.stretch_h))
            cv2.imshow("stretch image", img_perspective)
            print("[INFO] resolution of croped image: " + str(cap_crop.shape[0]) + " * " + str(cap_crop.shape[1]))
            kp_cap_crop, des_cap_crop = rs.compute(cap_crop)

            try:
                second_matching_points = bruteForce(des_tmp, des_cap_crop, p.second_matching_level)
            except:
                #exit()
                status = "Not matching: second matching failed"
                print("[INFO] " + status)
                captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                        interpolation=cv2.INTER_AREA)
                cv2.imwrite('result.jpg', captured_img)
                return status

            # img2 = cv.polylines(cap_scaled,[np.int32(dst)],True,255,3, cv.LINE_AA)
            print("[INFO] Good matching pairs of second match: " + str(len(second_matching_points)))
            score = getScore(second_matching_points, width, height)
            print("[INFO] score = " + str(score))
            if score >= 1000:
                status = "Not matching: too large score"
                print("[INFO] " + status)
            elif score >= p.score_thresh and len(second_matching_points) >= p.second_match_count:
                status = "Matching"
                print("[INFO] " + status)
                captured_img = cv2.polylines(captured_img, [np.int32(dst / p.resize_img)], True, (0,255,0), 13, cv2.LINE_AA)
                captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                        interpolation=cv2.INTER_AREA)
                cv2.imshow("results", captured_img)
                cv2.imwrite('result.jpg', captured_img)
            else:
                status = "Not matching: score <threshold"
                print("[INFO] " + status)
                captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                        interpolation=cv2.INTER_AREA)
                cv2.imshow("results", captured_img)
                cv2.imwrite('result.jpg', captured_img)
        elif (width < p.min_size and width > 0) or (height < p.min_size and height > 0):
            status = "Not matching: ROI too small"
            print("[INFO] " + status)
            #cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]
            # print("resolution of croped image= " + str(max_index[0, 0] - min_index[0, 0]) + "*" + str(max_index[0, 1] - min_index[0, 1]) + "(h * w)")
            # cv2.imshow("cropped image", cap_crop)
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imshow("results", captured_img)
            cv2.imwrite('result.jpg', captured_img)
        else:
            status = "Not matching: ROI out of range"
            print("[INFO] " + status)
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imshow("results", captured_img)
            cv2.imwrite('result.jpg', captured_img)
    else:
        status = "Not matching: first matching pairs < required"
        print("[INFO] " + status)
        #cv2.imshow("results", captured_img)
        captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                  interpolation=cv2.INTER_AREA)
        cv2.imshow("results", captured_img)
        cv2.imwrite('result.jpg', captured_img)
    # cv2.waitKey(2000)
    cv2.waitKey()
    return status

def detectionCamQR(p, camera, features, products, positions, product):
    # get product feature & position
    position, kp_tmp, des_tmp = getProductEx(features, products, positions, product)
    # Grab a single camera image
    captured_img = getImgFromCamera(camera)
    # Resize captured image: default 0.1
    cap_rows, cap_cols, cap_channels = captured_img.shape
    cap_scaled = cv2.resize(captured_img, (int(cap_cols * p.resize_img), int(cap_rows * p.resize_img)),
                            interpolation=cv2.INTER_AREA)
    # find the keypoints and descriptors with RootSIFT
    rs = RootSIFT()
    kp_cap_scaled, des_cap_scaled = rs.compute(cap_scaled)
    # first BF matching
    first_matching_points = bruteForce(des_tmp, des_cap_scaled, p.first_matching_level)
    print("[INFO] Good matching pairs: " + str(len(first_matching_points)))
    # first_match_count: default = 60
    if len(first_matching_points) >= p.first_match_count:
        src_pts = np.float32([kp_tmp[m.queryIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_cap_scaled[m.trainIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except:
            print("[WARNING] Fail to perform homography.")

        h = int((position[1][1] - position[0][1] + 1) * p.resize_crop)
        w = int((position[1][0] - position[0][0] + 1) * p.resize_crop)
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        max_index = np.max(np.int32(dst / p.resize_img), axis=0)
        min_index = np.min(np.int32(dst / p.resize_img), axis=0)
        #  top-left point:       (x,y) -> (min_index[0, 0], min_index[0, 1])
        #  bottom-right point:   (x,y) -> (max_index[0, 0], max_index[0, 1])
        width = abs(min_index[0, 1] - max_index[0, 1])
        height = abs(min_index[0, 0] - max_index[0, 0])
        if min_index[0, 0] >= 0 and max_index[0, 0] < cap_cols and min_index[0, 1] >= 0 and max_index[
            0, 1] < cap_rows and width >= p.min_size and height >= p.min_size:
            # height from min_index[0, 1] to max_index[0, 1];
            # width  from min_index[0, 0] to max_index[0, 0];
            cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]

            crop_rows, crop_cols, crop_channels = cap_crop.shape
            cap_crop = cv2.resize(cap_crop, (int(crop_cols * p.resize_crop), int(crop_rows * p.resize_crop)),
                                  interpolation=cv2.INTER_AREA)
            #cv2.imshow("cropped image", cap_crop)

            dst_resize = np.float32(dst / p.resize_img)
            stretch_points = np.float32([[0, 0], [0, p.stretch_h], [p.stretch_w, p.stretch_h], [p.stretch_w, 0]])
            M_perspective = cv2.getPerspectiveTransform(dst_resize, stretch_points)
            img_perspective = cv2.warpPerspective(captured_img, M_perspective, (p.stretch_w, p.stretch_h))
            cv2.imshow("stretch image", img_perspective)
            print("[INFO] resolution of croped image: " + str(cap_crop.shape[0]) + " * " + str(cap_crop.shape[1]))
            # QR code
            img_perspective = img_perspective[int(p.stretch_h*0.1):int(p.stretch_h*0.5), int(p.stretch_w*0.5):int(p.stretch_w)]
            #img_perspective = img_perspective[50:250, 150:300]
            cv2.imwrite("stretch_image_QR.jpg", img_perspective)
            #cv2.imshow("OCR region", img_perspective)
            QRCodes = codeReader('./stretch_image_QR.jpg')
            if len(QRCodes) > 0:
                status = "QR code detected."
                print(QRCodes)
            else:
                status = "No QR code is detected."
            print("[INFO] " + status)

        elif (width < p.min_size and width > 0) or (height < p.min_size and height > 0):
            print("[INFO] Not matching: too small size !")
            status = "Not matching: ROI too small"
            #cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]
            # print("resolution of croped image= " + str(max_index[0, 0] - min_index[0, 0]) + "*" + str(max_index[0, 1] - min_index[0, 1]) + "(h * w)")
            # cv2.imshow("cropped image", cap_crop)
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imshow("results", captured_img)
            cv2.imwrite('result.jpg', captured_img)
        else:
            print("[INFO] Not matching: ROI out of range !")
            status = "Not matching: ROI out of range"
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imshow("results", captured_img)
            cv2.imwrite('result.jpg', captured_img)
    else:
        print("[INFO] Not matching: matching pairs < required !")
        status = "Not matching: first matching pairs < required"
        captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                  interpolation=cv2.INTER_AREA)
        cv2.imshow("results", captured_img)
        cv2.imwrite('result.jpg', captured_img)
    #cv2.waitKey(2000)
    cv2.waitKey()
    return status, QRCodes

def detectionImgQR(p, captured_img, features, products, positions, product):
    # get product feature & position
    position, kp_tmp, des_tmp = getProductEx(features, products, positions, product)
    # Resize captured image: default 0.1
    cap_rows, cap_cols, cap_channels = captured_img.shape
    cap_scaled = cv2.resize(captured_img, (int(cap_cols * p.resize_img), int(cap_rows * p.resize_img)),
                            interpolation=cv2.INTER_AREA)
    # find the keypoints and descriptors with RootSIFT
    rs = RootSIFT()
    kp_cap_scaled, des_cap_scaled = rs.compute(cap_scaled)
    # first BF matching
    first_matching_points = bruteForce(des_tmp, des_cap_scaled, p.first_matching_level)
    print("[INFO] Good matching pairs: " + str(len(first_matching_points)))
    # first_match_count: default = 60
    if len(first_matching_points) >= p.first_match_count:
        src_pts = np.float32([kp_tmp[m.queryIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_cap_scaled[m.trainIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except:
            print("[WARNING] Fail to perform homography.")
        #matchesMask = mask.ravel().tolist()
        # h, w = template_img.shape
        h = int((position[1][1] - position[0][1] + 1) * p.resize_crop)
        w = int((position[1][0] - position[0][0] + 1) * p.resize_crop)
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        max_index = np.max(np.int32(dst / p.resize_img), axis=0)
        min_index = np.min(np.int32(dst / p.resize_img), axis=0)
        #  top-left point:       (x,y) -> (min_index[0, 0], min_index[0, 1])
        #  bottom-right point:   (x,y) -> (max_index[0, 0], max_index[0, 1])
        width = abs(min_index[0, 1] - max_index[0, 1])
        height = abs(min_index[0, 0] - max_index[0, 0])
        if min_index[0, 0] >= 0 and max_index[0, 0] < cap_cols and min_index[0, 1] >= 0 and max_index[
            0, 1] < cap_rows and width >= p.min_size and height >= p.min_size:
            # height from min_index[0, 1] to max_index[0, 1];
            # width  from min_index[0, 0] to max_index[0, 0];
            cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]

            crop_rows, crop_cols, crop_channels = cap_crop.shape
            cap_crop = cv2.resize(cap_crop, (int(crop_cols * p.resize_crop), int(crop_rows * p.resize_crop)),
                                  interpolation=cv2.INTER_AREA)
            #cv2.imshow("cropped image", cap_crop)

            dst_resize = np.float32(dst / p.resize_img)
            stretch_points = np.float32([[0, 0], [0, p.stretch_h], [p.stretch_w, p.stretch_h], [p.stretch_w, 0]])
            M_perspective = cv2.getPerspectiveTransform(dst_resize, stretch_points)
            img_perspective = cv2.warpPerspective(captured_img, M_perspective, (p.stretch_w, p.stretch_h))
            cv2.imshow("stretch image", img_perspective)
            print("[INFO] resolution of croped image: " + str(cap_crop.shape[0]) + " * " + str(cap_crop.shape[1]))
            # QR code
            img_perspective = img_perspective[int(p.stretch_h*0.1):int(p.stretch_h*0.5), int(p.stretch_w*0.5):int(p.stretch_w)]
            #img_perspective = img_perspective[50:250, 150:300]
            cv2.imwrite("stretch_image_QR.jpg", img_perspective)
            #cv2.imshow("OCR region", img_perspective)
            QRCodes = codeReader('./stretch_image_QR.jpg')
            if len(QRCodes) > 0:
                status = "QR code detected."
                print(QRCodes)
            else:
                status = "No QR code is detected."
            print("[INFO] " + status)

        elif (width < p.min_size and width > 0) or (height < p.min_size and height > 0):
            status = "Not matching: ROI too small"
            print("[INFO] " + status)
            #cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]
            # print("resolution of croped image= " + str(max_index[0, 0] - min_index[0, 0]) + "*" + str(max_index[0, 1] - min_index[0, 1]) + "(h * w)")
            # cv2.imshow("cropped image", cap_crop)
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imshow("results", captured_img)
            cv2.imwrite('result.jpg', captured_img)
        else:
            status = "Not matching: ROI out of range"
            print("[INFO] " + status)
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imshow("results", captured_img)
            cv2.imwrite('result.jpg', captured_img)
    else:
        status = "Not matching: matching pairs < required"
        print("[INFO] " + status)
        #cv2.imshow("results", captured_img)
        captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                  interpolation=cv2.INTER_AREA)
        cv2.imshow("results", captured_img)
        cv2.imwrite('result.jpg', captured_img)
    # cv2.waitKey(2000)
    cv2.waitKey()
    return status, QRCodes