import pypylon.pylon as py
import cv2
import numpy as np
import pickle
from imutils import paths
import os

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
                 resize_tem=0.2, first_matching_level = 0.8, second_matching_level = 0.8, first_match_count = 60,
                 score_thresh=160, second_match_count=150, min_size = 100):
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
    labels = data["feature"]
    products = data["product"]
    print("[INFO] Label data loaded.")
    return labels, products

# label + feature + position data
def initLabelEx(file):
    # Get feature from file
    # data = pickle.loads(open("labels.pkl", "rb").read())
    data = pickle.loads(open(file, "rb").read())
    labels = data["feature"]
    products = data["product"]
    positions = data["position"]
    print("[INFO] Label data loaded.")
    return labels, products, positions

# Initilize env: init camera + init feature
def initCamEnv(file):
    camera = initilizeCamera()
    labels, products = initLabel(file)
    return camera, labels, products

def initCamEnvEx(file):
    camera = initilizeCamera()
    labels, products, positions = initLabelEx(file)
    return camera, labels, products, positions

def initImgEnv(file):
    labels, products = initLabel(file)
    return labels, products

def initImgEnvEx(file):
    labels, products, positions = initLabelEx(file)
    return labels, products, positions

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



