import cv2
import numpy as np
import pickle
import util

# RootSift + Load Feature + ROI detection + BF matcher + single captured image
# Dependency: rootsift.py + feature.py

def detection(p, camera, labels, products, product):
    try:
        index = products.index(product)
        label = labels[index]
        kp_tmp, des_tmp = util.unpickle_keypoints(label)
        print("Feature included!")
    except:
        print("Product is not included!")
    # Get feature from file
    #kp_tmp, des_tmp = util.unpickle_keypoints(feature_database[product])
    # Grab a single camera image
    captured_img = util.getImgFromCamera(camera)
    # Resize captured image
    cap_rows, cap_cols, cap_channels = captured_img.shape
    cap_scaled = cv2.resize(captured_img, (int(cap_cols * p.resize_img), int(cap_rows * p.resize_img)),
                            interpolation=cv2.INTER_AREA)
    # find the keypoints and descriptors with RootSIFT
    rs = util.RootSIFT()
    kp_cap_scaled, des_cap_scaled = rs.compute(cap_scaled)
    # BF matching
    first_matching_points = util.bruteForce(des_tmp, des_cap_scaled, p.first_matching_level)
    print("Good matching pairs of first match: " + str(len(first_matching_points)))

    if len(first_matching_points) >= p.MIN_MATCH_COUNT:
        src_pts = np.float32([kp_tmp[m.queryIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_cap_scaled[m.trainIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #matchesMask = mask.ravel().tolist()
        # h, w = template_img.shape
        h = p.template_height
        w = p.templat_width
        # h, w, d= template_img.shape
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
            cv2.imshow("cropped image", cap_crop)
            print("resolution of croped image: " + str(cap_crop.shape[0]) + " * " + str(cap_crop.shape[1]))
            kp_cap_crop, des_cap_crop = rs.compute(cap_crop)

            try:
                second_matching_points = util.bruteForce(des_tmp, des_cap_crop, p.second_matching_level)
            except:
                #exit()
                print("Not matching: BF Matching failed.")
                #status = "Not matching: BF Matching failed"
                status = "102"
                captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                        interpolation=cv2.INTER_AREA)
                cv2.imwrite('result.jpg', captured_img)
                return status

            # img2 = cv.polylines(cap_scaled,[np.int32(dst)],True,255,3, cv.LINE_AA)
            print("Good matching pairs of second match: " + str(len(second_matching_points)))
            score = util.getScore(second_matching_points, width, height)
            print("score = " + str(score))
            if score >= p.score_thresh:
                print("Matching!!!")
                #status = "Matching!!!"
                status = "201"
                img2 = cv2.polylines(captured_img, [np.int32(dst / p.resize_img)], True, 255, 3, cv2.LINE_AA)
                cv2.imshow("results", captured_img)
                captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                        interpolation=cv2.INTER_AREA)
                cv2.imwrite('result.jpg', captured_img)
            else:
                print("Not matching: <threshold")
                #status = "Not matching: score <threshold"
                status = "103"
                cv2.imshow("results", captured_img)
                captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                        interpolation=cv2.INTER_AREA)
                cv2.imwrite('result.jpg', captured_img)
        elif (width < p.min_size and width > 0) or (height < p.min_size and height > 0):
            print('Not matching: too small size !')
            #status = "Not matching: ROI too small"
            status = "104"
            cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]
            # print("resolution of croped image= " + str(max_index[0, 0] - min_index[0, 0]) + "*" + str(max_index[0, 1] - min_index[0, 1]) + "(h * w)")
            # cv2.imshow("cropped image", cap_crop)
            cv2.imshow("results", captured_img)
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imwrite('result.jpg', captured_img)
        else:
            print('Not matching: ROI out of range !')
            #status = "Not matching: ROI out of range"
            status = "105"
            cv2.imshow("results", captured_img)
            captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                      interpolation=cv2.INTER_AREA)
            cv2.imwrite('result.jpg', captured_img)
    else:
        print('Not matching: matching pairs < required !')
        #status = "Not matching: matching pairs < required"
        status = "101"
        cv2.imshow("results", captured_img)
        captured_img = cv2.resize(captured_img, (int(cap_cols * 0.2), int(cap_rows * 0.2)),
                                  interpolation=cv2.INTER_AREA)
        cv2.imwrite('result.jpg', captured_img)
    cv2.waitKey(2000)
    return status

def detection2(p, camera, feature_database, product = 0):
    # Get feature from file
    kp_tmp, des_tmp = util.unpickle_keypoints(feature_database[product])
    # Grab a single camera image
    captured_img = util.getImgFromCamera(camera)
    # Resize captured image
    cap_rows, cap_cols, cap_channels = captured_img.shape
    cap_scaled = cv2.resize(captured_img, (int(cap_cols * p.resize_img), int(cap_rows * p.resize_img)),
                            interpolation=cv2.INTER_AREA)
    # find the keypoints and descriptors with RootSIFT
    rs = util.RootSIFT()
    kp_cap_scaled, des_cap_scaled = rs.compute(cap_scaled)
    # BF matching
    first_matching_points = util.bruteForce(des_tmp, des_cap_scaled, p.first_matching_level)
    print("Good matching pairs of first match: " + str(len(first_matching_points)))

    if len(first_matching_points) >= p.MIN_MATCH_COUNT:
        src_pts = np.float32([kp_tmp[m.queryIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_cap_scaled[m.trainIdx].pt for m in first_matching_points]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #matchesMask = mask.ravel().tolist()
        # h, w = template_img.shape
        h = p.template_height
        w = p.templat_width
        # h, w, d= template_img.shape
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
            cv2.imshow("cropped image", cap_crop)
            print("resolution of croped image: " + str(cap_crop.shape[0]) + " * " + str(cap_crop.shape[1]))
            kp_cap_crop, des_cap_crop = rs.compute(cap_crop)

            try:
                second_matching_points = util.bruteForce(des_tmp, des_cap_crop, p.second_matching_level)
            except:
                #exit()
                print("Not matching: BF Matching failed.")
                status = "Not matching: BF Matching failed"
                return status

            # img2 = cv.polylines(cap_scaled,[np.int32(dst)],True,255,3, cv.LINE_AA)
            print("Good matching pairs of second match: " + str(len(second_matching_points)))
            score = util.getScore(second_matching_points, width, height)
            print("score = " + str(score))
            if score >= p.score_thresh:
                print("Matching!!!")
                status = "Matching!!!"
                img2 = cv2.polylines(captured_img, [np.int32(dst / p.resize_img)], True, 255, 3, cv2.LINE_AA)
                cv2.imshow("results", captured_img)
            else:
                print("Not matching: <threshold")
                status = "Not matching: <threshold"
                cv2.imshow("results", captured_img)
        elif (width < p.min_size and width > 0) or (height < p.min_size and height > 0):
            print('Not matching: too small size !')
            status = "Not matching: Not matching: too small size"
            cap_crop = captured_img[min_index[0, 1]:max_index[0, 1], min_index[0, 0]:max_index[0, 0]]
            # print("resolution of croped image= " + str(max_index[0, 0] - min_index[0, 0]) + "*" + str(max_index[0, 1] - min_index[0, 1]) + "(h * w)")
            # cv2.imshow("cropped image", cap_crop)
            cv2.imshow("results", captured_img)
        else:
            print('Not matching: ROI out of range !')
            status = "Not matching: ROI out of range"
            cv2.imshow("results", captured_img)
    else:
        print('Not matching: matching pairs < required !')
        status = "Not matching: matching pairs < required"
        cv2.imshow("results", captured_img)
    cv2.waitKey(2000)
    return status

if __name__ == '__main__':
    # initilize parameters
    p = util.Parameter()
    #camera, feature_database = util.initEnv()
    camera, labels, products = util.initEnv2("labels.pkl")
    product = "NV230C"
    status = detection(p, camera, labels, products, product)

    print("Detection ends")


