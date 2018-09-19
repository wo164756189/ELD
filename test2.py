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
            # QR code
            img_perspective = img_perspective[int(p.stretch_h*0.1):int(p.stretch_h*0.5), int(p.stretch_w*0.5):int(p.stretch_w)]
            #img_perspective = img_perspective[50:250, 150:300]
            cv2.imwrite("stretch_image.jpg", img_perspective)
            #cv2.imshow("OCR region", img_perspective)
            codeReader('./stretch_image.jpg')

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
            if score >= p.score_thresh and len(second_matching_points) >= p.second_match_count:
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