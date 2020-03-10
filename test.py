from __future__ import print_function

import numpy as np
import cv2 as cv
from glob import glob
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from my_library import egi_mask

def weak_calibration(img_path, value=5):
    image = cv.imread(img_path)
    image = cv.resize(image, dsize=None, fx=4/3, fy=1, interpolation=cv.INTER_CUBIC)
    height, width = image.shape[:2]

    cx, cy = width / 2, height / 2
    calib_value = -value / 100_000_000

    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    D = np.hypot(X - cx, Y - cy)
    D2 = D + calib_value * np.power(D, 3)

    X2 = (cx + (X - cx) / (D + np.finfo(float).eps) * D2).astype(np.float32)
    Y2 = (cy + (Y - cy) / (D + np.finfo(float).eps) * D2).astype(np.float32)

    (map1, map2) = cv.convertMaps(X2, Y2, dstmap1type=0)
    dst = cv.remap(image, map1, map2, interpolation=cv.INTER_CUBIC)
    dst = cv.resize(dst, dsize=None, fx=3/4, fy=1, interpolation=cv.INTER_CUBIC)

    return dst

def basler3M_calibration_maps(image_size=None):
    """
    Use image_size=None if working with images in original resolution (2048x1536).
    If not, specify the real image size.
    """

    original_img_size = (2048, 1536)

    mtx = np.array([[1846.48412, 0.0,        1044.42589],
                    [0.0,        1848.52060, 702.441180],
                    [0.0,        0.0,        1.0]])

    dist = np.array([[-0.19601338, 0.07861078, 0.00182995, -0.00168376, 0.02604818]])

    new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(mtx, dist, original_img_size, 0, original_img_size)
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, new_camera_matrix, original_img_size, m1type=cv.CV_32FC1)

    if image_size is not None:
        mapx = cv.resize(mapx, (image_size[0], image_size[1])) * image_size[0] / original_img_size[0]
        mapy = cv.resize(mapy, (image_size[0], image_size[1])) * image_size[1] / original_img_size[1]

    return (mapx, mapy)

def calibrated(image, mapx, mapy):
    return cv.remap(image, mapx, mapy, interpolation=cv.INTER_CUBIC)

def calibrate_folder(folder, save_dir, img_size=None):
    images = glob(os.path.join(folder, "*.jpg"))
    mapx, mapy = basler3M_calibration_maps(img_size)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for image in images:
        basename = os.path.basename(image)
        img = cv.imread(image)
        cal_img = calibrated(img, mapx, mapy)
        cv.imwrite(os.path.join(save_dir, basename), cal_img)

# def estimate_rigid_tranform():
#     folder = "/media/deepwater/DATA/Shared/Louis/datasets/haricot_debug_montoldre_2"
#     image1 = os.path.join(folder, "im_03302.jpg")
#     image2 = os.path.join(folder, "im_03304.jpg")
#
#     img1 = cv.imread(image1)
#     img1 = cv.resize(img1, dsize=None, fx=4/3, fy=1, interpolation=cv.INTER_CUBIC)
#     img2 = cv.imread(image2)
#     img2 = cv.resize(img2, dsize=None, fx=4/3, fy=1, interpolation=cv.INTER_CUBIC)
#
#     transform = cv.calcOpticalFlowPyrLK(img1, img2, )

def find_corners(img, max_corners=50):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = egi_mask(img)
    mask = np.uint8(mask * 255)

    img_h, img_w = img.shape[:2]

    # Morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    mask = cv.dilate(mask, kernel, iterations=5)
    mask[:, :int(0.12 * img_w)] = 255
    mask[:, int(0.95 * img_w):] = 255
    mask[:int(0.07 * img_h), :] = 255
    mask[int(0.9 * img_h):, :] = 255

    corners = cv.goodFeaturesToTrack(img_gray,
        maxCorners=max_corners,
        qualityLevel=0.2,
        minDistance=0.05 * max(img_h, img_w),
        mask=~mask)

    return np.squeeze(corners, axis=1)

def estimate_homography(img1, img2):
    corners1 = find_corners(img1)
    corners2, status, _ = cv.calcOpticalFlowPyrLK(img1, img2,
        prevPts=corners1,
        nextPts=None)

    corners1 = corners1[np.squeeze(status) == 1, :]
    corners2 = corners2[np.squeeze(status) == 1, :]

    H, _ = cv.findHomography(corners1, corners2)
    print(H)

    (h, w) = img1.shape[:2]
    img3 = cv.warpPerspective(img1, H, dsize=(w, h))

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(img1)
    axarr[0, 0].scatter(corners1[:, 0], corners1[:, 1], s=10, c="r")
    axarr[1, 0].imshow(img2)
    axarr[1, 0].scatter(corners2[:, 0], corners2[:, 1], s=10, c="b")
    axarr[1, 1].imshow(img3)
    # plt.show()

    return H

def estimate_homographies(image_list_file):
    with open(image_list_file, "r") as f:
        images = f.readlines()
    images = [c.strip() for c in images]

    img1 = cv.imread(images[0])
    img1 = cv.resize(img1, dsize=None, fx=4/3, fy=1, interpolation=cv.INTER_CUBIC)

    (img_h, img_w) = img1.shape[:2]

    H = np.eye(3)

    for image in images[1:]:
        img2 = cv.imread(image)
        img2 = cv.resize(img2, dsize=None, fx=4/3, fy=1, interpolation=cv.INTER_CUBIC)

        h = estimate_homography(img1, img2)
        h = np.linalg.inv(h)  # Inverse transform
        H = np.matmul(H, h)  # Accumulate transforms

        img1 = img2
        img3 = cv.warpPerspective(img2, H,(img_w, img_h))

        # cv.imwrite("save/homography/{}".format(os.path.basename(image)), img3)
        plt.imshow(img3)
        # plt.show()

if __name__ == "__main__":
    img = "test.jpg"
    folder = "/media/deepwater/DATA/Shared/Louis/datasets/haricot_debug_montoldre_2"
    save_dir = "save/calibrated_images/"

    estimate_homographies("data/haricot_debug_long_2.txt")
