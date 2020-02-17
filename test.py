from __future__ import print_function

import numpy as np
import cv2 as cv
from glob import glob
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def calibrated_image(img_path, value=5):
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

def calibrate_folder(folder, value=5):
    images = glob(os.path.join(folder, "*.jpg"))
    values = [value for value in range(len(images))]

    for image in images:
        cal_image = calibrated_image(image, value)
        cv.imwrite(image, cal_image)

    # Parallel(n_jobs=-1, backend="multiprocessing")(delayed(calibrated_image)(arg) for arg in zip(images, values))


if __name__ == "__main__":
    img = "test.jpg"
    folder = "data/haricot_debug_montoldre_2_notcal/"

    calibrate_folder(folder, 10)


    # cv.imwrite("test_2.jpg", dst)
