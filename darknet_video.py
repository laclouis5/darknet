from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

from sort import *

class YoloModelPath:
    def __init__(self, model_folder):
        cfg = files_with_extension(model_folder, ".cfg")[0]
        weights = files_with_extension(model_folder, ".weights")[0]
        meta = files_with_extension(model_folder, ".data")[0]

        if not os.path.exists(cfg):
            raise ValueError("Invalid config file path '{}'".format(cfg))
        if not os.path.exists(weights):
            raise ValueError("Invalid weights file path '{}'".format(weights))
        if not os.path.exists(meta):
            raise ValueError("Invalid metadata file path '{}'".format(meta))

        self.cfg = cfg
        self.weights = weights
        self.meta = meta

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def xywh_to_xyx2y2(x, y, w, h):
    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2
    ymax = y + h / 2
    return xmin, ymin, xmax, ymax

def xyx2y2_to_xywh(xmin, ymin, xmax, ymax):
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return x, y, w, h

def to_relative(a, b, c, d, w, h):
    a_out = a / w
    b_out = b / h
    c_out = c / w
    d_out = d / h
    return a_out, b_out, c_out, d_out

def to_absolute(a, b, c, d, w, h):
    a_out = a * w
    b_out = b * h
    c_out = c * w
    d_out = d * h
    return a_out, b_out, c_out, d_out

def change_ref(a, b, c, d, old_size, new_size):
    rel = to_relative(a, b, c, d, old_size[0], old_size[1])
    abs = to_absolute(*rel, new_size[0], new_size[1])
    return abs

def cvDrawBoxes(detections, img):
    for detection in detections:
        xmin, ymin, xmax, ymax = convertBack(*detection[2])
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def files_with_extension(folder, extension):
    return [os.path.join(folder, item)
            for item in os.listdir(folder)
            if os.path.splitext(item)[1] == extension]

# ============================================================================#

netMain = None
metaMain = None
altNames = None

def YOLO():
    # Init and stuff
    global metaMain, netMain, altNames

    yolo= YoloModelPath("results/yolo_v3_spp_pan_csr50_3/")
    configPath = yolo.cfg
    weightPath = yolo.weights
    metaPath = yolo.meta

    tracker = Sort(max_age=100, min_hits=3)

    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass


    net_height = darknet.network_height(netMain)
    net_width = darknet.network_width(netMain)

    # Video Params
    cap = cv2.VideoCapture("/home/deepwater/Videos/GOPR1262.m4v")
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (img_width, img_height))

    # Main Loop
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(net_width, net_height, 3)

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()

        if ret == True:
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (net_width, net_height), interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.20)

            # Tracking
            # np_detections = np.array([[*xywh_to_xyx2y2(*det[2]), det[1]] for det in detections])
            # tracks = tracker.update(np_detections)
            # cv_detections = [('{}'.format(det[4]).encode("ascii"), 1, (*change_ref(*xyx2y2_to_xywh(*det[:-1]), (net_width, net_height), (img_width, img_height)), )) for det in tracks]

            # If no detections
            cv_detections = [(det[0], det[1], (*change_ref(*det[2], (net_width, net_height), (img_width, img_height)), )) for det in detections]

            image = cvDrawBoxes(cv_detections, frame_rgb)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out.write(image)

            print("Fps: {}".format(int(1 / (time.time() - prev_time))))
            cv2.imshow('Demo', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    out.release()


if __name__ == "__main__":
    YOLO()
