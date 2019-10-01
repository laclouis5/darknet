#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"


To use, either run performDetect() after import, or modify the end of this file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)


Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""

#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import math
import random
import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
try:
    import cv2 as cv
except:
    pass
from PIL import Image
from skimage import io, filters, morphology
from joblib import Parallel, delayed

from lxml.etree import Element, SubElement, tostring, parse
from test import read_detection_txt_file, save_yolo_detect_to_txt, yolo_det_to_bboxes, save_bboxes_to_txt, nms, create_dir

from utils import *
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class XMLTree:
    def __init__(self, image_name, width, height, user_name='Unknown user', date=str(datetime.date.today())):
        """
        Creates a ElementTree containing results of classification.

        Arguments:
        image_name -- String containing the name of the source image used for classification.
        width      -- Image width (integer).
        height     -- Image height (integer).
        user_name  -- String containing the name of the participant.
        date       -- String containing the result production date (YYYY-DD-MM).

        Returns:
        A ElementTree Element

        For additional information about format please refer to 'PE_ROSE_dryrn.pdf'
        """

        # Root node
        self.tree = Element('GEDI')

        # First level nodes
        document = SubElement(self.tree, 'DL_DOCUMENT')
        user = SubElement(self.tree, 'USER')

        # Second level nodes
        src = SubElement(document, 'SRC')
        tag = SubElement(document, 'DOC_TAG')
        w = SubElement(document, 'WIDTH')
        h = SubElement(document, 'HEIGHT')

        _name = SubElement(user, 'NAME')
        _date = SubElement(user, 'DATE')

        # Fill info
        src.text = image_name
        tag.text = 'xml'
        w.text = str(width)
        h.text = str(height)

        _name.text = user_name
        _date.text = date


    def add_mask_zone(self, plant_type, bbox, name=''):
        """
        Creates new mask zones in the input tree.

        Arguments:
        tree       -- The ElementTree to be filled with mask info.
        plant_type -- String containing the type of plant (either 'Adventice' or 'PlanteInteret')
        name       -- String (optional), plant name.
        """
        # Go to 'DL_DOCUMENT' node & retreive mask ID
        doc_node = self.tree.find('DL_DOCUMENT')
        nb_masks = self.get_next_mask_id()

        # Create new mask
        mask = Element('MASK_ZONE')
        _id = SubElement(mask, 'ID')
        _type = SubElement(mask, 'TYPE')
        _name = SubElement(mask, 'NAME')
        _bndbox = SubElement(mask, 'BNDBOX')

        _xmin = SubElement(_bndbox, 'XMIN')
        _ymin = SubElement(_bndbox, 'YMIN')
        _xmax = SubElement(_bndbox, 'XMAX')
        _ymax = SubElement(_bndbox, 'YMAX')

        # Fill info & append
        _id.text = str(nb_masks)
        _type.text = plant_type
        _name.text = name

        _xmin.text = str(bbox[0])
        _ymin.text = str(bbox[1])
        _xmax.text = str(bbox[2])
        _ymax.text = str(bbox[3])

        doc_node.append(mask)


    def get_next_mask_id(self):
        """
        Return the next unique ID for masks.
        """

        # Go to 'DL_DOCUMENT' node
        doc_node = self.tree.find('DL_DOCUMENT')

        # Compute new mask index
        masks_list = doc_node.findall('MASK_ZONE')
        return len(masks_list)


    def get_current_mask_id(self):
        return self.get_next_mask_id() - 1


    def save(self, xml_file_name):
        """
        Saves to disk the tree to XML file.

        Arguments:
        tree -- The ElementTree Element to save.
        xml_file_name -- String containing the name of the XML file. See 'PE_ROSE_dryrn.pdf' for more information.
        """

        # TreeElement to string representation
        tree_str = tostring(self.tree, encoding='unicode', pretty_print=True)

        # Write data
        with open(xml_file_name, 'w') as xml_file:
            xml_file.write(tree_str)


    def clean_xml(folders):
        for folder in folders:
            for file in os.listdir(folder):

                if(os.path.splitext(file)[1] != '.xml'):
                    continue

                tree = parse(os.path.join(folder, file)).getroot()

                path_field = tree.find('path')
                path_field.text = os.path.join(folder, file)

                with open(os.path.join(folder, file), 'w') as xml_file:
                    tree_str = tostring(tree, encoding='unicode', pretty_print=True)
                    xml_file.write(tree_str)


    def xlm_to_csv(folders, classes_to_keep=[], cvs_path=''):
        with open(os.path.join(csv_path, 'train_data.csv'), 'w') as csv_file:
            for folder in folders:
                for file in sorted(os.listdir(folder)):
                    # Check if XML file
                    if(os.path.splitext(file)[1] != '.xml'):
                        continue
                    # Retreive the XML etree
                    root = parse(os.path.join(folder, file)).getroot()

                    file_name = root.find('filename').text

                    # Retreive and process each 'object' in the etree
                    for obj in root.findall('object'):
                        name = obj.find('name').text

                        # Save only selected classes
                        # Comment to ignore class selection
                        if (classes_to_keep.count != 0) and (name not in classes_to_keep):
                            continue

                        # Retreive bounding box coordinates
                        bounding_box = obj.find('bndbox')
                        coords = []

                        for coord in bounding_box.getchildren():
                            coords.append(int(coord.text))

                        # Write CSV file
                        csv_file.write(os.path.join(folder, file_name) + ',')
                        for coord in coords:
                            csv_file.write(str(coord) + ',')
                        csv_file.write(name + '\n')


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `"+winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    im = load_image(image, 0, 0)
    if debug: print("Loaded image")
    ret = detect_image(net, meta, im, thresh, hier_thresh, nms, debug)
    free_image(im)
    if debug: print("freed image")
    return ret

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    #import cv2
    #custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
    #custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    #custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    #import scipy.misc
    #custom_image = scipy.misc.imread(image)
    #im, arr = array_to_image(custom_image)		# you should comment line below: free_image(im)
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, 0) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res


netMain = None
metaMain = None
altNames = None

def performDetect(imagePath="data/dog.jpg", thresh= 0.25, configPath = "./cfg/yolov3.cfg", weightPath = "yolov3.weights", metaPath= "./cfg/coco.data", showImage= True, makeImageOnly = False, initOnly= False):
    """
    Convenience function to handle the detection and returns of objects.

    Displaying bounding boxes requires libraries scikit-image and numpy

    Parameters
    ----------------
    imagePath: str
        Path to the image to evaluate. Raises ValueError if not found

    thresh: float (default= 0.25)
        The detection threshold

    configPath: str
        Path to the configuration file. Raises ValueError if not found

    weightPath: str
        Path to the weights file. Raises ValueError if not found

    metaPath: str
        Path to the data file. Raises ValueError if not found

    showImage: bool (default= True)
        Compute (and show) bounding boxes. Changes return.

    makeImageOnly: bool (default= False)
        If showImage is True, this won't actually *show* the image, but will create the array and return it.

    initOnly: bool (default= False)
        Only initialize globals. Don't actually run a prediction.

    Returns
    ----------------------


    When showImage is False, list of tuples like
        ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
        The X and Y coordinates are from the center of the bounding box. Subtract half the width or height to get the lower corner.

    Otherwise, a dict with
        {
            "detections": as above
            "image": a numpy array representing an image, compatible with scikit-image
            "caption": an image caption
        }
    """
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
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
    if initOnly:
        print("Initialized detector")
        return None
    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `"+os.path.abspath(imagePath)+"`")
    # Do the detection
    #detections = detect(netMain, metaMain, imagePath, thresh)	# if is used cv2.imread(image)
    detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    if showImage:
        try:
            from skimage import io, draw
            import numpy as np
            image = io.imread(imagePath)
            print("*** "+str(len(detections))+" Results, color coded by confidence ***")
            imcaption = []
            for detection in detections:
                label = detection[0]
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape
                # x = shape[1]
                # xExtent = int(x * bounds[2] / 100)
                # y = shape[0]
                # yExtent = int(y * bounds[3] / 100)
                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2]/2)
                yCoord = int(bounds[1] - bounds[3]/2)
                boundingBox = [
                    [xCoord, yCoord],
                    [xCoord, yCoord + yExtent],
                    [xCoord + xEntent, yCoord + yExtent],
                    [xCoord + xEntent, yCoord]
                ]
                # Wiggle it around to make a 3px border
                rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
                rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
                boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
                draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
                draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
                draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
                draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
            if not makeImageOnly:
                io.imshow(image)
                io.show()
            detections = {
                "detections": detections,
                "image": image,
                "caption": "\n<br/>".join(imcaption)
            }
            print(detections)
        except Exception as e:
            print("Unable to show image: "+str(e))
    return detections


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# from BoundingBox import BoundingBox
# from BoundingBoxes import BoundingBoxes
# from Evaluator import Evaluator
# from test import read_txt_annotation_file, parse_yolo_folder
# from utils import *
#
# def compute_mean_crop_annotation_to_squareaverage_precision(folder, model, config_file, data_obj):
#     files = os.listdir(folder)
#     images = [os.path.join(folder, file) for file in files if os.path.splitext(file)[1] == '.jpg']
#
#     bounding_boxes = parse_yolo_folder(folder)
#     bounding_boxes.mapLabels({0: "mais", 1: 'haricot', 2: 'carotte'})
#
#     for image in images:
#         detections = performDetect(
#             imagePath=image,
#             configPath=config_file,
#             weightPath=model,
#             metaPath=data_obj,
#             showImage=False)
#
#         img_size = Image.open(image).size
#
#         for detection in detections:
#             label, conf = detection[0], detection[1]
#             # Abs XYX2Y2
#             x_min, y_min, x_max, y_max = convertBack(*detection[2])
#
#             bounding_boxes.addBoundingBox(BoundingBox(
#                 imageName=os.path.basename(image),
#                 classId=label,
#                 x=x_min, y=y_min, w=x_max, h=y_max,
#                 bbType=BBType.Detected, classConfidence=conf,
#                 format=BBFormat.XYX2Y2,
#                 imgSize=img_size))
#
#     evaluator = Evaluator()
#     metrics = evaluator.GetPascalVOCMetrics(bounding_boxes)
#     for item in metrics:
#         (prec,  rec) = item["precision"], item["recall"]
#         print("{} - mAP: {:.4} %, TP: {}, FP: {}, tot. pos.: {}".format(item['class'], 100*item['AP'], item["total TP"], item["total FP"], item["total positives"]))


def save_detect_to_txt(folder_path, save_dir, model, config_file, data_file):
    img_gen = ImageGeneratorFromFolder(folder_path)
    create_dir(save_dir)

    for image in img_gen:
        detections = performDetect(image, thresh=0.005, configPath=config_file, weightPath=model, metaPath=data_file, showImage=False)

        save_name = os.path.join(save_dir, os.path.splitext(os.path.basename(image))[0]) + ".txt"
        print(save_name)

        (height, width) = cv.imread(image).shape[0:2]

        lines = []
        for detection in detections:
            box = detection[2]
            label = detection[0]
            # (xmin, ymin, xmax, ymax) = convertBack(box[0], box[1], box[2], box[3])
            (x, y, w, h) = box[0]/width, box[1]/height, box[2]/width, box[3]/height
            confidence = detection[1]
            lines.append("{} {} {} {} {} {}\n".format(map_labels[label], confidence, x, y, w, h))

        with open(save_name, 'w') as f:
            f.writelines(lines)


def convert_yolo_annot_to_XYX2Y2(annotation_dir, save_dir, lab_to_name):
    annotations = [os.path.join(annotation_dir, item) for item in os.listdir(annotation_dir) if os.path.splitext(item)[1] == '.txt']
    images = [os.path.splitext(item)[0] + '.jpg' for item in annotations]

    for (image, annotation) in zip(images, annotations):
        (img_w, img_h) = Image.open(image).size
        print('Image:      {}'.format(image))
        print('Annotation: {}'.format(annotation))
        print('Image Size: {} x {}'.format(img_w, img_h))

        with open(annotation, 'r') as f:
            content = f.readlines()
        content = [item.strip() for item in content]

        with open(os.path.join(save_dir, os.path.basename(annotation)), 'w') as fw:
            for line in content:
                line = line.split(' ')
                (label, x, y, w, h) = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
                # print('Line:       {} {:.4} {:.4} {:.4} {:.4}'.format(int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])))
                (xmin, ymin, xmax, ymax) = convertBack(x*img_w, y*img_h, w*img_w, h*img_h)
                fw.write('{} {} {} {} {}\n'.format(lab_to_name[label], xmin, ymin, xmax, ymax))
                # fw.write('{} {} {} {} {}\n'.format(lab_to_name[label], x*img_w, y*img_h, w*img_w, h*img_h))


def crop_annotation_to_square(annot_folder, save_dir, lab_to_name):
    annotations = [os.path.join(annot_folder, item) for item in os.listdir(annot_folder) if os.path.splitext(item)[1] == '.txt']

    for annotation in annotations:
        content_out = []
        corresp_img = os.path.splitext(annotation)[0] + '.jpg'
        (img_w, img_h) = Image.open(corresp_img).size

        print("In landscape mode: {} by {}".format(img_w, img_h))
        # Here are abs coords of square bounds (left and right)
        (w_lim_1, w_lim_2) = round(float(img_w)/2 - float(img_h)/2), round(float(img_w)/2 + float(img_h)/2)

        with open(annotation, 'r') as f:
            print("Reading annotation...")
            content = f.readlines()
            content = [line.strip() for line in content]

            for line in content:
                print("Reading a line...")
                line = line.split()
                # Get relative coords (in old coords system)
                (label, x, y, w, h) = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
                print("Line is: {} {} {} {} {}".format(label, x, y, w, h))

                # If bbox is not out of the new square frame
                if not (x*img_w < w_lim_1 or x*img_w > w_lim_2):
                    print("In square bounds")
                    # But if bbox spans out of one bound (l or r)
                    if (x - w/2.0) < (float(w_lim_1)/img_w):
                        print("Spans out of left bound")
                        # Then adjust bbox to fit in the square
                        w = w - (float(w_lim_1)/img_w - (x - w/2.0))
                        x = float(w_lim_1+1)/img_w + w/2.0
                    if (x + w/2.0) > (float(w_lim_2)/img_w):
                        print("Span out of right bound")
                        w = w - (x + w/2.0 - float(w_lim_2)/img_w)
                        x = float(w_lim_2)/img_w - w/2.0
                    else:
                        print("Does not spans outside")

                # If out of bounds...
                else:
                    print("Out of square bounds")
                    # ...do not process the line
                    continue

                # Do not forget to convert from old coord sys to new one
                x = (x*img_w - float(w_lim_1))/float(w_lim_2 - w_lim_1)
                w = w*img_w/float(w_lim_2 - w_lim_1)

                assert x >= 0, "Value was {}".format(x)
                assert x <= 1, "Value was {}".format(x)
                assert (x - w/2) >= 0, "Value was {}".format(x - w/2)
                assert (x + w/2) <= 1, "Value was {}".format(x + w/2)

                size = min(img_w, img_h)

                (xmin, ymin, xmax, ymax) = convertBack(x*size, y*size, w*size, h*size)

                new_line = "{} {} {} {} {}\n".format(lab_to_name[label], xmin, ymin, xmax, ymax)
                content_out.append(new_line)

        # Write updated content to TXt file
        with open(os.path.join(save_dir, os.path.basename(annotation)), 'w') as f:
            f.writelines(content_out)


def crop_detection_to_square(image_path, save_dir, model, config_file, meta_file):
    images = [os.path.join(image_path, item) for item in os.listdir(image_path) if os.path.splitext(item)[1] == '.jpg']
    images.sort()

    for image in images:
        content_out = []
        (img_w, img_h) = Image.open(image).size
        (w_lim_1, w_lim_2) = round(float(img_w)/2 - float(img_h)/2), round(float(img_w)/2 + float(img_h)/2)

        detections = performDetect(
            imagePath=image,
            configPath=config_file,
            weightPath=model,
            metaPath=meta_file,
            showImage=False)

        for detection in detections:
            label = detection[0]
            prob = detection[1]
            (x, y, w, h) = detection[2]
            (x, y, w, h) = (x/img_w, y/img_h, w/img_w, h/img_h)

            # If bbox is not out of the new square frame
            if not (x*img_w < w_lim_1 or x*img_w > w_lim_2):
                # But if bbox spans out of one bound (l or r)
                if (x - w/2.0) < (float(w_lim_1)/img_w):
                    # Then adjust bbox to fit in the square
                    w = w - (float(w_lim_1)/img_w - (x - w/2.0))
                    x = float(w_lim_1+1)/img_w + w/2.0
                if (x + w/2.0) > (float(w_lim_2)/img_w):
                    w = w - (x + w/2.0 - float(w_lim_2)/img_w)
                    x = float(w_lim_2)/img_w - w/2.0

            else: continue

            # Do not forget to convert from old coord sys to new one
            x = (x*img_w - float(w_lim_1))/float(w_lim_2 - w_lim_1)
            w = w*img_w/float(w_lim_2 - w_lim_1)

            assert x >= 0, "Value was {}".format(x)
            assert x <= 1, "Value was {}".format(x)
            assert (x - w/2) >= 0, "Value was {}".format(x - w/2)
            assert (x + w/2) <= 1, "Value was {}".format(x + w/2)

            size = min(img_w, img_h)

            (xmin, ymin, xmax, ymax) = convertBack(x*size, y*size, w*size, h*size)

            new_line = "{} {} {} {} {} {}\n".format(label, prob, xmin, ymin, xmax, ymax)
            content_out.append(new_line)

        # Write updated content to TXT file
        save_name = os.path.splitext(os.path.basename(image))[0] + '.txt'
        with open(os.path.join(save_dir, save_name), 'w') as f:
            f.writelines(content_out)


def draw_boxes(image, annotation, save_path, color=[255, 64, 0]):
    save_name        = os.path.join(save_path, os.path.basename(image))
    height, width, _ = cv.imread(image).shape

    boxes = read_detection_txt_file(annotation, (width, height))
    img = cv.imread(image)

    for box in boxes.getBoundingBoxes():
        add_bb_into_image(img, box, color=color, label=box.getClassId())

    cv.imwrite(os.path.join(save_name), img)


def draw_boxes_bboxes(image, bounding_boxes, save_dir, color=[255, 64, 0]):
    image = image.copy()
    for box in bounding_boxes.getBoundingBoxes():
        add_bb_into_image(image, box, color=color, label=box.getClassId())
        image_path = os.path.join(save_dir, box.getImageName())

    cv.imwrite(image_path, image)


def draw_deque_boxes(image, deq, save_path):
    image = image.copy()
    hsv = plt.get_cmap("cool")
    colors = hsv(np.linspace(0, 1, deq.maxlen))[..., :3]

    print("DEQUE: {}".format(deq))

    i=0
    for bboxes in reversed(deq):
        for box in bboxes.getBoundingBoxes():
            add_bb_into_image(image, box, 255*colors[i], label=box.getClassId())

        i += 1
    cv.imwrite(save_path, image)


def draw_boxes_folder(images_path, annotations_path, save_path):
    images = ImageGeneratorFromFolder(images_path)

    for image in images:
        annotation = os.path.splitext(os.path.basename(image))[0] + ".txt"
        annotation = os.path.join(annotations_path, annotation)

        draw_boxes(image, annotation, save_path)


def ImageGeneratorFromVideo(video_path, skip_frames=1, gray_scale=True, down_scale=1, ratio=None):
    video = cv.VideoCapture(video_path)
    ret = True
    while ret:
        for _ in range(skip_frames):
            ret, frame = video.read()

        if down_scale > 1:
            frame = frame[::down_scale, ::down_scale]

        if gray_scale:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if ratio is not None:
            height, width = frame.shape[0:2]
            new_width = ratio * height
            to_crop   = int((width - new_width) / 2)

            frame = frame[:, to_crop:-to_crop, :]

        yield(ret, frame)


def convert_to_grayscale(image):
    frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return frame


def ImageGeneratorFromFolder(folder, sorted=False):
    files = [os.path.join(folder, item) for item in os.listdir(folder) if os.path.splitext(item)[1] == ".jpg"]
    if sorted:
        files.sort()
    for file in files:
        yield file


def save_images_from_video(path_to_video, save_dir, nb_iter=100):
    skip_frames = 2
    video_gen = ImageGeneratorFromVideo(path_to_video, skip_frames=skip_frames, gray_scale=False)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    i = 0
    while i < nb_iter:
        _, frame = next(video_gen)
        height, width = frame.shape[0:2]
        new_ratio = 4/3
        new_width = new_ratio * height
        to_crop   = int((width - new_width) / 2)

        frame = frame[:, to_crop:-to_crop, :]

        frame_name = os.path.join(save_dir, "im_{}.jpg".format(i*skip_frames))
        cv.imwrite(frame_name, frame, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        i += 1


def filter_detections(video_path, video_param, save_dir, yolo_param, k=5):
    image_dir = os.path.join(save_dir, "images")
    annot_dir = os.path.join(save_dir, "annotations")
    draw_dir  = os.path.join(save_dir, "draw")

    create_dir(save_dir)
    create_dir(image_dir)
    create_dir(annot_dir)
    create_dir(draw_dir)

    images = ImageGeneratorFromVideo(
        video_path,
        skip_frames=video_param["skip_frames"],
        gray_scale=video_param["gray_scale"],
        down_scale=video_param["down_scale"],
        ratio=video_param["ratio"])

    boxes = deque(maxlen=k)
    _, first_image = next(images)

    image_name = os.path.join(image_dir, "im_0.jpg")
    cv.imwrite(image_name, first_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    detections = performDetect(
        image_name,
        configPath = yolo_param["cfg"],
        weightPath =yolo_param["model"],
        metaPath=yolo_param["obj"],
        showImage=False)

    bboxes = yolo_det_to_bboxes("im_0.jpg", detections)
    bboxes.keepOnlyName("leek")
    boxes.append(bboxes)
    save_bboxes_to_txt(bboxes, annot_dir)
    draw_deque_boxes(first_image, boxes, os.path.join(draw_dir, "im_0.jpg"))

    first_image = convert_to_grayscale(first_image)
    prev_opt_flow = np.zeros_like(first_image)
    file_nb = 1

    for _, image in images:
        second_image = convert_to_grayscale(image)

        optical_flow = cv.calcOpticalFlowFarneback(
            prev=first_image,
            next=second_image,
            flow=prev_opt_flow,
            pyr_scale=0.5,
            levels=4,
            winsize=32,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0)

        dx = optical_flow[..., 0]
        dy = optical_flow[..., 1]

        mean_dx = dx.sum() / dx.size
        mean_dy = dy.sum() / dy.size

        image_name = os.path.join(image_dir, "im_{}.jpg".format(file_nb))
        cv.imwrite(image_name, image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        detections = performDetect(
            image_name,
            configPath=yolo_param["cfg"],
            weightPath=yolo_param["model"],
            metaPath=yolo_param["obj"],
            showImage=False)

        bboxes = yolo_det_to_bboxes("im_{}.jpg".format(file_nb), detections)
        bboxes.keepOnlyName("leek")

        # Print stuff
        print(image_name)
        print("  Mean dx: {:.6}  Mean dy: {:.6}".format(mean_dx, mean_dy))
        print("  Detection Count: {}".format(len(bboxes.getBoundingBoxes())))

        [item.shiftBoundingBoxesBy(mean_dx, mean_dy) for item in boxes]
        boxes.append(bboxes)

        # Flatten
        boxes_to_save = []
        [boxes_to_save.extend(item.getBoundingBoxes()) for item in boxes]
        [box.setImageName("im_{}.jpg".format(file_nb)) for box in boxes_to_save]
        boxes_to_save = BoundingBoxes(boxes_to_save)

        # NMS stuff and draw
        boxes_to_save = nms(boxes_to_save)
        draw_deq = deque(maxlen=1)
        draw_deq.append(boxes_to_save)
        draw_deque_boxes(image, draw_deq, os.path.join(draw_dir, "im_{}.jpg".format(file_nb)))
        save_bboxes_to_txt(boxes_to_save, annot_dir)

        first_image = second_image
        prev_opt_flow = optical_flow
        file_nb += 1
        print()


if __name__ == "__main__":
    image_path  = "data/val/"
    model_path  = "results/yolo_v3_tiny_pan3_1/yolo_v3_tiny_pan3_aa_ae_mixup_scale_giou_best.weights"
    config_file = "results/yolo_v3_tiny_pan3_1/yolo_v3_tiny_pan3_aa_ae_mixup_scale_giou.cfg"
    meta_path   = "results/yolo_v3_tiny_pan3_1/obj.data"

    yolo_param  = {"model": model_path, "cfg": config_file, "obj": meta_path}

    video_path  = "/media/deepwater/Elements/Louis/2019-07-25_larrere_videos/demo_tele_4K.mp4"
    video_param = {"skip_frames": 5, "down_scale": 2, "gray_scale": False, "ratio": 4/3}

    consort  = 'Bipbip'
    save_dir = 'save/'
    labels_to_names = ['maize', 'bean', 'leek', 'maize_stem', 'bean_stem', 'leek_stem']
    map_labels      = {'maize': 0, 'bean': 1, 'leek': 2, 'stem_maize': 3, 'stem_bean': 4, 'stem_leek': 5}
    # save_dir = /Users/louislac/Downloads/save/

    plant_to_keep = []

    # Create a list of image names to process
    images = [os.path.join(image_path, item) for item in os.listdir(image_path) if os.path.splitext(item)[1] == ".jpg"]

    # compute_mean_average_precision(
    #     folder=image_path,
    #     model=model_path,
    #     config_file=config_file,
    #     data_obj=meta_path)

    # save_images_from_video(video_path, os.path.join(save_dir, "images_from_video/"), nb_iter=100)
    # save_detect_to_txt(os.path.join(save_dir, "images_from_video/"), save_dir+'result/', model_path, config_file, meta_path)
    # draw_boxes_folder(os.path.join(save_dir, "images_from_video"), os.path.join(save_dir, "result/"), save_path=save_dir)
    image_vid  = os.path.join(save_dir, "images_from_video")
    save_path  = os.path.join(save_dir, "save_dir")
    annot_path = os.path.join(save_dir, "result")

    filter_detections(video_path, video_param, save_path, yolo_param, k=6)

    # save_detect_to_txt(image_path, save_dir, model_path, config_file, meta_path)
    # convert_yolo_annot_to_XYX2Y2(image_path, save_dir+'ground-truth/', labels_to_names)

    # with open("data/val.txt", "r") as f:
    #     content = f.readlines()
    #
    # content = [item.strip() for item in content]
    # content = [os.path.join("data/img/val", os.path.basename(item)) for item in content]
    #
    # with open("data/val.txt", "w") as f:
    #     for line in content:
    #         f.write(line + "\n")

    # crop_annotation_to_square(image_path, save_dir+'ground-truth', labels_to_names)
    # crop_detection_to_square(image_path, save_dir+'detection-results', model_path, config_file, meta_path)
