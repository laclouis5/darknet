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
# import cv2 as cv
from PIL import Image
from skimage import io, filters, morphology
from joblib import Parallel, delayed

from lxml.etree import Element, SubElement, tostring, parse

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

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

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

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
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
    letter_box = 0
    #predict_image_letterbox(net, im)
    #letter_box = 1
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
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


from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import Evaluator
from test import read_txt_annotation_file, parse_yolo_folder
from utils import *

def compute_mean_average_precision(folder, model, config_file, data_obj):
    files = os.listdir(folder)
    images = [os.path.join(folder, file) for file in files if os.path.splitext(file)[1] == '.jpg']

    bounding_boxes = parse_yolo_folder(folder)
    bounding_boxes.mapLabels({0: "mais", 1: 'haricot', 2: 'carotte'})

    for image in images:
        detections = performDetect(
            imagePath=image,
            configPath=config_file,
            weightPath=model,
            metaPath=data_obj,
            showImage=False)

        img_size = Image.open(image).size

        for detection in detections:
            label, conf = detection[0], detection[1]
            # Abs XYX2Y2
            x_min, y_min, x_max, y_max = convertBack(*detection[2])

            bounding_boxes.addBoundingBox(BoundingBox(
                imageName=os.path.basename(image),
                classId=label,
                x=x_min, y=y_min, w=x_max, h=y_max,
                bbType=BBType.Detected, classConfidence=conf,
                format=BBFormat.XYX2Y2,
                imgSize=img_size))

    evaluator = Evaluator()
    metrics = evaluator.GetPascalVOCMetrics(bounding_boxes)
    for item in metrics:
        print("{} - mAP: {:.4} %, TP: {}, FP: {}, tot. pos.: {}".format(item['class'], 100*item['AP'], item["total TP"], item["total FP"], item["total positives"]))


def save_detect_to_txt(images, save_dir, model_path, cfg_path, data_path):
    names = [os.path.split(image)[1] for image in images]
    names = [os.path.join(save_dir, os.path.splitext(name)[0]) + '.txt' for name in names]

    for i, image in enumerate(images):
        detections = performDetect(imagePath=image, configPath=cfg_path, weightPath=model_path, metaPath=data_path, showImage=False)

        with open(names[i], 'w') as f:
            for detection in detections:
                box   = detection[2]
                (x_min, y_min, x_max, y_max) = convertBack(box[0], box[1], box[2], box[3])
                line = "{} {:.4} {} {} {} {}\n".format(detection[0], detection[1], x_min, y_min, x_max, y_max)

                f.write(line)


def detect_on_folder(images, save_dir, model_path, cfg_path, data_path):
    names = [os.path.split(image)[1] for image in images]
    names = [os.path.join(save_dir, os.path.splitext(name)[0] + ('.jpg')) for name in names]

    for i, image in enumerate(images):
        detections = performDetect(imagePath=image, configPath=cfg_path, weightPath=model_path, metaPath=data_path, showImage=False)
        img_out = cvDrawBoxes(detections, cv.imread(image))

        cv.imwrite(names[i], img_out)


def cv_egi_mask(image, thresh=40):
    image_np = np.array(image).astype(np.float32)
    image_np = 2 * image_np[:, :, 1] - image_np[:, :, 0] - image_np[:, :, 2]

    image_gf = cv.GaussianBlur(src=image_np, ksize=(0, 0), sigmaX=3)

    image_bin = image_gf > thresh

    nb_components, output, stats, _ = cv.connectedComponentsWithStats(image_bin.astype(np.uint8), connectivity=8)

    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img_out = np.zeros((output.shape))

    for i in range(0, nb_components):
        if sizes[i] >= 500:
            img_out[output == i + 1] = 255

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    image_morph = cv.morphologyEx(img_out, op=cv.MORPH_CLOSE, kernel=kernel)

    return image_morph


def egi_mask(image, thresh=40):
    image_np  = np.array(image).astype(float)
    image_egi = 2 * image_np[:, :, 1] - image_np[:, :, 0] - image_np[:, :, 2]
    image_gf  = filters.gaussian(image_egi, sigma=1, mode='reflect')
    image_bin = image_gf > thresh

    image_morph = morphology.remove_small_objects(image_bin, 400)
    image_out   = morphology.remove_small_holes(image_morph, 800)

    return image_out


def create_operose_result(image):
    # Creates and populate XML tree, save plant masks as PGM and XLM file
    # for each images

    img_name  = os.path.split(os.path.splitext(image)[0])[1]
    image_egi = egi_mask(io.imread(image))
    im_in     = Image.fromarray(np.uint8(255 * image_egi))
    # image_egi = cv_egi_mask(cv.imread(image))
    # im_in     = Image.fromarray(image_egi)

    h, w = image_egi.shape

    # Perform detection using Darknet[1]
    detections = performDetect(
        imagePath=image,
        configPath=config_file,
        weightPath=model_path,
        metaPath=meta_path,
        showImage=False)

    # XML tree init
    xml_tree = XMLTree(
        image_name=img_name,
        width=w,
        height=h,
        user_name=consort)

    # For every detection save PGM mask and add field to the xml tree
    for detection in detections:
        name = detection[0]

        if (name not in plant_to_keep) and plant_to_keep: continue

        bbox = detection[2]
        xmin, ymin, xmax, ymax = convertBack(bbox[0], bbox[1], bbox[2], bbox[3])
        box = (xmin, ymin, xmax, ymax)

        xml_tree.add_mask_zone(plant_type='PlanteInteret', bbox=box, name=name)

        im_out = Image.new(mode='1', size=(w, h))
        region = im_in.crop(box)
        im_out.paste(region, box)

        im_out.save('{}{}_{}_{}.pgm'.format(
            save_dir,
            consort,
            img_name,
            str(xml_tree.get_current_mask_id())))

    xml_tree.save('{}{}_{}.xml'.format(
        save_dir,
        consort,
        img_name))


if __name__ == "__main__":
    image_path  = "data/val/"
    model_path  = "results/yolov3-tiny_8/yolov3-tiny_obj_7400.weights"
    config_file = "results/yolov3-tiny_8/yolov3-tiny_obj.cfg"
    meta_path   = "results/yolov3-tiny_8/obj.data"

    consort  = 'bipbip'
    save_dir = 'save/'
    # save_dir = /Users/louislac/Downloads/save/

    plant_to_keep = []

    # Create a list of image names to process
    images = os.listdir(image_path)
    images = [os.path.join(image_path, item) for item in images if os.path.splitext(item)[1] == ".jpg"]

    compute_mean_average_precision(
        folder=image_path,
        model=model_path,
        config_file=config_file,
        data_obj=meta_path)

    # Parallel computation for every images
    # Parallel(n_jobs=-1, backend="multiprocessing")(map(
    #     delayed(create_operose_result),
    #     images))
