# Created by Louis LAC 2019

from skimage import io, data, filters, feature, color, exposure, morphology
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import cv2 as cv
except:
    pass
import os
from PIL import Image
from BoxLibrary import *

def egi_mask(image, thresh=40):
    '''
    Takes as input a numpy array describing an image and return a
    binary mask thresholded over the Excess Green Index.
    '''
    image_np  = np.array(image).astype(np.float)
    image_egi = 2 * image_np[:, :, 1] - image_np[:, :, 0] - image_np[:, :, 2]
    image_gf  = filters.gaussian(image_egi, sigma=1, mode='reflect')
    image_bin = image_gf > 40
    image_out = morphology.remove_small_objects(image_bin, 500)
    image_out = morphology.remove_small_holes(image_out, 800)

    return image_out


def cv_egi_mask(image, thresh=40):
    '''
    Takes as input a numpy array describing an image and return a
    binary mask thresholded over the Excess Green Index. OpenCV implementaition.
    '''
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


def create_dir(directory):
    '''
    Creates the spedified directory if doesn't exist.
    '''
    if not os.path.isdir(directory):
        os.mkdir(directory)


# Obsolete
def yolo_det_to_bboxes(image_name, yolo_detections):
    '''
    Takes as input a list of tuples (label, conf, x, w, w, h) predicted by
    yolo framework and returns a boundingBoxes object representing the boxes.
    image_name is required.
    '''
    bboxes = []

    for detection in yolo_detections:
        label = detection[0]
        confidence = detection[1]
        box = detection[2]
        (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(box[0], box[1], box[2], box[3])

        bbox = BoundingBox(imageName=image_name, classId=label, x=xmin, y=ymin, w=xmax, h=ymax, typeCoordinates=CoordinatesType.Absolute, classConfidence=confidence, bbType=BBType.Detected, format=BBFormat.XYX2Y2)

        bboxes.append(bbox)

    return BoundingBoxes(bounding_boxes=bboxes)

# Obsolete: see BoundingBoxes.save() and BoundingBox.description()
def save_bboxes_to_txt(bounding_boxes, save_dir):
    '''
    Saves boxes wrapped in a boundingBoxes object to a yolo annotation file in
    the specified save_dir directory. Format is XYX2Y2 abs (hard coded).
    '''
    # Saves all detections in a BBoxes object as txt file
    names = bounding_boxes.getNames()

    for name in names:
        boxes = bounding_boxes.getBoundingBoxesByImageName(name)
        boxes = [box for box in boxes if box.getBBType() == BBType.Detected]

        string = ""
        for box in boxes:
            label = box.getClassId()
            conf = box.getConfidence()
            (xmin, ymin, xmax, ymax) = box.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

            string += "{} {} {} {} {} {}\n".format(label, str(conf), str(xmin), str(ymin), str(xmax), str(ymax))

        save_name = os.path.splitext(boxes[0].getImageName())[0] + ".txt"

        with open(os.path.join(save_dir, save_name), 'w') as f:
            f.writelines(string)

# Obsolete: see Parser class
def read_detection_txt_file(file_path, img_size=None):
    '''
    Takes a detection file and its correponding image size and returns
    a boundingBoxes object representing boxes. Detection file is XYX2Y2 abs
    (hard coded).
    '''
    bounding_boxes = BoundingBoxes(bounding_boxes=[])
    image_name = os.path.basename(os.path.splitext(file_path)[0] + '.jpg')

    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, conf, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4]), float(det[5])
        bounding_boxes.addBoundingBox(BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Absolute, format=BBFormat.XYX2Y2, imgSize=img_size, bbType=BBType.Detected, classConfidence=conf))

    return bounding_boxes

# Obsolete: see Parser class
def read_gt_annotation_file(file_path, img_size=None):
    '''
    Takes a yolo GT file and its correponding image size and returns
    a boundingBoxes object representing boxes. Yolo format is XYWH relative,
    image_size must be provided.
    '''
    bounding_boxes = BoundingBoxes(bounding_boxes=[])
    image_name = os.path.basename(os.path.splitext(file_path)[0] + '.jpg')

    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4])
        bounding_boxes.addBoundingBox(BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, imgSize=img_size))

    return bounding_boxes

# Obsolete: see Parser class
def parse_yolo_folder(data_dir):
    '''
    Parsed a folder containing yolo GT annotations and their corresponding
    images with the same name. Returns a boundingBoxes object.
    '''
    annotations = [os.path.join(data_dir, item) for item in os.listdir(data_dir) if os.path.splitext(item)[1] == '.txt']
    images = [os.path.splitext(item)[0] + '.jpg' for item in annotations]
    bounding_boxes = BoundingBoxes(bounding_boxes=[])

    for (img, annot) in zip(images, annotations):
        img_size = Image.open(img).size
        image_boxes = read_gt_annotation_file(annot, img_size)
        [bounding_boxes.addBoundingBox(bb) for bb in image_boxes.getBoundingBoxes()]

    return bounding_boxes


def xywh_to_xyx2y2(x, y, w, h):
    '''
    Takes as input absolute coords and returns integers.
    '''
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def xywh_to_xyx2y2_float(x, y, w, h):
    '''
    Takes as input absolute coords and returns integers.
    '''
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax


def xyx2y2_to_xywh(xmin, ymin, xmax, ymax):
    x = (xmax + xmin) / 2.0
    y = (ymax + ymin) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return x, y, w, h

# Obsolete
def save_yolo_detect_to_txt(yolo_detections, save_name):
    """
    Takes a list of yolo detections (tuples returned by the framework) and
    saved those detections in XYX2Y2 abs format in save_name file.
    """
    lines = []

    for detection in yolo_detections:
        box = detection[2]
        (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(box[0], box[1], box[2], box[3])
        confidence = detection[1]
        lines.append("{} {} {} {} {} {}\n".format(detection[0], confidence, xmin, ymin, xmax, ymax))

    with open(save_name, 'w') as f:
        f.writelines(lines)


def nms(bboxes, conf_thresh=0.25, nms_thresh=0.1):
    """
    Wrapper for OpenCV NMS.
    Takes as input a boundingBoxes object containg ONLY boxes for one images
    and returns filtered boxes.
    This function is not finished.
    """
    labels = bboxes.getClasses()
    filtered_boxes = []

    for label in labels:
        boxes_label = bboxes.getBoundingBoxByClass(label)
        boxes = [box.getAbsoluteBoundingBox(BBFormat.XYWH) for box in boxes_label]
        boxes = [[box[0], box[1], box[2], box[3]] for box in boxes]
        conf  = [box.getConfidence() for box in boxes_label]

        indices = cv.dnn.NMSBoxes(boxes, conf, conf_thresh, nms_thresh)
        indices = [index for list in indices for index in list]

        boxes_to_keep = np.array(boxes_label)[indices]
        boxes_to_keep = boxes_to_keep.tolist()

        filtered_boxes += boxes_to_keep

    return BoundingBoxes(bounding_boxes=filtered_boxes)


def remap_yolo_GT_file_labels(file_path, to_keep):
    """
    Takes path to yolo GT file, reads the file and removes lines with
    the labels specified in the to_keep mapping list.
    """
    content = []
    with open(file_path, "r") as f_read:
        content = f_read.readlines()

    content = [c.strip().split() for c in content]
    content = [line for line in content if (line[0] in to_keep.keys())]

    with open(file_path, "w") as f_write:
        for line in content:
            f_write.write("{} {} {} {} {}\n".format(to_keep[line[0]], line[1], line[2], line[3], line[4]))


def remap_yolo_GT_files_labels(folder, to_keep):
    files = [os.path.join(folder, item) for item in os.listdir(folder) if os.path.splitext(item)[1] == ".txt"]
    for file in files:
        remap_yolo_GT_file_labels(file, to_keep)


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
                    if (x - w / 2.0) < float(w_lim_1) / img_w:
                        print("Spans out of left bound")
                        # Then adjust bbox to fit in the square
                        w = w - (float(w_lim_1) / img_w - (x - w / 2.0))
                        x = float(w_lim_1 + 1) / img_w + w / 2.0
                    if (x + w / 2.0) > float(w_lim_2) / img_w:
                        print("Span out of right bound")
                        w = w - (x + w / 2.0 - float(w_lim_2) / img_w)
                        x = float(w_lim_2) / img_w - w / 2.0
                    else:
                        print("Does not spans outside")

                # If out of bounds...
                else:
                    print("Out of square bounds")
                    # ...do not process the line
                    continue

                # Do not forget to convert from old coord sys to new one
                x = (x * img_w - float(w_lim_1)) / float(w_lim_2 - w_lim_1)
                w = w * img_w / float(w_lim_2 - w_lim_1)

                assert x >= 0, "Value was {}".format(x)
                assert x <= 1, "Value was {}".format(x)
                assert (x - w / 2) >= 0, "Value was {}".format(x - w / 2)
                assert (x + w / 2) <= 1, "Value was {}".format(x + w / 2)

                size = min(img_w, img_h)

                (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(x * size, y * size, w * size, h * size)

                new_line = "{} {} {} {} {}\n".format(lab_to_name[label], xmin, ymin, xmax, ymax)
                content_out.append(new_line)

        # Write updated content to TXt file
        with open(os.path.join(save_dir, os.path.basename(annotation)), 'w') as f:
            f.writelines(content_out)


def crop_detection_to_square(image_path, save_dir, model, config_file, meta_file):
    images = files_with_extension(image_path, ".jpg")
    images.sort()

    for image in images:
        content_out = []
        (img_w, img_h) = Image.open(image).size
        (w_lim_1, w_lim_2) = round(float(img_w) / 2 - float(img_h) / 2), round(float(img_w) / 2 + float(img_h) / 2)

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
            (x, y, w, h) = (x / img_w, y / img_h, w / img_w, h / img_h)

            # If bbox is not out of the new square frame
            if not (x * img_w < w_lim_1 or x * img_w > w_lim_2):
                # But if bbox spans out of one bound (l or r)
                if x - w / 2.0 < float(w_lim_1) / img_w:
                    # Then adjust bbox to fit in the square
                    w = w - (float(w_lim_1) / img_w - (x - w / 2.0))
                    x = float(w_lim_1 + 1) / img_w + w / 2.0
                if x + w / 2.0 > float(w_lim_2) / img_w:
                    w = w - (x + w / 2.0 - float(w_lim_2) / img_w)
                    x = float(w_lim_2) / img_w - w / 2.0

            else: continue

            # Do not forget to convert from old coord sys to new one
            x = (x * img_w - float(w_lim_1)) / float(w_lim_2 - w_lim_1)
            w = w * img_w / float(w_lim_2 - w_lim_1)

            assert x >= 0, "Value was {}".format(x)
            assert x <= 1, "Value was {}".format(x)
            assert (x - w / 2) >= 0, "Value was {}".format(x - w / 2)
            assert (x + w / 2) <= 1, "Value was {}".format(x + w / 2)

            size = min(img_w, img_h)

            (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(x * size, y * size, w * size, h * size)

            new_line = "{} {} {} {} {} {}\n".format(label, prob, xmin, ymin, xmax, ymax)
            content_out.append(new_line)

        # Write updated content to TXT file
        save_name = os.path.splitext(os.path.basename(image))[0] + '.txt'
        with open(os.path.join(save_dir, save_name), 'w') as f:
            f.writelines(content_out)


def clip_box_to_size(box, size):
    (x, y, w, h) = box # Absolute size
    (im_w, im_h) = size

    # Max length
    l = max(w, h)
    if l >= min(im_w, im_h):
        l = min(im_w, im_h)

    # Make it square, expand a little
    new_x = x
    new_y = y
    new_w = l + 0.075 * min(im_w, im_h)
    new_h = l + 0.075 * min(im_w, im_h)

    # Then clip shape to stay in original image
    xmin, ymin, xmax, ymax = xywh_to_xyx2y2(new_x, new_y, new_w, new_h)
    if xmin < 0:
        new_x = x - xmin
    if xmax >= im_w:
        new_x = x - (xmax - im_w)
    if ymin < 0:
        new_y = y - ymin
    if ymax >= im_h:
        new_y = y - (ymax - im_h)

    return  (new_x, new_y, new_w, new_h)
