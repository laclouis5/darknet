from enum import Enum
import os
import PIL

class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    EveryPointInterpolation = 0
    ElevenPointInterpolation = 1


class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    Relative = 0
    Absolute = 1


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    GroundTruth = 0
    Detected = 1


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    It can be (X,Y,width,height) => XYWH
    or (X1,Y1,X2,Y2) => XYX2Y2

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    XYWH = 0
    XYX2Y2 = 1
    XYC = 2

# Equivalent to xyx2y2_to_xywh()
def convertToAbsCenterValues(xmin, ymin, xmax, ymax):
    x = (xmax + xmin) / 2.0
    y = (ymax + ymin) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return (x, y, w, h)

def convertToRelativeValues(size, box):
     # box is (xmin, ymin, xmax, ymax)
    dw = 1. / (size[0]) # Width
    dh = 1. / (size[1]) # Height
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    return (x, y, w, h)

def convertToAbsoluteValues(size, box):
    # box is (x, y, w, h)
    xIn = (2 * float(box[0]) - float(box[2])) * size[0] / 2
    yIn = (2 * float(box[1]) - float(box[3])) * size[1] / 2
    xEnd = xIn + float(box[2]) * size[0]
    yEnd = yIn + float(box[3]) * size[1]
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    # (xMin, yMin, Xmax, yMax)
    return (xIn, yIn, xEnd, yEnd)

def files_with_extension(folder, extension):
    return [os.path.join(folder, item)
            for item in os.listdir(folder)
            if os.path.splitext(item)[1] == extension]

def create_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

def image_size(image):
    return PIL.Image.open(image).size
