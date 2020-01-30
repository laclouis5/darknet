from .utils import *
from math import sqrt

class BoundingBox:
    """
    Represents a bounding box with a classId, associated to an image name and with an optional confidence.

    Constructor Parameters:
        imageName (str):
             The image name.
        classId (str):
            The class id or label.
        x (float):
            The X upper-left coordinate of the bounding box.
        y (float):
            The Y upper-left coordinate of the bounding box.
        w (float):
            The width bounding box.
        h (float):
            The height bounding box.
        typeCoordinates (optional CoordinatesType):
            Enum (Relative or Absolute) representing if the bounding box coordinates (x,y,w,h) are absolute or relative to size of the image. Default:'Absolute'.
        imgSize (optional tuple(float, flaot)):
            2D vector (width, height) represents the size of the image of the bounding box. If typeCoordinates is 'Relative', imgSize is required.
        bbType (optional BBType):
            Enum (Groundtruth or Detection) identifies if the bounding box represents a ground truth or a detection. If it is a detection, the classConfidence has to be informed.
        classConfidence (float):
            The confidence of the detected class. If detectionType is Detection, classConfidence needs to be informed.
        format (optional BBFormat):
            Enum (XYWH, XYX2Y2 or XYC) indicating the format of the coordinates of the bounding boxes. XYWH: <left> <top> <width> <height>, XYX2Y2: <left> <top> <right> <bottom>, XYC: <xCenter> <yCenter> <width> <height>.
    """
    
    def __init__(self, imageName, classId, x, y, w, h, typeCoordinates=CoordinatesType.Absolute, imgSize=None, bbType=BBType.GroundTruth, classConfidence=None, format=BBFormat.XYWH):
        self._imageName = imageName
        self._typeCoordinates = typeCoordinates
        if typeCoordinates == CoordinatesType.Relative and imgSize is None:
            raise IOError(
                'Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if bbType == BBType.Detected and classConfidence is None:
            raise IOError(
                'For bbType=\'Detection\', it is necessary to inform the classConfidence value.')
        # if classConfidence != None and (classConfidence < 0 or classConfidence > 1):
        # raise IOError('classConfidence value must be a real value between 0 and 1. Value: %f' %
        # classConfidence)

        self._classConfidence = classConfidence
        self._bbType = bbType
        self._classId = classId
        self._format = format

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (typeCoordinates == CoordinatesType.Relative):
            if format == BBFormat.XYWH:
                (self._x, self._y, self._x2, self._y2) = convertToAbsoluteValues(imgSize, (x, y, w, h))
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
                self._width_img = imgSize[0]
                self._height_img = imgSize[1]
            else:
                raise IOError(
                    'For relative coordinates, the format must be XYWH (x,y,width,height)')
        # For absolute coords: (x,y,w,h)=real bb coords
        else:
            if format == BBFormat.XYWH:
                self._x = x
                self._y = y
                self._w = w
                self._h = h
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            elif format == BBFormat.XYX2Y2:  # format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                self._x = x
                self._y = y
                self._x2 = w
                self._y2 = h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            else:
                self._w = w
                self._h = h
                self._x = x - self._w / 2.0
                self._y = y - self._h / 2.0
                self._x2 = x + self._w / 2.0
                self._y2 = y + self._h / 2.0

        if imgSize is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = imgSize[0]
            self._height_img = imgSize[1]

    def getAbsoluteBoundingBox(self, format=BBFormat.XYWH):
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)
        elif format == BBFormat.XYC:
            x = (self._x + self._x2) / 2.0
            y = (self._y + self._y2) / 2.0
            return (x, y, self._w, self._h)

    def getRelativeBoundingBox(self, imgSize=None):
        if imgSize is None and (self._width_img is None or self._height_img) is None:
            raise IOError(
                'Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if imgSize is not None:
            return convertToRelativeValues((imgSize[0], imgSize[1]),
                                           (self._x, self._x2, self._y, self._y2))
        else:
            return convertToRelativeValues((self._width_img, self._height_img),
                                           (self._x, self._x2, self._y, self._y2))

    def getImageName(self):
        return self._imageName

    def getConfidence(self):
        return self._classConfidence

    def getFormat(self):
        return self._format

    def getClassId(self):
        return self._classId

    def getImageSize(self):
        return (self._width_img, self._height_img)

    def getCoordinatesType(self):
        return self._typeCoordinates

    def getBBType(self):
        return self._bbType

    def getArea(self):
        area = (self._w + 1) * (self._h + 1)
        return area


    def clip(self, size=None):
        """
        Crops the bounding box to fit into a given rectangle. It correponds to the partition that is common with the given rectangle.

        Parameters:
            size (float, float): The width and height in absolute coordinates of the bounding frame. If size is None and an image size is specified in the BoundingBox object, the last in used.
        """
        if (self._width_img is None or self._height_img is None) and size is None:
            raise IOError('Parameter \'size\' is required. It is necessary to inform the size.')

        def clip(box, size):
            (xmin, ymin, xmax, ymax) = box.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
            if xmin < 0:
                xmin = 0
            if xmax >= size[0]:
                xmax = size[0] - 1
            if ymin < 0:
                ymin = 0
            if ymax >= size[1]:
                ymax = size[1] - 1

            box._x = xmin
            box._y = ymin
            box._x2 = xmax
            box._y2 = ymax
            box._w = xmax - xmin
            box._h = ymax - ymin

        if size is not None:
            clip(self, size)
        else:
            clip(self, self.getImageSize())

    def cliped(self, size=None):
        """
        Crops the bounding box to fit into a given rectangle and returns it as a new box. It correponds to the partition that is common with the given rectangle.

        Parameters:
            size (float, float): The width and height in absolute coordinates of the bounding frame. If size is None and an image size is specified in the BoundingBox object, the last in used.

        Returns:
            box (BoundingBox): The clipped box.
        """
        box = self.copy()
        box.clip(size)
        return box

    def centerIsIn(self, rect=None):
        """
        Returns True if the BoundingBox center is in a given rectangle. If no rectangle is provided, the image size stored in the BoundingBox is used, if there is one.

        Parameters:
            rect [float]: The coordinnates of the rectangle with format [xMin, yMin, xMax, yMax].

        Returns:
            bool: Boolean indicading if the BoundingBox center is in the Rectangle.
        """

        if (self._width_img is None or self._height_img is None) and rect is None:
            raise IOError('Parameter \'rect\' is required. It is necessary to inform it.')

        def centerIsIn(x, y, rect):
            if x < rect[0]: return False
            if x > rect[2]: return False
            if y < rect[1]: return False
            if y > rect[3]: return False
            return True

        (x, y, _, _) = self.getAbsoluteBoundingBox(format=BBFormat.XYC)

        if rect is not None:
            return centerIsIn(x, y, rect)
        else:
            (w, h) = self.getImageSize()
            rect = [0, 0, w, h]
            return centerIsIn(x, y, rect)

    def iou(self, other):
        boxA = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        boxB = other.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

        if not self.intersects(other): return 0

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        intersection = (xB - xA + 1) * (yB - yA + 1)

        areaA = self.getArea()
        areaB = other.getArea()

        union = areaA + areaB - intersection

        return intersection / union

    def intersects(self, other):
        """
        Returns True if the BoundingBox intersects the other BoundingBox.

        Parameters:
            other (BoundingBox): The other box to compare with.

        Returns:
            bool: Boolean indicating if the intersection of the two boxes is not zero.
        """
        boxA = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        boxB = other.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

        return boxA[0] < boxB[2] and boxB[0] < boxA[2] and boxA[3] > boxB[1]  and boxB[3] > boxA[1]

    def distance(self, other):
        """
        Returns the distance from the center of this BoundingBox to the center of another BoundingBox (Euclidian distance).

        Parameters:
            other (BoundingBox): The other box.

        Returns:
            float: The distance between box centers.
        """
        boxA = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        boxB = other.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

        cxa = (boxA[2] + boxA[0]) / 2
        cya = (boxA[3] + boxA[1]) / 2
        cxb = (boxB[2] + boxB[0]) / 2
        cyb = (boxB[3] + boxB[1]) / 2

        vx = cxb - cxa
        vy = cyb - cyb

        dist = sqrt(pow(vx, 2) + pow(vy, 2))

        return dist

    def setClassId(self, new_class_id):
        self._classId = new_class_id

    def description(self, type_coordinates=None, format=None):
        if type_coordinates is None:
            type_coordinates = self._typeCoordinates
        if format is None:
            format = self._format

        if type_coordinates == CoordinatesType.Relative:
            bbox = self.getRelativeBoundingBox()
        else:
            bbox = self.getAbsoluteBoundingBox(format)
            # bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            bbox = (bbox[0], bbox[1], bbox[2], bbox[3])

        if self._bbType == BBType.Detected:
            return "{} {} {} {} {} {}\n".format(self._classId, self._classConfidence, *bbox)
        else:
            return "{} {} {} {} {}\n".format(self._classId, *bbox)


    def addIntoImage(self, image, color=None, thickness=2):
        import cv2

        # Choose color if not specified
        if color is None:
            if self._bbType == BBType.GroundTruth:
                color = (127, 255, 127)
            else:
                color = (255, 100, 100)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontThickness = 1

        x1, y1, x2, y2 = self.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Add label
        label = self._classId
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (x1 + thickness, y1 - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = y1 + th  # put it inside the bb
        r_Xin = x1 - int(thickness / 2)
        r_Yin = y1 - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), color,
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
        return image

    def copy(self):
        return BoundingBox(
            self._imageName,
            self._classId,
            self._x,
            self._y,
            self._w,
            self._h,
            typeCoordinates=self._typeCoordinates,
            imgSize=self.getImageSize(),
            bbType=self._bbType,
            classConfidence=self._classConfidence,
            format=BBFormat.XYWH)
