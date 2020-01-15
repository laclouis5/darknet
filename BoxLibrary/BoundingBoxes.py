from .BoundingBox import BoundingBox
from .utils import *

from random import shuffle
import matplotlib.pyplot as plt
from collections.abc import MutableSequence

class BoundingBoxes(MutableSequence):
    def __init__(self, bounding_boxes=None):
        if bounding_boxes is not None:
            self._boundingBoxes = bounding_boxes
        else:
            self._boundingBoxes = []

    def __len__(self):
        return len(self._boundingBoxes)

    def __getitem__(self, index):
        return self._boundingBoxes[index]

    def __setitem__(self, box):
        self._boundingBoxes.append(box)

    def __delitem__(self, box):
        self._boundingBoxes.remove(box)

    def __add__(self, otherBoxes):
        return BoundingBoxes(self._boundingBoxes + otherBoxes._boundingBoxes)

    def insert(self, index, box):
        self._boundingBoxes.insert(index, box)

    def getClasses(self):
        return sorted(set([box.getClassId() for box in self]))

    def getNames(self):
        return sorted(set([box.getImageName() for box in self]))

    def getBoundingBoxesByType(self, bbType):
        return BoundingBoxes([d for d in self if d.getBBType() == bbType])

    def getBoundingBoxesByImageName(self, imageName):
        return BoundingBoxes([d for d in self if d.getImageName() == imageName])

    def getBoundingBoxByClass(self, classId):
        return BoundingBoxes([bb for bb in self if bb.getClassId() == classId])

    def getDetectionBoxesAsNPArray(self):
        import numpy as np
        detection_boxes = self.getBoundingBoxesByType(BBType.Detected)
        array = [[*box.getAbsoluteBoundingBox(BBFormat.XYX2Y2), box.getConfidence()] for box in detection_boxes]
        return np.array(array)

    def imageSize(self, imageName):
        box = next((box for box in self if box.getImageName() == imageName), None)
        if box is None:
            raise IOError("Image with name '{}' not found, can't return its size.".format(imageName))
        return box.getImageSize()

    def mapLabels(self, mapping):
        for box in self:
            box.mapLabel(mapping)

    def shuffleBoundingBoxes(self):
        shuffle(self._boundingBoxes)

    def count(self, bbType=None):
        if bbType is None:  # Return all bounding boxes
            return len(self)
        count = 0
        for d in self:
            if d.getBBType() == bbType:  # get only specified bb type
                count += 1
        return count

    def stats(self):
        print("{:<20} {:<15} {}".format(
            "Label:", "Nb Images:", "Nb Annotations:"))
        for label in self.getClasses():
            boxes = self.getBoundingBoxByClass(label)
            nb_images = len(set([item.getImageName() for item in boxes]))
            nb_annot = len(boxes)

            print("{:<20} {:<15} {}".format(label, nb_images, nb_annot))

    def save(self, type_coordinates=CoordinatesType.Relative, format=BBFormat.XYWH, save_dir=None):
        """
        Save all bounding boxes as Yolo annotation files in the specified directory.
        """
        for imageName in self.getNames():
            description = ""
            for box in self.getBoundingBoxesByImageName(imageName):
                description += box.description(type_coordinates, format)

            fileName = os.path.splitext(imageName)[0] + ".txt"
            if save_dir is not None:
                create_dir(save_dir)
                fileName = os.path.join(save_dir, os.path.basename(fileName))

            with open(fileName, "w") as f:
                f.write(description)

    def squareStemBoxes(self, ratio=0.075):
        boxes = BoundingBoxes()
        for box in self:
            if "stem" not in box.getClassId():
                boxes.append(box)
            else:
                (x, y, _, _) = box.getRelativeBoundingBox()
                (img_w, img_h) = box.getImageSize()
                new_size = ratio * min(img_w, img_h)
                w = new_size / img_w
                h = new_size / img_h
                boxes.append(BoundingBox(imageName=box.getImageName(), classId=box.getClassId(), x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, imgSize=box.getImageSize(), bbType=box.getBBType(), classConfidence=box.getConfidence, format=BBFormat.XYWH))
        return boxes

    def plotHistogram(self):
        areas = []
        for label in self.getClasses():
            boxes = self.getBoundingBoxByClass(label)
            areas.append([bbox.getArea() for bbox in boxes])
        # plt.hist(areas, bins=200, stacked=True, density=True, label=labels)
        # plt.hist([bbox.getArea() for bbox in boxes], bins=100)
        plt.title('Stacked histogram of bounding box area')
        # plt.title('Histogram for class "{}"'.format(c))
        plt.legend()
        plt.xlabel('Area in pixels')
        plt.ylabel('Normalized histogram')
        # plt.ylabel('Number of bbox')
        plt.show()

    def drawCVImage(self, image, imageName):
        """
        draws boxes for imageName on the CV image and returns it.
        """
        boxes = self.getBoundingBoxesByImageName(imageName)
        for box in boxes:
            box.addIntoImage(image)
        return image

    def drawImage(self, name, save_dir=None):
        """
        draws boxes for 'name' on its corresponding image name specified
        in box.imageName and saves it in save_dir if given.
        """
        import cv2 as cv

        if save_dir is not None:
            create_dir(save_dir)
            save_name = os.path.join(save_dir, os.path.basename(name))

        image = self.drawCVImage(cv.imread(name), name)
        cv.imwrite(save_name, image)

    def drawAll(self, save_dir="annotated_images/"):
        """
        draws all boxes on their correponding image stored in box.imageName and
        saves images in save_dir.
        """
        from joblib import Parallel, delayed

        names = self.getNames()
        save_dir = [save_dir for _ in range(len(names))]

        Parallel(n_jobs=-1, backend="multiprocessing")(delayed(self.drawImage)(name, sd) for (name, sd) in zip(names, save_dir))

    def copy(self):
        newBoundingBoxes = BoundingBoxes([box.copy() for box in self._boundingBoxes])
