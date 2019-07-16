from BoundingBox import *
from utils import *
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

class BoundingBoxes:
    def __init__(self, bounding_boxes=[]):
        self._boundingBoxes = bounding_boxes

    def repr(self):
        repr = ""
        for d in self._boundingBoxes:
            box = d.getRelativeBoundingBox()
            if d.getBBType() is BBType.GroundTruth:
                repr += "{} {} {} {} {}\n".format(d.getClassId(), box[0], box[1], box[2], box[3])
            elif d.getBBType() is BBType.Detected:
                    repr += "{} {} {} {} {} {}\n".format(d.getClassId(), d.getConfidence(), box[0], box[1], box[2], box[3])
        return repr

    def addBoundingBox(self, bb):
        if isinstance(bb, list):
            self._boundingBoxes.extend(bb)
        else:
            self._boundingBoxes.append(bb)

    def removeBoundingBox(self, _boundingBox):
        for d in self._boundingBoxes:
            if d == _boundingBox:
                del self._boundingBoxes[d]
                return

    def removeAllBoundingBoxes(self):
        self._boundingBoxes = []

    def getBoundingBoxes(self):
        return self._boundingBoxes

    def getBoundingBoxByClass(self, classId):
        boundingBoxes = []
        for d in self._boundingBoxes:
            if d.getClassId() == classId:  # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def mapLabels(self, dic):
        for bbox in self._boundingBoxes:
            bbox._classId = dic[bbox.getClassId()]

    def shuffleBoundingBoxes(self):
        shuffle(self._boundingBoxes)

    def getClasses(self):
        classes = []
        for d in self._boundingBoxes:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return sorted(classes)

    def getNames(self):
        names = []
        for bbox in self._boundingBoxes:
            name = bbox.getImageName()
            if name not in names:
                names.append(name)
        return sorted(names)

    def getBoundingBoxesByType(self, bbType):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getBBType() == bbType]

    def getBoundingBoxesByImageName(self, imageName):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getImageName() == imageName]

    def plotHistogram(self):
        areas   = []
        classes = sorted(self.getClasses())
        labels  = ['{}'.format(c) for c in classes]
        for c in classes:
            boxes = self.getBoundingBoxByClass(c)
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

    def __len__(self):
        return len(self._boundingBoxes)

    def count(self, bbType=None):
        if bbType is None:  # Return all bounding boxes
            return len(self._boundingBoxes)
        count = 0
        for d in self._boundingBoxes:
            if d.getBBType() == bbType:  # get only specified bb type
                count += 1
        return count

    def stats(self):
        classes = self.getClasses()
        stats = dict.fromkeys(classes, 0)
        stats_image = dict.fromkeys(classes, 0)

        nb_images = len(self.getNames())

        for bbox in self._boundingBoxes:
            stats[bbox.getClassId()] += 1

        for image_name in self.getNames():
            boxes = self.getBoundingBoxesByImageName(image_name)
            label = boxes[0].getClassId()
            stats_image[label] += 1

        print("Total number of images: {}".format(nb_images))
        for (key, val) in stats_image.items():
            print("  {}: {}".format(key, val))
        print("Total number of bounding boxes: {}".format(self.count()))
        for (key, val) in stats.items():
            print("  {}: {}".format(key, val))


    def clone(self):
        newBoundingBoxes = BoundingBoxes()
        for d in self._boundingBoxes:
            det = BoundingBox.clone(d)
            newBoundingBoxes.addBoundingBox(det)
        return newBoundingBoxes

    def drawAllBoundingBoxes(self, image, imageName):
        bbxes = self.getBoundingBoxesByImageName(imageName)
        for bb in bbxes:
            if bb.getBBType() == BBType.GroundTruth:  # if ground truth
                image = add_bb_into_image(image, bb, color=(0, 255, 0))  # green
            else:  # if detection
                image = add_bb_into_image(image, bb, color=(255, 0, 0))  # red
        return image

    # def drawAllBoundingBoxes(self, image):
    #     for gt in self.getBoundingBoxesByType(BBType.GroundTruth):
    #         image = add_bb_into_image(image, gt ,color=(0,255,0))
    #     for det in self.getBoundingBoxesByType(BBType.Detected):
    #         image = add_bb_into_image(image, det ,color=(255,0,0))
    #     return image
