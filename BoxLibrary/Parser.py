from .BoundingBox import BoundingBox
from .BoundingBoxes import BoundingBoxes
from .utils import *
import os
import lxml.etree as ET
import PIL
import json

class Parser:
    @staticmethod
    def parse_xml_directories(directories, classes=None):
        boxes = BoundingBoxes()

        for directory in directories:
            boxes += Parser.parse_xml_folder(directory, classes=classes)

        return boxes

    @staticmethod
    def parse_xml_folder(folder, classes=None):
        boxes = BoundingBoxes()

        for file in files_with_extension(folder, ".xml"):
            boxes += Parser.parse_xml_file(file, classes=classes)

        return boxes

    @staticmethod
    def parse_xml_file(file, classes=None):
        boxes = BoundingBoxes()
        tree = ET.parse(file).getroot()

        if classes is not None:
            classes = [str(item) for item in classes]

        name = os.path.join(os.path.dirname(tree.find('path').text), tree.find('filename').text)
        width = tree.find('size').find('width').text
        height = tree.find('size').find('height').text

        for object in tree.findall('object'):
            class_id = object.find('name').text
            if classes and (class_id not in classes):
                continue

            xmin = float(object.find('bndbox').find('xmin').text)
            ymin = float(object.find('bndbox').find('ymin').text)
            xmax = float(object.find('bndbox').find('xmax').text)
            ymax = float(object.find('bndbox').find('ymax').text)

            box = BoundingBox(name, class_id, xmin, ymin, xmax, ymax, format=BBFormat.XYX2Y2, imgSize=(int(width), int(height)))
            boxes.append(box)

        return boxes

    @staticmethod
    def parse_yolo_gt_directories(directories, classes=None):
        boxes = BoundingBoxes()

        for directory in directories:
            boxes += Parser.parse_yolo_gt_folder(directory, classes=classes)

        return boxes

    @staticmethod
    def parse_yolo_gt_folder(folder, classes=None):
        boxes = BoundingBoxes()

        for file in files_with_extension(folder, ".txt"):
            boxes += Parser.parse_yolo_gt_file(file, classes=classes)

        return boxes

    @staticmethod
    def parse_yolo_det_folder(folder, img_folder, classes=None, bbFormat=BBFormat.XYWH, typeCoordinates=CoordinatesType.Relative):
        boxes = BoundingBoxes()

        for file in files_with_extension(folder, ".txt"):
            image_name = os.path.join(img_folder, os.path.basename(os.path.splitext(file)[0] + ".jpg"))
            img_size = PIL.Image.open(image_name).size
            boxes += Parser.parse_yolo_det_file(file, img_size, classes, bbFormat, typeCoordinates)

        return boxes

    @staticmethod
    def parse_yolo_gt_file(file, classes=None):
        '''
        Designed to read Yolo annotation files that are in the same folders
        as their corresponding image.
        '''
        boxes = BoundingBoxes()
        image_name = os.path.splitext(file)[0] + '.jpg'
        img_size = PIL.Image.open(image_name).size

        if classes:
            classes = [str(item) for item in classes]

        content = open(file, "r").readlines()
        content = [line.strip().split() for line in content]

        for det in content:
            (label, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4])

            if classes and label not in classes:
                continue

            box = BoundingBox(imageName=image_name, classId=label,x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, format=BBFormat.XYWH, imgSize=img_size, bbType=BBType.GroundTruth)

            boxes.append(box)

        return boxes

    @staticmethod
    def parse_yolo_det_file(file, img_size=None, classes=None, bbFormat=BBFormat.XYWH, typeCoordinates=CoordinatesType.Relative):
        """
        If coordinates are relative you should provide img_size.
        """
        boxes = BoundingBoxes()
        image_name = os.path.splitext(file)[0] + '.jpg'

        if classes:
            classes = [str(item) for item in classes]

        content = open(file, "r").readlines()
        content = [line.strip().split() for line in content]

        for det in content:
            (label, confidence, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4]), float(det[5])

            if classes and label not in classes:
                continue

            box = BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, classConfidence=confidence, typeCoordinates=typeCoordinates, format=bbFormat, imgSize=img_size, bbType=BBType.Detected)
            boxes.append(box)

        return boxes

    @staticmethod
    def parse_yolo_darknet_detections(detections, image_name, img_size=None, classes=None):
        """
        Parses a detection returned by yolo detector wrapper.
        """
        boxes = BoundingBoxes()
        if classes:
            classes = [str(item) for item in classes]

        for detection in detections:
            (label, confidence, box) = detection
            (x, y, w, h) = box

            if classes and label not in classes:
                continue

            box = BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, classConfidence=confidence, typeCoordinates=CoordinatesType.Absolute, format=BBFormat.XYC, imgSize=img_size, bbType=BBType.Detected)
            boxes.append(box)

        return boxes

    @staticmethod
    def parse_coco(gt, det, image_path=None):
        (images, categories, img_sizes) = Parser.parse_coco_params(gt)
        gt_boxes = Parser.parse_coco_gt(gt, image_path)
        det_boxes = Parser.parse_coco_det(det, images, categories, img_sizes, image_path)

        return gt_boxes + det_boxes

    @staticmethod
    def parse_coco_gt(gt, img_path=None):
        gt_dict = json.load(open(gt, "r"))

        categories = {item["id"]: item["name"] for item in gt_dict["categories"]}
        images = {item["id"]: item["file_name"] for item in gt_dict["images"]}
        img_sizes = {item["id"]: (item["width"], item["height"]) for item in gt_dict["images"]}

        boxes = BoundingBoxes()

        for annotation in gt_dict["annotations"]:
            img_name = images[annotation["image_id"]]
            if img_path is not None:
                img_name = os.path.join(img_path, img_name)
            label = categories[annotation["category_id"]]
            (x, y, w, h) = annotation["bbox"]
            (width, height) = img_sizes[annotation["image_id"]]

            box = BoundingBox(imageName=img_name, classId=label, x=float(x), y=float(y), w=float(w), h=float(h), imgSize=(int(width), int(height)), bbType=BBType.GroundTruth, format=BBFormat.XYWH)

            boxes.append(box)

        return boxes

    @staticmethod
    def parse_coco_det(det, images, categories, img_sizes=None, img_path=None):
        det_dict = json.load(open(det, "r"))

        boxes = BoundingBoxes()

        for detection in det_dict:
            img_name = images[detection["image_id"]]
            if img_path is not None:
                img_name = os.path.join(img_path, img_name)

            label = categories[detection["category_id"]]
            (x, y, w, h) = detection["bbox"]
            confidence = detection["score"]

            if img_sizes is None:
                img_size = None
            else:
                (width, height) = img_sizes[detection["image_id"]]
                img_size = (int(width), int(height))

            box = BoundingBox(imageName=img_name, classId=label, x=float(x), y=float(y), w=float(w), h=float(h), imgSize=img_size, bbType=BBType.Detected, format=BBFormat.XYWH, classConfidence=float(confidence))

            boxes.append(box)

        return boxes

    @staticmethod
    def parse_coco_params(gt):
        gt_dict = json.load(open(gt, "r"))

        categories = {item["id"]: item["name"] for item in gt_dict["categories"]}
        images = {item["id"]: item["file_name"] for item in gt_dict["images"]}
        img_sizes = {item["id"]: (item["width"], item["height"]) for item in gt_dict["images"]}

        return (images, categories, img_sizes)
