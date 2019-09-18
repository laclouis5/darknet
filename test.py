from skimage import io, data, filters, feature, color, exposure, morphology
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import os
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from utils import *
from PIL import Image


# def egi_mask(image, thresh=1.15):
#     image_np = np.array(image).astype(float)
#
#     image_np = 2*image_np[:, :, 1] / (image_np[:, :, 0] + image_np[:, :, 2] + 0.001)
#     image_gf = filters.gaussian(image_np, sigma=1, mode='reflect')
#
#     image_bin = image_gf > 1.15
#
#     image_morph = morphology.binary_erosion(image_bin, morphology.disk(3))
#     image_morph = morphology.binary_dilation(image_morph, morphology.disk(3))
#
#     image_out = morphology.remove_small_objects(image_morph, 400)
#     image_out = morphology.remove_small_holes(image_out, 800)
#
#     return image_out

def egi_mask(image, thresh=40):
    image_np  = np.array(image).astype(np.float)
    image_egi = 2 * image_np[:, :, 1] - image_np[:, :, 0] - image_np[:, :, 2]
    image_gf  = filters.gaussian(image_egi, sigma=1, mode='reflect')
    image_bin = image_gf > 40
    image_out = morphology.remove_small_objects(image_bin, 500)
    image_out = morphology.remove_small_holes(image_out, 800)

    return image_out


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


def scatter3d(image, egi_mask):
    x_rand = np.random.randint(0, 2448, 4000)
    y_rand = np.random.randint(0, 2048, 4000)

    list   = []
    colors = []
    for x, y in zip(x_rand, y_rand):
        list.append(image[y, x, :])
        if egi_mask[y, x]:
            colors.append('g')
        else:
            colors.append('k')

    r, g, b = zip(*list)

    # HSV
    image_2 = color.rgb2hsv(image)

    list_2   = []
    for x, y in zip(x_rand, y_rand):
        list_2.append(image_2[y, x, :])

    h, s, v = zip(*list_2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = fig.gca(projection='3d')

    ax.scatter(r, g, b, c=colors)
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    plt.show()

    # ax.scatter(h, s, v, c=colors)
    # ax.set_xlabel("H")
    # ax.set_ylabel("S")
    # ax.set_zlabel("V")
    # plt.show()

def compute_struct_tensor(image_path, w, sigma=1.5):
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)

        border_type = cv.BORDER_REFLECT

        # Gradients
        Gx = cv.Sobel(img, cv.CV_32F, 1, 0, 3, borderType=border_type)
        Gy = cv.Sobel(img, cv.CV_32F, 0, 1, 3, borderType=border_type)

        # Filtered Structure Tensor Components
        Axx = cv.GaussianBlur(Gx * Gx, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=border_type)
        Ayy = cv.GaussianBlur(Gy * Gy, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=border_type)
        Axy = cv.GaussianBlur(Gx * Gy, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=border_type)

        # Eigenvalues
        tmp1 = Axx + Ayy
        tmp2 = (Axx - Ayy) * (Axx - Ayy)
        tmp3 = Axy * Axy
        tmp4 = cv.sqrt(tmp2 + 4.0 * tmp3)

        lambda1 = tmp1 + tmp4
        lambda2 = tmp1 - tmp4

        # Coherency and Orientation
        img_coherency = (lambda1 - lambda2) / (lambda1 + lambda2)
        img_orientation = 0.5 * cv.phase(Axx - Ayy, 2.0 * Axy, angleInDegrees=True)

        return img_coherency, img_orientation


def create_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def read_gt_annotation_file(file_path, img_size):
    bounding_boxes = BoundingBoxes(bounding_boxes=[])
    image_name = os.path.basename(os.path.splitext(file_path)[0] + '.jpg')

    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, x, y, w, h) = int(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
        bounding_boxes.addBoundingBox(BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, imgSize=img_size))

    return bounding_boxes


def yolo_det_to_bboxes(image_name, yolo_detections):
    bboxes = []

    for detection in yolo_detections:
        label      = detection[0]
        confidence = detection[1]
        box        = detection[2]
        (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(box[0], box[1], box[2], box[3])

        bbox = BoundingBox(imageName=image_name, classId=label, x=xmin, y=ymin, w=xmax, h=ymax, typeCoordinates=CoordinatesType.Absolute, classConfidence=confidence, bbType=BBType.Detected, format=BBFormat.XYX2Y2)

        bboxes.append(bbox)

    return BoundingBoxes(bounding_boxes=bboxes)


def save_bboxes_to_txt(bounding_boxes, save_dir):
    # Saves all detections in a BBoxes object as txt file
    names = bounding_boxes.getNames()

    for name in names:
        boxes = bounding_boxes.getBoundingBoxesByImageName(name)
        boxes = [box for box in boxes if box.getBBType() == BBType.Detected]

        string = ""
        for box in boxes:
            label = box.getClassId()
            conf  = box.getConfidence()
            (xmin, ymin, xmax, ymax) = box.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

            string += "{} {} {} {} {} {}\n".format(label, str(conf), str(xmin), str(ymin), str(xmax), str(ymax))

        save_name = os.path.splitext(boxes[0].getImageName())[0] + ".txt"

        with open(os.path.join(save_dir, save_name), 'w') as f:
            f.writelines(string)


def read_detection_txt_file(file_path, img_size):
    bounding_boxes = BoundingBoxes(bounding_boxes=[])
    image_name = os.path.basename(os.path.splitext(file_path)[0] + '.jpg')

    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, conf, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4]), float(det[5])
        bounding_boxes.addBoundingBox(BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Absolute, format=BBFormat.XYX2Y2, imgSize=img_size, bbType=BBType.Detected, classConfidence=conf))

    return bounding_boxes


def parse_yolo_folder(data_dir):
    annotations = os.listdir(data_dir)
    annotations = [os.path.join(data_dir, item) for item in annotations if os.path.splitext(item)[1] == '.txt']
    images = [os.path.splitext(item)[0] + '.jpg' for item in annotations]
    bounding_boxes = BoundingBoxes(bounding_boxes=[])

    for (img, annot) in zip(images, annotations):
        img_size = Image.open(img).size
        image_boxes = read_txt_annotation_file(annot, img_size)
        [bounding_boxes.addBoundingBox(bb) for bb in image_boxes.getBoundingBoxes()]

    return bounding_boxes


def xywh_to_xyx2y2(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def save_yolo_detect_to_txt(yolo_detections, save_name):
    lines = []

    for detection in yolo_detections:
        box = detection[2]
        (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(box[0], box[1], box[2], box[3])
        confidence = detection[1]
        lines.append("{} {} {} {} {} {}\n".format(detection[0], confidence, xmin, ymin, xmax, ymax))

    with open(save_name, 'w') as f:
        f.writelines(lines)


def nms(bboxes, conf_thresh=0.25, nms_thresh=0.4):
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

# image = cv.imread('data/val/im_335.jpg')
# out = egi_mask_2(image)
# plt.imshow(out)
# plt.show()

# image = io.imread("data/carotte.jpg")
# mask = egi_mask(image)
# image_green = image.copy()
# image_green[mask==0] = 0
# # plt.subplot(221)
# # plt.imshow(image)
# # plt.subplot(222)
# # plt.imshow(mask)
# # plt.subplot(223)
# # plt.imshow(image_green)
# # plt.show()
#
# # scatter3d(image, mask)
# #structure_tensor(image)
#
# coherence, orientation = compute_struct_tensor("data/im_33.jpg", 32, 10)
#
# plt.subplot(121)
# plt.title("Coherence")
# plt.imshow(coherence)
# plt.subplot(122)
# plt.title("Orientation")
# plt.imshow(orientation)
# plt.show()
