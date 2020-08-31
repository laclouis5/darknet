try:
    import cv2 as cv
except:
    from skimage import io, filters, morphology
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import os

import argparse
import glob

from darknet import *
from my_library import *
from reg_plane import *
from BoxLibrary import *
from my_xml_toolbox import XMLTree
# from test import egi_mask, cv_egi_mask, create_dir

def create_operose_result(args):
    (image, save_dir, network_params, plants_to_keep) = args

    config_file = network_params["cfg"]
    model_path = network_params["model"]
    meta_path = network_params["obj"]

    consort = "Bipbip"

    # print(image)

    # Creates and populate XML tree, save plant masks as PGM and XLM file
    # for each images
    img_name = os.path.basename(image)

    try:
        image_egi = cv_egi_mask(cv.imread(image))
        im_in = Image.fromarray(image_egi)
    except:
        image_egi = egi_mask(io.imread(image))
        im_in = Image.fromarray(np.uint8(255 * image_egi))

    h, w = image_egi.shape[:2]

    # Perform detection using Darknet
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
        height=h)

    # For every detection save PGM mask and add field to the xml tree
    for detection in detections:
        name = detection[0]
        if (plants_to_keep is not None) and (name not in plants_to_keep):
            continue

        bbox = detection[2]
        box  = convertBack((bbox[0]), bbox[1], bbox[2], bbox[3])
        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        xml_tree.add_mask(name)

        im_out = Image.new(mode='1', size=(w, h))
        region = im_in.crop(box)
        im_out.paste(region, box)

        image_name_out = '{}_{}_{}.png'.format(
            consort,
            os.path.splitext(img_name)[0],
            str(xml_tree.plant_count-1))

        im_out.save(os.path.join(save_dir, image_name_out))

    xml_tree.save(save_dir)


def process_operose(image_path, network_params, save_dir="operose/", plants_to_keep=None, nb_proc=-1):
    create_dir(save_dir)

    def ArgsGenerator(image_path, network_params, save_dir, plants_to_keep):
        images = [os.path.join(image_path, item) for item in os.listdir(image_path) if os.path.splitext(item)[1] == ".jpg"]
        for image in images:
            yield (image, save_dir, network_params, plants_to_keep)

    args = ArgsGenerator(image_path, network_params, save_dir, plants_to_keep)

    Parallel(n_jobs=nb_proc, backend="multiprocessing")(delayed(create_operose_result)(arg) for arg in args)


def operose(txt_file, yolo, label, min_dets, max_dist,
    conf_thresh=0.25, save_dir="operose/"
):
    """Can add a caching functionnality for optical flow values."""
    print("WARNING: This function is hardcoded for images of size (H: 632, W: 632).")
    # TODO:
    # - Reactive programming pipeline for faster speeds

    create_dir(save_dir)
    (cfg, weights, meta) = yolo.get_cfg_weight_meta()
    images = read_image_txt_file(txt_file)
    opt_flow_estimator = OpticalFlow(mask_border=True, mask_egi=True)
    opt_flows = [BivariateFunction(lambda x, y, ex=0.0, ey=0.0: (ex, ey))]
    tracker = Tracker(min_confidence=conf_thresh, min_points=min_dets, dist_thresh=max_dist)
    past_img = None

    for image in images:
        img = cv.imread(image)

        if past_img is not None:
            displacement = opt_flow_estimator.displacement_eq(img, past_img)
            opt_flows.append(displacement)

        detections = performDetect(imagePath=image, thresh=conf_thresh, configPath=cfg, weightPath=weights, metaPath=meta, showImage=False)
        image_boxes = Parser.parse_yolo_darknet_detections(detections, image_name=image, img_size=image_size(image), classes=[label])
        tracker.update(image_boxes, optical_flow=opt_flows[-1])

        past_img = img

    boxes = tracker.get_filtered_boxes()
    boxes = box_association(boxes, images, opt_flows)
    boxes = BoundingBoxes([box for box in boxes if box.centerIsIn([32, 32, 600, 600])])  # Hardcoded

    for image in images:
        image_name = os.path.basename(image)
        image_boxes = boxes.getBoundingBoxesByImageName(image)
        img = cv.imread(image)
        (img_height, img_width) = img.shape[:2]
        radius = int(5/100 * min(img_width, img_height) / 2)

        xml_tree = XMLTree(image_name=image_name, width=img_width, height=img_height)

        for box in image_boxes:
            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)
            label = box.getClassId()
            xml_tree.add_mask(label)
            out_name = f"Bipbip_{os.path.splitext(image_name)[0]}_{xml_tree.plant_count-1}.png"

            stem_mask = Image.new(mode="1", size=(img_width, img_height))
            # stem_mask = Image.open(image)
            box = [int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius]
            stem_mask.paste(Image.new(mode="1", size=(radius*2, radius*2), color=1), box)
            stem_mask.save(os.path.join(save_dir, out_name))

            xml_tree.save(save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_path", help="Path where images for detection are stored.")
    parser.add_argument("save_dir", help="Directory to save masks and xml files.")
    parser.add_argument("model", help="Directory containing cfg, data ans weights files.")
    parser.add_argument("-l", action="append", dest="labels", help="Labels to keep. Default is all.")
    parser.add_argument("-nproc", type=int, help="Number of proc to use to speed up inference. Default is 1. -1 for using all available procs.")
    args = parser.parse_args()

    image_path = os.path.join(args.images_path)
    save_dir_operose = os.path.join(args.save_dir)
    keep_challenge = args.labels
    path = args.model
    nproc = args.nproc

    model_path  = glob.glob(os.path.join(path, "*.weights"))[0]
    config_file = glob.glob(os.path.join(path, "*.cfg"))[0]
    meta_path   = glob.glob(os.path.join(path, "*.data"))[0]
    yolo_param  = {"model": model_path, "cfg": config_file, "obj": meta_path}

    print()
    print("PARAM USED: ")
    print("Image directory: {}".format(image_path))
    print("Save directory: {}".format(save_dir_operose))
    if keep_challenge is None:
        print("Labels to keep: all")
    else:
        print("Labels to keep: {}".format(keep_challenge))
    if nproc is None:
        print("Number of proc used: 1")
    else:
        print("Number of proc used: {}".format(nproc))
    print("Model in use: ")
    print("  {}".format(model_path))
    print("  {}".format(config_file))
    print("  {}".format(meta_path))
    print()

    if nproc is not None:
        process_operose(image_path, yolo_param, plants_to_keep=keep_challenge, save_dir=save_dir_operose, nb_proc=nproc)
    else:
        process_operose(image_path, yolo_param, plants_to_keep=keep_challenge, save_dir=save_dir_operose, nb_proc=1)


if __name__ == "__main__":
    # main()
    yolo = YoloModelPath("results/yolov4-tiny_6")
    (cfg, weights, obj) = yolo.get_cfg_weight_meta()
    yolo_params = {"model": weights, "obj": obj, "cfg": cfg}

    # operose(txt_file="data/haricot_debug_long_2.txt", yolo=YoloModelPath("results/yolov4-tiny_1"), label="stem_bean", min_dets=4, max_dist=9/100)
    process_operose("/media/deepwater/DATA/Shared/Louis/datasets/haricot_debug_montoldre_2/", network_params=yolo_params, save_dir="operose/", plants_to_keep=["bean"])
