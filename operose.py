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

from darknet import performDetect, convertBack
from my_xml_toolbox import XMLTree, XMLTree2
from test import egi_mask, cv_egi_mask, create_dir

def create_operose_result(args):
    (image, save_dir, network_params, plants_to_keep) = args

    config_file = network_params["cfg"]
    model_path  = network_params["model"]
    meta_path   = network_params["obj"]

    consort = "Bipbip"

    print(image)

    # Creates and populate XML tree, save plant masks as PGM and XLM file
    # for each images
    img_name = os.path.basename(image)

    try:
        image_egi = cv_egi_mask(cv.imread(image))
        im_in     = Image.fromarray(image_egi)
    except:
        image_egi = egi_mask(io.imread(image))
        im_in     = Image.fromarray(np.uint8(255 * image_egi))

    h, w = image_egi.shape[0:2]

    # Perform detection using Darknet
    detections = performDetect(
        imagePath=image,
        configPath=config_file,
        weightPath=model_path,
        metaPath=meta_path,
        showImage=False)

    # XML tree init
    xml_tree = XMLTree2(
        image_name=img_name,
        width=w,
        height=h)

    # For every detection save PGM mask and add field to the xml tree
    for detection in detections:
        name = detection[0]

        if (plants_to_keep is not None) and (name not in plants_to_keep):
            continue

        bbox = detection[2]
        box  = convertBack(bbox[0], bbox[1], bbox[2], bbox[3])

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
    print("PARAM USED:")
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
    main()
