try:
    import cv2 as cv
except:
    from skimage import io, filters, morphology
<<<<<<< HEAD
=======

>>>>>>> 88757f85ba576d7d0a5cb97fce6c2616a40d9bc2
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import os

from darknet import performDetect, convertBack
from my_xml_toolbox import XMLTree
from test import egi_mask, cv_egi_mask, create_dir

def create_operose_result(args):
    (image, save_dir, network_params, plants_to_keep) = args

    config_file = network_params["cfg"]
    model_path  = network_params["model"]
    meta_path   = network_params["obj"]

<<<<<<< HEAD
    print(args)

=======
>>>>>>> 88757f85ba576d7d0a5cb97fce6c2616a40d9bc2
    consort = "Bipbip"

    # Creates and populate XML tree, save plant masks as PGM and XLM file
    # for each images
    img_name  = os.path.basename(image)
<<<<<<< HEAD
    image_egi = egi_mask(io.imread(image))
    im_in     = Image.fromarray(np.uint8(255 * image_egi))
=======
    try:
        image_egi = cv_egi_mask(cv.imread(image))
        im_in     = Image.fromarray(image_egi)
    except:
        image_egi = egi_mask(io.imread(image))
        im_in     = Image.fromarray(np.uint8(255 * image_egi))
>>>>>>> 88757f85ba576d7d0a5cb97fce6c2616a40d9bc2

    h, w = image_egi.shape[0:2]

    # Perform detection using Darknet
    detections = performDetect(
        imagePath=image,
        configPath=config_file,
        weightPath=model_path,
        metaPath=meta_path,
        showImage=False)

<<<<<<< HEAD
    [print(detection[2]) for detection in detections]

=======
>>>>>>> 88757f85ba576d7d0a5cb97fce6c2616a40d9bc2
    # XML tree init
    xml_tree = XMLTree(
        image_name=img_name,
        width=w,
        height=h,
        user_name=consort)

    # For every detection save PGM mask and add field to the xml tree
    for detection in detections:
        name = detection[0]

        if plants_to_keep is not None:
            if name in plants_to_keep:
                continue

        bbox = detection[2]
        box  = convertBack(bbox[0], bbox[1], bbox[2], bbox[3])

        xml_tree.add_mask_zone(plant_type='PlanteInteret', bbox=box, name=name)

        im_out = Image.new(mode='1', size=(w, h))
        region = im_in.crop(box)
        im_out.paste(region, box)

        im_out.save('{}{}_{}_{}.pgm'.format(
            save_dir,
            consort,
            os.path.splitext(img_name)[0],
            str(xml_tree.get_current_mask_id())))

    xml_tree.save('{}{}_{}.xml'.format(
        save_dir,
        consort,
        os.path.splitext(img_name)[0]))


def process_operose(image_path, network_params, save_dir="operose/", plants_to_keep=None, nb_proc=4):
    create_dir(save_dir)

    def ArgsGenerator(image_path, network_params, save_dir, plants_to_keep):
        images = [os.path.join(image_path, item) for item in os.listdir(image_path) if os.path.splitext(item)[1] == ".jpg"]
        for image in images:
            yield (image, save_dir, network_params, plants_to_keep)

    args = ArgsGenerator(image_path, network_params, save_dir, plants_to_keep)

    Parallel(n_jobs=nb_proc, backend="multiprocessing")(delayed(create_operose_result)(arg) for arg in args)


if __name__ == "__main__":
    image_path = "data/val/"

    model_path  = "results/yolo_v3_tiny_pan3_1/yolo_v3_tiny_pan3_aa_ae_mixup_scale_giou_best.weights"
    config_file = "results/yolo_v3_tiny_pan3_1/yolo_v3_tiny_pan3_aa_ae_mixup_scale_giou.cfg"
    meta_path   = "results/yolo_v3_tiny_pan3_1/obj.data"

    yolo_param = {"model": model_path, "cfg": config_file, "obj": meta_path}

    keep_challenge   = ["maize", "bean"]
    save_dir_operose = os.path.join("save/operose/")

<<<<<<< HEAD
    process_operose(image_path, yolo_param, plants_to_keep=keep_challenge, save_dir=save_dir_operose, nb_proc=1)
=======
    process_operose(image_path, yolo_param, plants_to_keep=keep_challenge, save_dir=save_dir_operose)
>>>>>>> 88757f85ba576d7d0a5cb97fce6c2616a40d9bc2
