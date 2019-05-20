from skimage import io, filters, morphology
import numpy as np
from PIL import Image
from joblib import Parallel, delayed


def __init_(self, images, weights, cfg, meta, plant_to_keep=[], user_name='bipbip', save_dir = 'result_operose/', parallel=True):
    self.images        = images
    self.weights       = weights
    self.cfg           = cfg
    self.meta          = meta,
    self.plant_to_keep = plant_to_keep
    self.user_name     = user_name
    self.save_dir      = save_dir

    def egi_mask(image_path, thresh=1.15):
        image    = io.imread(image)
        image_np = np.array(image).astype(float)

        image_np = 2*image_np[:, :, 1] / (image_np[:, :, 0] + image_np[:, :, 2] + 0.001)
        image_gf = filters.gaussian(image_np, sigma=1, mode='reflect')

        image_bin = image_gf > 1.15

        image_morph = morphology.binary_erosion(image_bin, morphology.disk(3))
        image_morph = morphology.binary_dilation(image_morph, morphology.disk(3))

        image_out = morphology.remove_small_objects(image_morph, 400)
        image_out = morphology.remove_small_holes(image_out, 800)

        return image_out


    def process_image(image, plant_to_keep):
        # Creates and populate XML tree, save plant masks as PGM and XLM file
        # for each images

        img_name  = os.path.split(os.path.splitext(image)[0])[1]
        image_egi = egi_mask(cv.imread(image))
        im_in     = Image.fromarray(np.uint8(255 * image_egi))

        h, w = image_egi.shape

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
            height=h,
            user_name=consort)

        # For every detection save PGM mask and add field to the xml tree
        for detection in detections:
            name = detection[0]

            if (name not in plant_to_keep) and plant_to_keep: continue

            bbox = detection[2]
            xmin, ymin, xmax, ymax = convertBack(bbox[0], bbox[1], bbox[2], bbox[3])
            bbox = [xmin, ymin, xmax, ymax]

            xml_tree.add_mask_zone(plant_type='PlanteInteret', bbox=bbox, name=name)

            box = (xmin, ymin, xmax, ymax)

            im_out = Image.new(mode='1', size=(w, h))
            region = im_in.crop(box)
            im_out.paste(region, box)

            im_out.save('{}{}_{}_{}.pgm'.format(
                save_dir,
                consort,
                img_name,
                str(xml_tree.get_current_mask_id())))

        xml_tree.save('{}{}_{}.xml'.format(
            save_dir,
            consort,
            img_name))
