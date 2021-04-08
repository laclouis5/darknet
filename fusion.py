import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.contrib import tenumerate, tzip
import torch
from torch.nn.functional import max_pool2d


def read_image_file(path):
    with path.open() as f:
        data = f.readlines()
        data = [line.strip() for line in data]
        return [Path(path) for path in data]

def read_optical_flow(path):
    with path.open() as f:
        data = f.readlines()
        data = [line.strip().split(",") for line in data]
        data = [[float(line[0]), float(line[1])] for line in data]
        return np.array(data)
 
def read_annotation(path):
    data = json.load(path.open())
    annotations = []

    for obj in data["objects"]:
        for part in obj["parts"]:
            if part["kind"] != "stem": 
                continue

            loc = part["location"]
            x, y = loc["x"], loc["y"]
            annotations.append((x, y))

    return np.array(annotations, dtype=float)

def annotation_for_image(path):
    path = path.with_suffix(".json")
    
    try: 
        return read_annotation(path)
    except: 
        return None

# def read_mat_file(path):
#     return loadmat(path)["heatmaps"][0]  # Bean heatmap

def read_image(path, reshape_size):
    img = Image.open(path).resize(reshape_size)
    return np.array(img)

def read_mat_file(path):
    mat = loadmat(path)
    heatmaps = mat["heatmaps"]
    part_heatmaps = mat["part_heatmaps"]
    offsets = mat["offsets"]
    embeddings = mat["embeddings"]
    return heatmaps, part_heatmaps, offsets, embeddings

def heatmap_for_image(path):
    return read_mat_file(path.with_suffix(".mat"))[0]

def mat_data_for_image(path):
    return read_mat_file(path.with_suffix(".mat"))

def create_mosaic(image_list, flows):
    assert len(image_list) > 0, "List is empty."

    _, hm_h, hm_w = heatmap_for_image(image_list[0]).shape
    img_w, img_h = Image.open(image_list[0]).size

    flows = flows * (hm_w / img_w, hm_h / img_h)
    acc_flows = np.cumsum(flows, axis=0)
    acc_flows = acc_flows.round().astype(int)
    
    min_x, min_y = acc_flows.min(axis=0)
    max_x, max_y = acc_flows.max(axis=0) + (hm_w, hm_h) - 1

    acc_flows -= (min_x, min_y)

    mos_h, mos_w = (max_y - min_y + 1, max_x - min_x + 1)
    hms_mosaic = np.zeros((2, mos_h, mos_w), dtype=float)
    pms_mosaic = np.zeros((1, mos_h, mos_w), dtype=float)
    emb_mosaic = np.zeros((2, mos_h, mos_w), dtype=float)
    off_mosaic = np.zeros((2, mos_h, mos_w), dtype=float)
    mask = np.full((mos_h, mos_w), np.finfo(float).tiny)
    annotations = []

    for image, (tx, ty) in tzip(image_list, acc_flows):
        heatmaps, part_heatmaps, offsets, embeddings = mat_data_for_image(image)
        hms_mosaic[:, ty:ty+hm_h, tx:tx+hm_w] += heatmaps
        pms_mosaic[:, ty:ty+hm_h, tx:tx+hm_w] += part_heatmaps
        emb_mosaic[:, ty:ty+hm_h, tx:tx+hm_w] += embeddings
        off_mosaic[:, ty:ty+hm_h, tx:tx+hm_w] += offsets
        mask[ty:ty+hm_h, tx:tx+hm_w] += 1.0

        if (locations := annotation_for_image(image)) is not None:
            locations = locations * (hm_w / img_w, hm_h / img_h) + (tx, ty)
            annotations.extend(locations)

    return hms_mosaic / mask, pms_mosaic / mask, emb_mosaic / mask, off_mosaic / mask, np.array(annotations)

def create_panorama(image_list, flows, image_size=(1024, 1024)):
    assert len(image_list) > 0, "List is empty."

    img_w, img_h = Image.open(image_list[0]).size
    hm_w, hm_h = image_size

    flows = flows * (hm_w / img_w, hm_h / img_h)
    flows = flows.round().astype(int)
    acc_flows = np.cumsum(flows, axis=0)
    acc_flows = acc_flows.astype(int)

    min_x, min_y = acc_flows.min(axis=0)
    max_x, max_y = acc_flows.max(axis=0) + (hm_w, hm_h) - 1

    acc_flows -= (min_x, min_y)

    shape = (max_y - min_y + 1, max_x - min_x + 1, 3)
    panorama = np.zeros(shape, dtype=np.uint8)

    first_img = read_image(image_list[0], reshape_size=(hm_w, hm_h))[:, :int(hm_h/2)]
    tx, ty = acc_flows[0]
    panorama[ty:ty+hm_h, tx:tx+int(hm_w/2)] = first_img

    last_img = read_image(image_list[-1], reshape_size=(hm_w, hm_h))[:, int(hm_h/2):]
    tx, ty = acc_flows[-1]
    panorama[ty:ty+hm_h, tx+int(hm_w/2):tx+hm_w] = last_img

    for image, (tx, ty), (dx, _) in tzip(image_list, acc_flows, flows):
        img = read_image(image, reshape_size=image_size)[:, hm_w-dx-int(hm_w/2):hm_w-int(hm_w/2)]
        panorama[ty:ty+hm_h, tx+hm_w-dx-int(hm_w/2):tx+hm_w-int(hm_w/2)] = img

    return panorama

# heatmap: (B, H, W)
def nms(heatmap, kernel_size=5):
    padding = kernel_size // 2
    max_values = max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=padding)
    return (heatmap == max_values) * heatmap  # (B, H, W)

# feat: (B, J, C), ind: (B, N)
def gather(feat, ind):
    ind = ind.unsqueeze(-1).expand(-1, -1, feat.size(2))  #  (B, N, C)
    feat = feat.gather(1, ind) # (B, N, C)
    return feat  # (B, N, C)

# feat: (B, C, H, W), ind: (B, N)
def transpose_and_gather(feat, ind):
    ind = ind.unsqueeze(1).expand(-1, feat.size(1), -1)  # (B, C, N)
    feat = feat.view(feat.size(0), feat.size(1), -1)  # (B, C, J = H * W)
    feat = feat.gather(2, ind)  # (B, C, N)
    feat = feat.permute(0, 2, 1)  # (B, N, C)
    return feat  # (B, N, C)

# scores: (B, C, H, W)
def topk(scores, k=100):
    (batch, cat, _, width) = scores.size()

    # (B, C, K)
    (topk_scores, topk_inds) = torch.topk(scores.view(batch, cat, -1), k)

    # (B, C, K)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # (B, K)
    (topk_score, topk_ind) = torch.topk(topk_scores.view(batch, -1), k)
    topk_clses = (topk_ind // k)

    # (B, K)
    topk_inds = gather(topk_inds.view(batch, -1, 1), topk_ind).squeeze(-1)
    topk_ys = gather(topk_ys.view(batch, -1, 1), topk_ind).squeeze(-1)
    topk_xs = gather(topk_xs.view(batch, -1, 1), topk_ind).squeeze(-1)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs  # (B, K)

def inference(mosaic, kernel_size=5, conf_threshold=25/100, k=200):
    heatmap = torch.from_numpy(mosaic)[None, ...]  # (1, C, H, W)
    heatmap = nms(heatmap, kernel_size=kernel_size)
    scores, _, _, ys, xs = topk(heatmap, k=k)  # (1, K)
    output = torch.stack([xs, ys, scores], dim=1).squeeze(0)  # (3, K)

    for idx, (_, _, score) in enumerate(output.t()):
        if score < conf_threshold:
            return output[:, :idx].t()  # (K, 3)

if __name__ == "__main__":
    folder = Path("/media/deepwater/DATA/Shared/Louis/datasets/tache_detection/haricot")
    image_file = folder / "image_list.txt"
    optlow_file = folder / "optical_flow.txt"
    
    images = [folder / path.name for path in read_image_file(image_file)]
    flows = read_optical_flow(optlow_file)
    
    mosaic = np.load("save/mosaic.npy")
    part_mosaic = np.load("save/part_mosaic.npy")
    emb_mosaic = np.load("save/emb_mosaic.npy")
    off_mosaic = np.load("save/off_mosaic.npy")
    annotations = np.load("save/annotations.npy")
    panorama = Image.open("save/panorama.jpg")

    mosaic, part_mosaic, emb_mosaic, off_mosaic, annotations = create_mosaic(images, flows)
    # panorama = create_panorama(images, flows, (1024, 1024))
    result = inference(mosaic, conf_threshold=25/100)
    mosaic_img = Image.fromarray(mosaic.sum(axis=0) * 255)
    part_mosaic_img = Image.fromarray(part_mosaic.sum(axis=0) * 255)
    (m_w, m_h) = mosaic_img.size
    mosaic_img = mosaic_img.resize((m_w*8, m_h*8))
    part_mosaic_img = part_mosaic_img.resize((m_w*8, m_h*8))
    emb_norm = np.sqrt(emb_mosaic[1, ...]**2 + emb_mosaic[0, ...]**2)
    off_norm = np.sqrt(off_mosaic[1, ...]**2 + off_mosaic[0, ...]**2)

    # np.save("save/mosaic.npy", mosaic)
    # np.save("save/part_mosaic.npy", part_mosaic)
    # np.save("save/emb_mosaic.npy", emb_mosaic)
    # np.save("save/off_mosaic.npy", off_mosaic)
    # np.save("save/annotations.npy", annotations)
    # Image.fromarray(panorama).save("save/panorama.jpg")

    fig, axes = plt.subplots(5, 1, gridspec_kw={"left": .05, "right": .95})
    axes[0].imshow(mosaic_img)
    axes[1].imshow(panorama)
    axes[1].scatter(annotations[:, 0]*8, annotations[:, 1]*8, c="red", label="Ground truths")
    axes[1].scatter(result[:, 0]*8, result[:, 1]*8, c="blue", label="Predictions")
    axes[2].imshow(emb_norm)
    axes[3].imshow(off_norm)
    axes[4].imshow(part_mosaic_img)
    plt.legend()
    plt.show()