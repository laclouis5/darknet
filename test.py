from skimage import io, data, filters, feature, color, exposure, morphology
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv

def egi_mask(image, thresh=1.15):
    image_np = np.array(image).astype(float)

    image_np = 2*image_np[:, :, 1] / (image_np[:, :, 0] + image_np[:, :, 2] + 0.001)
    image_gf = filters.gaussian(image_np, sigma=1, mode='reflect')

    image_bin = image_gf > 1.15

    image_morph = morphology.binary_erosion(image_bin, morphology.disk(3))
    image_morph = morphology.binary_dilation(image_morph, morphology.disk(3))

    image_out = morphology.remove_small_objects(image_morph, 400)
    image_out = morphology.remove_small_holes(image_out, 800)

    return image_out


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
        Gx  = cv.Sobel(img, cv.CV_32F, 1, 0, 3, borderType=border_type)
        Gy  = cv.Sobel(img, cv.CV_32F, 0, 1, 3, borderType=border_type)

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
