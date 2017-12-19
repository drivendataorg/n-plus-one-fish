import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from scipy.misc import face
# from keras.preprocessing import image
# import PIL.Image
from scipy.misc import imresize

def grayscale(rgb):
    return rgb.dot([0.299, 0.587, 0.114])


def saturation(rgb, variance=0.5, r=None):
    if r is None:
        r = np.random.random()
    gs = grayscale(rgb)
    alpha = 2 * r * variance
    alpha += 1 - variance
    rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
    return np.clip(rgb, 0, 1.0)


def brightness(rgb, variance=0.5, r=None):
    if r is None:
        r = np.random.random()

    alpha = 2 * r * variance
    alpha += 1 - variance
    # lab = color.rgb2lab(rgb)
    # lab[:, :, 0] = np.clip(0.0, lab[:, :, 0]*alpha, 1.0)
    # rgb = color.lab2rgb(lab)
    rgb = rgb * alpha
    return np.clip(rgb, 0, 1.0)


def contrast(rgb, variance=0.5, r=None):
    if r is None:
        r = np.random.random()

    gs = grayscale(rgb).mean() * np.ones_like(rgb)
    alpha = 2 * r * variance
    alpha += 1 - variance
    rgb = rgb * alpha + (1 - alpha) * gs
    return np.clip(rgb, 0, 1.0)


def lighting(img, variance=0.5, r=None):
    if r is None:
        r = np.random.randn(3)

    orig = img.copy
    cov = np.cov(img.reshape(-1, 3) / 1.0, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    noise = r * variance
    noise = eigvec.dot(eigval * noise) * 1.0
    img += noise
    try:
        img += noise
        return np.clip(img, 0, 1.0)
    except TypeError:
        return orig


def horizontal_flip(img):
    return img[:, ::-1]


def vertical_flip(img):
    return img[::-1]


def blurred_by_downscaling(img, ratio):
    # ratio = np.random.choice([1, 1, 1, 2, 2.5, 3])
    resampling = np.random.choice(['nearest', 'lanczos', 'bilinear', 'bicubic'])

    if ratio == 1:
        return img

    w = img.shape[1]
    h = img.shape[0]
    small = imresize(img, ratio, interp=resampling)
    large = imresize(small, size=(h, w), interp='bilinear').astype(np.float32)/255.0
    return large

if __name__ == '__main__':
    np.random.seed(1337)

    img = face() / 256.0

    import utils
    utils.print_stats('img', img)
    utils.print_stats('img_aug', blurred_by_downscaling(img, 0.3))
    utils.print_stats('img_aug', blurred_by_downscaling(img, 0.3))
    utils.print_stats('img_aug', blurred_by_downscaling(img, 0.3))
    utils.print_stats('img_aug', blurred_by_downscaling(img, 0.3))

    plt.imshow(img)
    plt.figure()
    # plt.imshow(grayscale(img), cmap='gray')
    plt.imshow(blurred_by_downscaling(img, 0.3))
    plt.figure()
    plt.imshow(blurred_by_downscaling(img, 0.3))
    plt.figure()
    plt.imshow(blurred_by_downscaling(img, 0.3))
    plt.figure()
    plt.imshow(blurred_by_downscaling(img, 0.1))
    plt.figure()
    plt.imshow(blurred_by_downscaling(img, 0.1))
    plt.show()

    fig, axarr = plt.subplots(4, 3)
    for i, f in enumerate([saturation, brightness, contrast, lighting]):
        axarr[i, 0].imshow(f(img, variance=0.8, r=0.0))
        axarr[i, 1].imshow(f(img, variance=0.8, r=0.5))
        axarr[i, 2].imshow(f(img, variance=0.8, r=1.0))
    plt.show()

    fig, axarr = plt.subplots(4, 3)
    for i, f in enumerate([saturation, brightness, contrast, lighting]):
        axarr[i, 0].imshow(f(img, variance=0.5))
        axarr[i, 1].imshow(f(img, variance=0.5))
        axarr[i, 2].imshow(f(img, variance=0.5))
    plt.show()
