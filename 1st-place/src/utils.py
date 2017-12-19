import math
import pickle
import time
import random
from contextlib import contextmanager
import concurrent.futures
from queue import Queue

import skimage.io
import skimage.transform
from skimage.transform import SimilarityTransform, AffineTransform
import numpy as np
import matplotlib.pyplot as plt


def crop_edge(img, x, y, w, h, mode='edge'):
    img_w = img.shape[1]
    img_h = img.shape[0]

    if x >= 0 and y >= 0 and x + w <= img_w and y + h < img_h:
        return img[int(y):int(y + h), int(x):int(x + w)].astype('float32') / 255.0

    tform = SimilarityTransform(translation=(x, y))
    return skimage.transform.warp(img, tform, mode=mode, output_shape=(h, w))

def preprocessed_input_to_img_resnet(x):
    # Zero-center by mean pixel
    x = x.copy()
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 'BGR' -> RGB
    img = x.copy()
    img[:, :, 0] = x[:, :, 2]
    img[:, :, 1] = x[:, :, 1]
    img[:, :, 2] = x[:, :, 0]
    return img / 255.0


def bbox_for_line(p0, p1, aspect_ratio=0.5):
    """
        Calculate bounding rect around box with line in the center
        :param p0:
        :param p1:
        :param aspect_ratio: rect aspect ratio, 0.5 - width == 0.5 line norm
        :return:
    """
    # p0 = np.array([x0, y0])
    # p1 = np.array([x1, y1])
    p = p1-p0
    p90 = np.array([p[1], -p[0]])*aspect_ratio*0.5  # vector perpendicular to p0-p1 with 0.5 aspect ratio norm

    points = np.row_stack([p0+p90, p0-p90, p1+p90, p1-p90])
    return np.min(points, axis=0), np.max(points, axis=0)

def check_bbox_for_line():
    for p0, p1 in [
        [[0, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 0], [1, 1]],
        [[1, 1], [0, 0]],
        [[1, 2], [3, 4]],
                  ]:
        box_p0, box_p1 = bbox_for_line(np.array(p0), np.array(p1))

        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], c='b')
        w, h = box_p1-box_p0
        plt.gca().add_patch(plt.Rectangle(box_p0, w, h, fill=False, edgecolor='g', linewidth=2))
        plt.show()
# check_bbox_for_line()

@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l) // n * n + n - 1, n):
        if len(l[i:i + n]):
            yield l[i:i + n]


def load_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(data, file_name):
    pickle.dump(data, open(file_name, 'wb'))


def lock_layers_until(model, first_trainable_layer, verbose=False):
    found_first_layer = False
    for layer in model.layers:
        if layer.name == first_trainable_layer:
            found_first_layer = True

        if verbose and found_first_layer and not layer.trainable:
            print('Make layer trainable:', layer.name)
            layer.trainable = True

        layer.trainable = found_first_layer


def get_image_crop(full_rgb, rect, scale_rect_x=1.0, scale_rect_y=1.0,
                   shift_x_ratio=0.0, shift_y_ratio=0.0,
                   angle=0.0, out_size=299, order=3):
    center_x = rect.x + rect.w / 2
    center_y = rect.y + rect.h / 2
    size = int(max(rect.w, rect.h))
    size_x = size * scale_rect_x
    size_y = size * scale_rect_y

    center_x += size * shift_x_ratio
    center_y += size * shift_y_ratio

    scale_x = out_size / size_x
    scale_y = out_size / size_y

    out_center = out_size / 2

    tform = AffineTransform(translation=(center_x, center_y))
    tform = AffineTransform(rotation=angle * math.pi / 180) + tform
    tform = AffineTransform(scale=(1 / scale_x, 1 / scale_y)) + tform
    tform = AffineTransform(translation=(-out_center, -out_center)) + tform
    return skimage.transform.warp(full_rgb, tform, mode='edge', order=order, output_shape=(out_size, out_size))


def crop_zero_pad(img, x, y, w, h):
    img_w = img.shape[1]
    img_h = img.shape[0]

    if x >= 0 and y >= 0 and x + w <= img_w and y + h < img_h:
        return img[int(y):int(y + h), int(x):int(x + w)]
    else:
        res = np.zeros((h, w)+img.shape[2:])
        x_min = int(max(x, 0))
        y_min = int(max(y, 0))
        x_max = int(min(x + w, img_w))
        y_max = int(min(y + h, img_h))
        res[y_min - y:y_max-y, x_min - x:x_max-x] = img[y_min:y_max, x_min:x_max]
        return res


def overlapped_crops_shape(img, crop_w, crop_h, overlap):
    img_h, img_w = img.shape[:2]
    n_h = int(np.ceil((img_h + overlap/2 - 1) / (crop_h - overlap)))
    n_w = int(np.ceil((img_w + overlap/2 - 1) / (crop_w - overlap)))
    return [n_h, n_w]


def generate_overlapped_crops_with_positions(img, crop_w, crop_h, overlap):
    n_h, n_w = overlapped_crops_shape(img, crop_w, crop_h, overlap)

    res = np.zeros((n_w*n_h, crop_h, crop_w, ) + img.shape[2:])
    positions = []

    for i_h in range(n_h):
        for i_w in range(n_w):
            x = -overlap // 2 + i_w * (crop_w - overlap)
            y = -overlap // 2 + i_h * (crop_h - overlap)
            res[i_h * n_w + i_w] = crop_zero_pad(img, x, y, crop_w, crop_h)
            positions.append((x, y, crop_w, crop_h))

    return res, positions


def generate_overlapped_crops(img, crop_w, crop_h, overlap):
    return generate_overlapped_crops_with_positions(img, crop_w, crop_h, overlap)[0]


def rand_or_05():
    if random.random() > 0.5:
        return random.random()
    return 0.5


def rand_scale_log_normal(mean_scale, one_sigma_at_scale):
    """
    Generate a distribution of value at log  scale around mean_scale

    :param mean_scale:  
    :param one_sigma_at_scale: 67% of values between  mean_scale/one_sigma_at_scale .. mean_scale*one_sigma_at_scale
    :return: 
    """

    log_sigma = math.log(one_sigma_at_scale)
    return mean_scale*math.exp(random.normalvariate(0.0, log_sigma))


def print_stats(title, array):
    print('{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}'.format(
        title,
        array.shape,
        array.dtype,
        np.min(array),
        np.max(array),
        np.mean(array),
        np.median(array)
    ))


class ImageCache:
    """
        Helper class to keep the queue of a number of images to be used for training,
        to avoid cost of loading new image for each training / verification sample.
    """

    def __init__(self, id_set, loader, is_sequential, images_to_keep=8, replace_period=4):
        self.is_sequential = is_sequential
        self.loader = loader
        self.id_set = id_set

        self.images_to_keep = images_to_keep
        self.replace_period = replace_period
        self.loaded_items = []
        self.loaded_sequential_img_id = 0
        self.step = -1

    def load(self):
        self.step += 1

        if self.is_sequential:
            if self.step % self.replace_period == 0:
                self.loaded_items.clear()
                item_to_load = self.id_set[self.loaded_sequential_img_id % len(self.id_set)]
                self.loaded_sequential_img_id += 1
                self.loaded_items.append((item_to_load, self.loader(item_to_load)))
            return self.loaded_items[0]
        else:
            # keep 8 images loaded and replace one each 8 steps
            if len(self.loaded_items) < self.images_to_keep or self.step % self.replace_period == 0:
                if len(self.loaded_items) == self.images_to_keep:
                    self.loaded_items.pop(0)

                item_to_load = random.choice(self.id_set)
                self.loaded_items.append((item_to_load, self.loader(item_to_load)))
            return random.choice(self.loaded_items)


def parallel_generator(orig_gen, executor):
    queue = Queue(maxsize=8)

    def bg_task():
        for i in orig_gen:
            # print('bg_task', i)
            queue.put(i)
        # print('bg_task', None)
        queue.put(None)

    task = executor.submit(bg_task)
    while True:
        value = queue.get()
        if value is not None:
            yield value
            queue.task_done()
        else:
            queue.task_done()
            break
    task.result()


def test_parallel_generator():
    def task(i):
        time.sleep(0.1)
        print('task', i)
        return i

    def orig_gen(n):
        for i in range(n):
            yield task(i)

    res_orig = []
    with timeit_context('orig gen'):
        for i in orig_gen(5):
            time.sleep(0.1)
            res_orig.append(i)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    res_par = []
    with timeit_context('parallel gen'):
        for i in parallel_generator(orig_gen(5), executor):
            time.sleep(0.1)
            res_par.append(i)

    assert res_orig == res_par


if __name__ == '__main__':
    pass
    test_parallel_generator()
    # test_chunks()
    #
    # img = skimage.io.imread('../train/ALB/img_00003.jpg')
    # print(img.shape)
    #
    # with timeit_context('Generate crops'):
    #     crop_edge(img, 10, 10, 400, 400)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.subplot(222)
    # plt.imshow(crop_edge(img, 1280-200, 720-200, 400, 400, mode='edge'))
    # plt.subplot(223)
    # plt.imshow(crop_edge(img, 1280 - 200, 720 - 200, 400, 400, mode='wrap'))
    # plt.subplot(224)
    # plt.imshow(crop_edge(img, 1280 - 200, 720 - 200, 400, 400, mode='reflect'))
    #
    # plt.show()
