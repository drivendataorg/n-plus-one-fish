import numpy as np
import pandas as pd
import argparse
import math
import os
import os.path
import pickle
import random
from copy import copy
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import scipy.misc
import skimage
import skimage.io
import skimage.transform
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Activation, BatchNormalization, UpSampling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from skimage.transform import SimilarityTransform, AffineTransform

import utils
import dataset
from dataset import IMAGES_DIR, MASKS_DIR, AVG_MASKS_DIR
from dataset import IMAGES_DIR_TEST, MASKS_DIR_TEST, AVG_MASKS_DIR_TEST
import fish_detection

NUM_CLASSES = 1
INPUT_SHAPE = (360, 640, 3)
IMG_WITH = 640
IMG_HEIGHT = 360


def model_unet(input_shape=INPUT_SHAPE):
    def add_levels(input_tensor, sizes):
        filters = sizes[0]

        down = Conv2D(filters, (3, 3), padding='same')(input_tensor)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)
        down = Conv2D(filters, (3, 3), padding='same')(down)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)

        if len(sizes) == 1:
            return down

        down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down)

        subnet = add_levels(down_pool, sizes[1:])

        up = UpSampling2D((2, 2))(subnet)
        up = concatenate([down, up], axis=3)
        up = Conv2D(filters, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        up = Conv2D(filters, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        up = Conv2D(filters, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        return up

    inputs = Input(shape=input_shape)
    unet = add_levels(input_tensor=inputs, sizes=[32, 64, 128, 256])
    x = Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', padding='same')(unet)
    # x = Reshape((INPUT_SHAPE[0] * INPUT_SHAPE[1], NUM_CLASSES))(x)
    # x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def preprocess_input(x):
    return x.astype(np.float32) / 128.0 - 1.0


def unprocess_input(x):
    return ((x + 1.0) * 128.0).astype(np.uint8)


class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self, img_idx,
                 scale_rect_x=1.0,
                 scale_rect_y=1.0,
                 shift_x_ratio=0.0,
                 shift_y_ratio=0.0,
                 angle=0.0,
                 saturation=0.5, contrast=0.5, brightness=0.5,  # 0.5  - no changes, range 0..1
                 hflip=False,
                 vflip=False):
        self.angle = angle
        self.shift_y_ratio = shift_y_ratio
        self.shift_x_ratio = shift_x_ratio
        self.scale_rect_y = scale_rect_y
        self.scale_rect_x = scale_rect_x
        self.img_idx = img_idx
        self.vflip = vflip
        self.hflip = hflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blurred_by_downscaling = None

    def __lt__(self, other):
        return True

    def __str__(self):
        dc = copy(self.__dict__)
        del dc['img']
        return str(dc)


class Dataset:
    def __init__(self, update_cache=False):
        mask_file_names = open('../input/masks.txt', 'r').read().split('\n')
        self.mask_file_names = [f for f in mask_file_names if f.endswith('png')]

        if update_cache:
            self.images, self.masks = self.load_images()
            np.save('../output/masks_images.npy', self.images)
            np.save('../output/masks.npy', self.masks)
        else:
            self.images = np.load('../output/masks_images.npy')
            self.masks = np.load('../output/masks.npy')

        print('images', self.images.shape, self.images.dtype)
        print('masks', self.masks.shape, self.masks.dtype)

        all_idx = list(range(self.images.shape[0]))
        self.train_idx = all_idx[:-96]
        self.test_idx = all_idx[-96:]

    def load_images(self):
        def load_image(mask_fn):
            img_fn = mask_fn.replace('.png', '.jpg')
            img_fn = img_fn.replace('masks/', '')

            img_data = scipy.misc.imread(os.path.join(IMAGES_DIR, img_fn))
            img_data = scipy.misc.imresize(img_data, 0.5, interp='cubic')
            return img_data

        def load_mask(mask_fn):
            mask_data = scipy.misc.imread(os.path.join(IMAGES_DIR, mask_fn), mode='L')
            mask_data = scipy.misc.imresize(mask_data, 0.5, interp='bilinear', mode='L')
            return mask_data

        pool = ThreadPool(processes=8)
        images = pool.map(load_image, self.mask_file_names)
        images = np.array(images)
        masks = pool.map(load_mask, self.mask_file_names)
        masks = np.array(masks)
        return images, masks

    def prepare_x(self, cfg: SampleCfg):
        img = preprocess_input(self.images[cfg.img_idx])
        return img

    def prepare_y(self, cfg: SampleCfg):
        return np.expand_dims(self.masks[cfg.img_idx].astype(np.float32) / 256.0, axis=3)

    def generate(self, batch_size):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]


        while True:
            img_idx = random.choice(self.train_idx)
            cfg = SampleCfg(img_idx=img_idx)
            samples_to_process.append(cfg)

            if len(samples_to_process) == batch_size:
                X_batch = np.array(pool.map(self.prepare_x, samples_to_process))
                y_batch = np.array(pool.map(self.prepare_y, samples_to_process))
                samples_to_process = []
                yield X_batch, y_batch

    def generate_validation(self, batch_size):
        pool = ThreadPool(processes=8)
        while True:
            for idxs in utils.chunks(self.test_idx, batch_size):
                samples_to_process = [SampleCfg(idx) for idx in idxs]
                X_batch = np.array(pool.map(self.prepare_x, samples_to_process))
                y_batch = np.array(pool.map(self.prepare_y, samples_to_process))
                yield X_batch, y_batch


def check_dataset():
    dataset = Dataset(update_cache=False)
    for batch_x, batch_y in dataset.generate(batch_size=16):
        print(batch_x.shape, batch_y.shape)

        plt.imshow(unprocess_input(batch_x[0]))
        plt.imshow(batch_y[0], alpha=0.2)
        plt.show()


def train_unet(continue_from_epoch=-1, weights='', batch_size=8):
    dataset = Dataset()

    model = model_unet(INPUT_SHAPE)
    model.summary()

    model_name = 'model_unet1'
    checkpoints_dir = '../output/checkpoints/mask_unet/' + model_name
    tensorboard_dir = '../output/tensorboard/mask_unet/' + model_name
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    if len(weights) > 0:
        model.load_weights(weights)

    checkpoint_best = ModelCheckpoint(checkpoints_dir + "/checkpoint-best-{epoch:03d}-{val_loss:.4f}.hdf5",
                                      verbose=1,
                                      save_weights_only=False,
                                      save_best_only=True)
    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5",
                                            verbose=1,
                                            save_weights_only=True,
                                            period=8)
    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=False, write_images=True)

    def cheduler(epoch):
        if epoch < 10:
            return 1e-3
        if epoch < 25:
            return 2e-4
        if epoch < 60:
            return 1e-4
        if epoch < 80:
            return 5e-5
        return 2e-5

    lr_sched = LearningRateScheduler(schedule=cheduler)

    nb_epoch = 400
    validation_batch_size = 4
    model.fit_generator(dataset.generate(batch_size=batch_size),
                        steps_per_epoch=60,
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=[checkpoint_periodical, checkpoint_best, tensorboard, lr_sched],
                        validation_data=dataset.generate_validation(batch_size=validation_batch_size),
                        validation_steps=len(dataset.test_idx)//validation_batch_size,
                        initial_epoch=continue_from_epoch + 1)


def check_unet(weights):
    dataset = Dataset()
    model = model_unet(INPUT_SHAPE)
    model.load_weights(weights)
    batch_size = 16

    for batch_x, batch_y in dataset.generate(batch_size=batch_size):
        print(batch_x.shape, batch_y.shape)
        with utils.timeit_context('predict 16 images'):
            prediction = model.predict_on_batch(batch_x)

        for i in range(batch_size):
            plt.imshow(unprocess_input(batch_x[i]))
            plt.imshow(prediction[i, :, :, 0], alpha=0.75)
            plt.show()


def predict_masks(fold):
    weights = '../output/checkpoints/mask_unet/model_unet1/checkpoint-best-019-0.0089.hdf5'
    model = model_unet(INPUT_SHAPE)
    model.load_weights(weights)
    batch_size = 16

    input_samples = []
    processed_samples = 0

    dest_dir = '../output/ruler_masks'

    for dir_name in os.listdir(IMAGES_DIR):
        clip_dir = os.path.join(IMAGES_DIR, dir_name)
        os.makedirs(os.path.join(dest_dir, dir_name), exist_ok=True)

        for frame_name in os.listdir(clip_dir):
            if not frame_name.endswith('.jpg'):
                continue
            input_samples.append((dir_name, frame_name))

    if fold == 1:
        input_samples = input_samples[: len(input_samples) // 2]
    elif fold == 2:
        input_samples = input_samples[len(input_samples) // 2:]

    pool = ThreadPool(processes=8)
    save_batch_size = 64
    for batch_input_samples in utils.chunks(input_samples, batch_size*save_batch_size):
        def process_sample(sample):
            img_data = scipy.misc.imread(os.path.join(IMAGES_DIR, sample[0], sample[1]))
            img_data = scipy.misc.imresize(img_data, 0.5, interp='cubic')
            return preprocess_input(img_data)

        def generate_x():
            while True:
                for samples in utils.chunks(batch_input_samples, batch_size):
                    yield np.array(pool.map(process_sample, samples))

        with utils.timeit_context('predict {} images, {}/{}, {:.1}%'.format(
                                  batch_size*save_batch_size, processed_samples, len(input_samples),
                                  100.0*processed_samples/len(input_samples))):
            predictions = model.predict_generator(generate_x(), steps=save_batch_size, verbose=1)

        for i in range(predictions.shape[0]):
            dir_name, fn = input_samples[processed_samples]
            processed_samples += 1
            fn = fn.replace('jpg', 'png')
            scipy.misc.imsave(os.path.join(dest_dir, dir_name, fn), (predictions[i, :, :, 0]*255.0).astype(np.uint8))


def predict_masks_test():
    weights = '../output/checkpoints/mask_unet/model_unet1/checkpoint-best-019-0.0089.hdf5'
    model = model_unet(INPUT_SHAPE)
    model.load_weights(weights)
    batch_size = 16

    input_samples = []
    processed_samples = 0

    dest_dir = '../output/ruler_masks_test'

    for dir_name in os.listdir(IMAGES_DIR_TEST):
        clip_dir = os.path.join(IMAGES_DIR_TEST, dir_name)
        os.makedirs(os.path.join(dest_dir, dir_name), exist_ok=True)

        for frame_name in os.listdir(clip_dir):
            if not frame_name.endswith('.jpg'):
                continue
            input_samples.append((dir_name, frame_name))

    pool = ThreadPool(processes=8)
    save_batch_size = 64
    for batch_input_samples in utils.chunks(input_samples, batch_size*save_batch_size):
        def process_sample(sample):
            img_data = scipy.misc.imread(os.path.join(IMAGES_DIR_TEST, sample[0], sample[1]))
            img_data = scipy.misc.imresize(img_data, 0.5, interp='cubic')
            return preprocess_input(img_data)

        def generate_x():
            while True:
                for samples in utils.chunks(batch_input_samples, batch_size):
                    yield np.array(pool.map(process_sample, samples))

        with utils.timeit_context('predict {} images, {}/{}, {:.1f}%'.format(
                                  batch_size*save_batch_size, processed_samples, len(input_samples),
                                  100.0*processed_samples/len(input_samples))):
            predictions = model.predict_generator(generate_x(), steps=save_batch_size, verbose=1)

        for i in range(predictions.shape[0]):
            dir_name, fn = input_samples[processed_samples]
            processed_samples += 1
            fn = fn.replace('jpg', 'png')
            scipy.misc.imsave(os.path.join(dest_dir, dir_name, fn), (predictions[i, :, :, 0]*255.0).astype(np.uint8))


def rotate(img, angle, dest_shape):
    h, w = dest_shape
    src_h, src_w = img.shape[:2]

    tform = AffineTransform(translation=(src_w / 2, src_h / 2))
    tform = AffineTransform(rotation=angle * math.pi / 180) + tform
    tform = AffineTransform(translation=(-w / 2, -h / 2)) + tform
    return skimage.transform.warp(img, tform, mode='constant', cval=0, order=1, output_shape=(h, w)), tform


def find_ruler_rect(video_id, masks_dir='../output/ruler_masks', output_dir=AVG_MASKS_DIR):
    print(video_id)
    # masks_dir = '../output/ruler_masks'
    masks = []
    clip_dir = os.path.join(masks_dir, video_id)
    for frame_name in os.listdir(clip_dir):
        if not frame_name.endswith('.png'):
            continue
        masks.append(scipy.misc.imread(os.path.join(clip_dir, frame_name)))

    all_masks = np.array(masks)
    all_masks[all_masks < 0.1] = 0.0
    print(all_masks.shape)
    avg_mask = all_masks.mean(axis=0)
    # downscale to 180x320
    avg_mask = scipy.misc.imresize(avg_mask, 0.5, interp='bilinear', mode='L')

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, video_id + '.npy'), avg_mask)

    # area to paint mask on
    w = 400

    angles = np.linspace(-90.0, 90.0, 180)
    ranges = []
    for angle in angles:
        rotated, tform = rotate(avg_mask, angle, dest_shape=(w, w))

        col = np.cumsum(np.mean(rotated, axis=1))
        col /= col[-1]
        range = np.sum(np.all([col > 0.1, col < 0.9], axis=0))
        # print(angle, range)
        ranges.append(range)

    min_angle = angles[np.argmin(ranges)]
    return min_angle


def find_ruler_angles():
    video_ids = os.listdir(MASKS_DIR)

    pool = ThreadPool(processes=8)
    angles = pool.map(find_ruler_rect, video_ids)

    df = pd.DataFrame(data={'video_id': video_ids, 'ruler_angle': angles})
    df.to_csv('../output/ruler_angles.csv', index=False)


def find_ruler_angles_test():
    video_ids = os.listdir(MASKS_DIR_TEST)

    pool = ThreadPool(processes=8)
    angles = pool.map(lambda video_id: find_ruler_rect(video_id, masks_dir='../output/ruler_masks_test', output_dir=AVG_MASKS_DIR_TEST), video_ids)

    df = pd.DataFrame(data={'video_id': video_ids, 'ruler_angle': angles})
    df.to_csv('../output/ruler_angles_test.csv', index=False)


def find_ruler_points(avg_masks_dir=AVG_MASKS_DIR, res_suffix=''):
    def find_one_ruler_points(video_id, angle):
        mask = np.load(os.path.join(avg_masks_dir, video_id + '.npy'))
        h = 400
        w = 400
        rotated, tform = rotate(mask, angle, dest_shape=(h, w))

        col = np.cumsum(np.mean(rotated, axis=0))
        col /= col[-1]
        row = np.cumsum(np.mean(rotated, axis=1))
        row /= row[-1]

        col0 = np.where(col > 0.02)[0][0]
        col1 = np.where(col < 0.98)[0][-1]
        row0 = np.where(row > 0.5)[0][0]

        col0 -= (col1 - col0) * 0.05
        col1 += (col1 - col0) * 0.05
        # print(col0, col1, row0)

        # find transform back to mask
        rotated_points = np.array([[col0, row0], [col1, row0]])
        src_points = tform(rotated_points)

        img_src_points = src_points * 4.0  # source image has been downsampled 4 times
        return img_src_points

    # clips = video_clips()
    angles = pd.read_csv('../output/ruler_angles{}.csv'.format(res_suffix))
    points = []
    for _, row in angles.iterrows():
        print(row.video_id, row.ruler_angle)
        points.append(find_one_ruler_points(row.video_id, row.ruler_angle))
    points = np.array(points)
    print(points.shape)
    angles['ruler_x0'] = points[:, 0, 0]
    angles['ruler_y0'] = points[:, 0, 1]
    angles['ruler_x1'] = points[:, 1, 0]
    angles['ruler_y1'] = points[:, 1, 1]
    angles.to_csv('../output/ruler_points{}.csv'.format(res_suffix), index=False)


def check_ruler_points():
    points = pd.read_csv('../output/ruler_points.csv')
    for _, row in points.iterrows():
        video_id = row.video_id
        img = scipy.misc.imread(os.path.join(IMAGES_DIR, video_id, "0001.jpg"))

        dst_w = 720
        dst_h = 360
        ruler_points = np.array([[row.ruler_x0, row.ruler_y0], [row.ruler_x1, row.ruler_y1]])
        img_points = np.array([[dst_w*0.1, dst_h/2], [dst_w*0.9, dst_h/2]])

        tform = SimilarityTransform()
        tform.estimate(dst=ruler_points, src=img_points)
        crop = skimage.transform.warp(img, tform, mode='edge', order=3, output_shape=(dst_h, dst_w))

        print('ruler:\n', ruler_points)
        print('img:\n', img_points)

        print('ruler from img:\n', tform(img_points))
        print('img from ruler:\n', tform.inverse(ruler_points))
        print('scale', tform.scale)

        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.plot([row.ruler_x0, row.ruler_x1], [row.ruler_y0, row.ruler_y1])

        plt.subplot(2, 1, 2)
        plt.imshow(crop)
        plt.show()


def generate_crops():
    detection_dataset = fish_detection.FishDetectionDataset()
    def decode_clip(video_id):
        frames = detection_dataset.video_clips[video_id]
        print(video_id)
        os.makedirs('{}/{}'.format(dataset.RULER_CROPS_DIR, video_id), exist_ok=True)
        dest_w = 720
        dest_h = 360
        transform = detection_dataset.transform_for_clip(video_id, dest_w, dest_h)
        for frame in frames:
            src_fn = '{}/{}/{}.jpg'.format(IMAGES_DIR, video_id, frame)
            dst_fn = '{}/{}/{}.jpg'.format(dataset.RULER_CROPS_DIR, video_id, frame)

            if os.path.isfile(dst_fn):
                continue
            img = scipy.misc.imread(src_fn)
            crop = skimage.transform.warp(img, transform, mode='edge', order=3, output_shape=(dest_h, dest_w))
            scipy.misc.imsave(dst_fn, crop)

    pool = ThreadPool(processes=8)
    pool.map(decode_clip, detection_dataset.video_clips.keys())


def generate_crops_test():
    detection_dataset = fish_detection.FishDetectionDataset(is_test=True)

    def decode_clip(video_id):
        frames = detection_dataset.video_clips[video_id]
        print(video_id)
        os.makedirs('{}/{}'.format(dataset.RULER_CROPS_DIR_TEST, video_id), exist_ok=True)
        dest_w = 720
        dest_h = 360
        transform = detection_dataset.transform_for_clip(video_id, dest_w, dest_h)
        for frame in frames:
            src_fn = '{}/{}/{}.jpg'.format(IMAGES_DIR_TEST, video_id, frame)
            dst_fn = '{}/{}/{}.jpg'.format(dataset.RULER_CROPS_DIR_TEST, video_id, frame)

            if os.path.isfile(dst_fn):
                continue
            img = scipy.misc.imread(src_fn)
            crop = skimage.transform.warp(img, transform, mode='edge', order=3, output_shape=(dest_h, dest_w))
            scipy.misc.imsave(dst_fn, crop)

    pool = ThreadPool(processes=8)
    pool.map(decode_clip, detection_dataset.video_clips.keys())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ruler masks')
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--fold', type=int, default=0)

    args = parser.parse_args()

    action = args.action

    if action == 'train':
        train_unet(continue_from_epoch=8,
                   weights='../output/checkpoints/mask_unet/model_unet1/checkpoint-best-007-0.0187.hdf5')
    elif action == 'check':
        check_unet(weights=args.weights)
    elif action == 'predict':
        predict_masks(args.fold)
    elif action == 'predict_test':
        predict_masks_test()
    elif action == 'find_ruler':
        find_ruler_rect('00WK7DR6FyPZ5u3A')
    elif action == 'find_ruler_angles':
        find_ruler_angles()
    elif action == 'find_ruler_angles_test':
        find_ruler_angles_test()
    elif action == 'find_ruler_vectors':
        find_ruler_points()
    elif action == 'find_ruler_vectors_test':
        find_ruler_points(avg_masks_dir=AVG_MASKS_DIR_TEST, res_suffix='_test')
    elif action == 'check_ruler_points':
        check_ruler_points()
    elif action == 'generate_crops':
        generate_crops()
    elif action == 'generate_crops_test':
        generate_crops_test()

