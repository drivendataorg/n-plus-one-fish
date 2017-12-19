from collections import namedtuple
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
import sklearn
import sklearn.model_selection
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, Activation, BatchNormalization, UpSampling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.imagenet_utils import preprocess_input as preprocess_input_imagenet
import img_augmentation
import utils

NUM_CLASSES = 1
INPUT_SHAPE = (300, 300, 3)
IMG_WITH = 300
IMG_HEIGHT = 300


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
        pad_rows = int(down.get_shape()[1] - up.get_shape()[1])
        pad_cols = int(down.get_shape()[2] - up.get_shape()[2])
        print(down.get_shape(), up.get_shape())
        print('pad values:', pad_rows, pad_cols)

        if max(pad_cols, pad_rows) > 0:
            up = ZeroPadding2D(padding=((0, pad_rows), (0, pad_cols)))(up)

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
    unet = add_levels(input_tensor=inputs, sizes=[32, 64, 128, 256, 512])
    x = Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid', padding='same')(unet)
    # x = Reshape((INPUT_SHAPE[0] * INPUT_SHAPE[1], NUM_CLASSES))(x)
    # x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def preprocess_input(x):
    return preprocess_input_imagenet(np.expand_dims(x, axis=0))[0]


def unprocess_input(x):
    return utils.preprocessed_input_to_img_resnet(x)


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
                 vflip=False,
                 blurred_by_downscaling=1):
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
        self.blurred_by_downscaling = blurred_by_downscaling

    def __lt__(self, other):
        return True

    def __str__(self):
        dc = copy(self.__dict__)
        del dc['img']
        return str(dc)


Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])


class Dataset:
    def __init__(self, update_cache=False):
        mask_file_names = open('../output/fish_masks_train.txt', 'r').read().split('\n')
        self.mask_file_names = [f for f in mask_file_names if f.endswith('png')]

        self.images, self.masks = self.load_images()

        print('images', self.images.shape, self.images.dtype)
        print('masks', self.masks.shape, self.masks.dtype)

        all_idx = list(range(self.images.shape[0]))
        self.train_idx, self.test_idx = sklearn.model_selection.train_test_split(
            all_idx, test_size=100, random_state=42)

    def load_images(self):
        def load_image(mask_fn):
            img_fn = mask_fn.replace('.png', '.jpg')
            img_fn = img_fn.replace('masks/', '')

            img_data = scipy.misc.imread(os.path.join('../output', img_fn))
            return img_data

        def load_mask(mask_fn):
            mask_data = scipy.misc.imread(os.path.join('../output', mask_fn), mode='L')
            return mask_data

        pool = ThreadPool(processes=8)
        images = pool.map(load_image, self.mask_file_names)
        images = np.array(images)
        masks = pool.map(load_mask, self.mask_file_names)
        masks = np.array(masks)
        return images, masks

    def prepare_x(self, cfg: SampleCfg):
        img = self.images[cfg.img_idx]
        crop = utils.get_image_crop(full_rgb=img, rect=Rect(0, 0, IMG_WITH, IMG_HEIGHT),
                                    scale_rect_x=cfg.scale_rect_x, scale_rect_y=cfg.scale_rect_y,
                                    shift_x_ratio=cfg.shift_x_ratio, shift_y_ratio=cfg.shift_y_ratio,
                                    angle=cfg.angle, out_size=IMG_WITH)

        crop = crop.astype('float32')
        if cfg.saturation != 0.5:
            crop = img_augmentation.saturation(crop, variance=0.2, r=cfg.saturation)

        if cfg.contrast != 0.5:
            crop = img_augmentation.contrast(crop, variance=0.25, r=cfg.contrast)

        if cfg.brightness != 0.5:
            crop = img_augmentation.brightness(crop, variance=0.3, r=cfg.brightness)

        if cfg.hflip:
            crop = img_augmentation.horizontal_flip(crop)

        if cfg.vflip:
            crop = img_augmentation.vertical_flip(crop)

        if cfg.blurred_by_downscaling != 1:
            crop = img_augmentation.blurred_by_downscaling(crop, 1.0 / cfg.blurred_by_downscaling)

        return preprocess_input(crop * 255.0)

    def prepare_y(self, cfg: SampleCfg):
        img = self.masks[cfg.img_idx].astype(np.float32) / 255.0

        crop = utils.get_image_crop(full_rgb=img, rect=Rect(0, 0, IMG_WITH, IMG_HEIGHT),
                                    scale_rect_x=cfg.scale_rect_x, scale_rect_y=cfg.scale_rect_y,
                                    shift_x_ratio=cfg.shift_x_ratio, shift_y_ratio=cfg.shift_y_ratio,
                                    angle=cfg.angle, out_size=IMG_WITH, order=1)

        crop = crop.astype('float32')

        if cfg.hflip:
            crop = img_augmentation.horizontal_flip(crop)

        if cfg.vflip:
            crop = img_augmentation.vertical_flip(crop)

        return np.expand_dims(crop, axis=3)

    def generate(self, batch_size):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        from utils import rand_or_05

        while True:
            img_idx = random.choice(self.train_idx)
            cfg = SampleCfg(
                img_idx=img_idx,
                saturation=rand_or_05(),
                contrast=rand_or_05(),
                brightness=rand_or_05(),
                # color_shift=rand_or_05(),
                shift_x_ratio=random.uniform(-0.2, 0.2),
                shift_y_ratio=random.uniform(-0.2, 0.2),
                angle=random.uniform(-20.0, 20.0),
                hflip=random.choice([True, False]),
                vflip=random.choice([True, False]),
                blurred_by_downscaling=np.random.choice([1, 1, 1, 1, 1, 1, 2, 2.5, 3, 4, 6, 8])
            )

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
    dataset = Dataset(update_cache=True)
    for batch_x, batch_y in dataset.generate(batch_size=16):
        print(batch_x.shape, batch_y.shape)

        plt.imshow(unprocess_input(batch_x[0]))
        plt.imshow(batch_y[0, :, :, 0], alpha=0.2)
        plt.show()


def train_unet(continue_from_epoch=-1, weights='', batch_size=8):
    dataset = Dataset()

    model = model_unet(INPUT_SHAPE)
    model.summary()

    model_name = 'model_fish_unet2'
    checkpoints_dir = '../output/checkpoints/fish_mask_unet/' + model_name
    tensorboard_dir = '../output/tensorboard/fish_mask_unet/' + model_name
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
            return 5e-6
        if epoch < 80:
            return 2e-5
        return 1e-5

    lr_sched = LearningRateScheduler(schedule=cheduler)

    nb_epoch = 400
    validation_batch_size = 8
    model.fit_generator(dataset.generate(batch_size=batch_size),
                        steps_per_epoch=200,
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

    for batch_x, batch_y in dataset.generate_validation(batch_size=batch_size):
        print(batch_x.shape, batch_y.shape)
        with utils.timeit_context('predict 16 images'):
            prediction = model.predict_on_batch(batch_x)

        for i in range(batch_size):
            # plt.imshow(unprocess_input(batch_x[i]))
            # plt.imshow(prediction[i, :, :, 0], alpha=0.75)
            img = batch_x[i].astype(np.float32)
            mask = prediction[i, :, :, 0]

            utils.print_stats('img', img)
            utils.print_stats('mask', mask)

            img[:, :, 0] *= mask
            img[:, :, 1] *= mask
            img[:, :, 2] *= mask
            img = unprocess_input(img)
            plt.imshow(img)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ruler masks')
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--continue_from_epoch', type=int, default=-1)

    args = parser.parse_args()

    action = args.action

    if action == 'check_dataset':
        check_dataset()
    if action == 'train':
        train_unet(continue_from_epoch=args.continue_from_epoch, weights=args.weights)
    elif action == 'check':
        check_unet(weights=args.weights)
