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
from keras.layers import Conv1D, MaxPooling1D, Cropping1D, UpSampling1D
from keras.layers import Input, Activation, BatchNormalization, GlobalAveragePooling1D, Flatten, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from skimage.transform import SimilarityTransform, AffineTransform

import utils
import dataset
from dataset import IMAGES_DIR, MASKS_DIR, AVG_MASKS_DIR
import fish_detection

NUM_CLASSES = 1
IMG_WITH = 720
IMG_HEIGHT = 360

NB_FEATURES = 17
NB_STEPS = 256
NB_STEPS_CROP = 32
NB_RES_STEPS = NB_STEPS - NB_STEPS_CROP * 2
INPUT_SHAPE = (NB_STEPS, NB_FEATURES)


def model_unet(input_shape=INPUT_SHAPE):
    def add_levels(input_tensor, sizes):
        filters = sizes[0]

        down = Conv1D(filters, 3, padding='same')(input_tensor)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)
        down = Conv1D(filters, 3, padding='same')(down)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)

        if len(sizes) == 1:
            return down

        down_pool = MaxPooling1D(2)(down)

        subnet = add_levels(down_pool, sizes[1:])

        up = UpSampling1D(2)(subnet)
        up = concatenate([down, up], axis=2)
        up = Conv1D(filters, 3, padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        up = Conv1D(filters, 3, padding='same')(up)
        # up = BatchNormalization()(up)
        up = Activation('relu')(up)
        return up

    inputs = Input(shape=input_shape)
    unet = add_levels(input_tensor=inputs, sizes=[16, 24, 32, 32])
    x = concatenate([unet, inputs], axis=2)
    x = Conv1D(8, 3, padding='same')(x)
    x = Conv1D(1, 1, activation='sigmoid', padding='same')(x)
    x = Cropping1D(cropping=(NB_STEPS_CROP, NB_STEPS_CROP))(x)
    x = Flatten()(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def model_unet_cumsum(input_shape=INPUT_SHAPE):
    def add_levels(input_tensor, sizes):
        filters = sizes[0]

        down = Conv1D(filters, 3, padding='same')(input_tensor)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)
        down = Conv1D(filters, 3, padding='same')(down)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)

        if len(sizes) == 1:
            return down

        down_pool = MaxPooling1D(2)(down)

        subnet = add_levels(down_pool, sizes[1:])

        up = UpSampling1D(2)(subnet)
        up = concatenate([down, up], axis=2)
        up = Conv1D(filters, 3, padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        up = Conv1D(filters, 3, padding='same')(up)
        # up = BatchNormalization()(up)
        up = Activation('relu')(up)
        return up

    inputs = Input(shape=input_shape)
    unet = add_levels(input_tensor=inputs, sizes=[16, 24, 32, 32])
    x = concatenate([unet, inputs], axis=2)
    x = Conv1D(8, 3, padding='same')(x)
    x = Conv1D(1, 1, activation='sigmoid', padding='same')(x)
    x = Cropping1D(cropping=(NB_STEPS_CROP, NB_STEPS_CROP))(x)
    x = Flatten(name='current_values')(x)

    cumsum_value = Lambda(lambda a: K.cumsum(a, axis=1), name='cumsum_values')(x)

    model = Model(inputs=inputs, outputs=[x, cumsum_value])
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'current_values': 'binary_crossentropy', 'cumsum_values': 'mse'},
                  loss_weights={'current_values': 1.0, 'cumsum_values': 0.001})

    return model


class Dataset:
    def __init__(self, fold,
                 detection_results_dir='../output/detection_results',
                 classification_results_dir='../output/classification_results',
                 load_only_video_ids=None):
        self.video_clips = dataset.video_clips()

        if load_only_video_ids is not None:
            all_video_ids = set(load_only_video_ids)
        else:
            all_video_ids = set(self.video_clips.keys())
        self.test_video_ids = set(dataset.fold_test_video_ids(fold))
        self.train_video_ids = all_video_ids.difference(self.test_video_ids)

        self.gt = pd.read_csv('../input/N1_fish_N2_fish_-_Training_set_annotations.csv')
        self.gt.dropna(axis=0, inplace=True)
        self.gt['have_frame'] = 1.0

        self.video_frames_count = {}
        self.video_data = {}
        self.video_data_gt = {}

        for video_id in sorted(all_video_ids):
            try:
                ds_detection = pd.read_csv(os.path.join(detection_results_dir, video_id + '_ssd_detection.csv'))
                ds_classification = pd.read_csv(os.path.join(classification_results_dir, video_id + '_categories.csv'))
            except FileNotFoundError:
                continue

            ds_combined = ds_classification.join(ds_detection, on='frame', how='left', rsuffix='_det')
            ds_combined.x /= IMG_WITH
            ds_combined.y /= IMG_HEIGHT
            ds_combined.w /= IMG_WITH
            ds_combined.h /= IMG_HEIGHT

            nb_frames = len(self.video_clips[video_id])
            self.video_frames_count[video_id] = nb_frames
            all_frames = pd.DataFrame({'frame': list(range(nb_frames))})
            ds_combined = all_frames.merge(ds_combined, on='frame', how='left').fillna(0.0)

            ds_combined['species__'] = 1.0 - (
                ds_combined['species_fourspot'] +
                ds_combined['species_grey sole'] +
                ds_combined['species_other'] +
                ds_combined['species_plaice'] +
                ds_combined['species_summer'] +
                ds_combined['species_windowpane'] +
                ds_combined['species_winter'])

            self.columns = ['species__', 'species_fourspot', 'species_grey sole', 'species_other', 'species_plaice',
                         'species_summer', 'species_windowpane', 'species_winter', 'no fish', 'hand over fish',
                         'fish clear', 'x', 'y', 'w', 'h', 'detection_conf', 'detection_species']
            self.video_data[video_id] = ds_combined.as_matrix(columns=self.columns)

            gt_combined = all_frames.merge(self.gt.loc[self.gt.video_id == video_id], on='frame', how='left').fillna(0.0)
            self.video_data_gt[video_id] = gt_combined.as_matrix(columns=['have_frame'])

    def generate_x(self, video_id, offset):
        res = np.zeros((NB_STEPS, NB_FEATURES))

        nb_frames = self.video_frames_count[video_id]
        steps_before = min(NB_STEPS_CROP, offset)
        steps_after = min(NB_STEPS_CROP, nb_frames - offset - NB_RES_STEPS)

        res[NB_STEPS_CROP - steps_before:NB_STEPS - NB_STEPS_CROP + steps_after, :] = \
            self.video_data[video_id][offset-steps_before:offset+NB_RES_STEPS+steps_after, :]
        return res

    def generate_y(self, video_id, offset):
        res = np.zeros((NB_RES_STEPS,))
        nb_frames = self.video_frames_count[video_id]
        frames_used = min(NB_RES_STEPS, nb_frames-offset)
        res[0:frames_used] = self.video_data_gt[video_id][offset:offset+frames_used, 0]
        return res

    def generate(self, batch_size):
        valid_video_ids = list(self.train_video_ids.intersection(self.video_data.keys()))

        batch_x = np.zeros((batch_size,)+INPUT_SHAPE, dtype=np.float32)
        batch_y = np.zeros((batch_size, NB_RES_STEPS), dtype=np.float32)
        # batch_y_sum = np.zeros((batch_size, NB_RES_STEPS), dtype=np.float32)
        while True:
            for batch_idx in range(batch_size):
                video_id = random.choice(valid_video_ids)

                if self.video_frames_count[video_id] < NB_RES_STEPS:
                    offset = 0
                else:
                    offset = random.randrange(0, self.video_frames_count[video_id]-NB_RES_STEPS)

                batch_x[batch_idx] = self.generate_x(video_id, offset)
                batch_y[batch_idx] = self.generate_y(video_id, offset)

            yield batch_x, {'current_values': batch_y, 'cumsum_values': np.cumsum(batch_y, axis=1)}

    def test_batches(self, batch_size):
        valid_video_ids = sorted(self.test_video_ids.intersection(self.video_data.keys()))
        batch_idx = 0
        batches_count = 0
        for video_id in valid_video_ids:
            for offset in range(0, self.video_frames_count[video_id], NB_RES_STEPS):
                batch_idx += 1

                if batch_idx == batch_size:
                    batch_idx = 0
                    batches_count += 1
        print('val batches count:', batches_count)
        return batches_count

    def generate_test(self, batch_size, verbose=False):
        valid_video_ids = sorted(self.test_video_ids.intersection(self.video_data.keys()))

        batch_x = np.zeros((batch_size,)+INPUT_SHAPE, dtype=np.float32)
        batch_y = np.zeros((batch_size, NB_RES_STEPS), dtype=np.float32)
        while True:
            batch_idx = 0
            for video_id in valid_video_ids:
                for offset in range(0, self.video_frames_count[video_id], NB_RES_STEPS):
                    if verbose:
                        print(video_id, offset)
                    batch_x[batch_idx] = self.generate_x(video_id, offset)
                    batch_y[batch_idx] = self.generate_y(video_id, offset)
                    batch_idx += 1

                    self.last_offset = offset

                    if batch_idx == batch_size:
                        batch_idx = 0
                        yield batch_x, {'current_values': batch_y, 'cumsum_values': np.cumsum(batch_y, axis=1)}


def train(fold=1):
    data = Dataset(fold)

    model = model_unet_cumsum()
    model.summary()

    model_name = 'model_squence_unet_cumsum'
    checkpoints_dir = '../output/checkpoints/sequence/{}_fold_{}'.format(model_name, fold)
    tensorboard_dir = '../output/tensorboard/sequence/{}_fold_{}'.format(model_name, fold)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    def cheduler(epoch):
        if epoch < 1:
            return 5e-4
        if epoch < 5:
            return 2e-4
        if epoch < 15:
            return 1e-4
        if epoch < 30:
            return 5e-5
        return 2e-5

    validation_batch_size = 16

    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5",
                                            verbose=1,
                                            save_weights_only=True,
                                            period=1)
    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=False, write_images=True)
    lr_sched = LearningRateScheduler(schedule=cheduler)

    nb_epoch = 800
    batch_size = 512
    model.fit_generator(data.generate(batch_size=batch_size),
                        steps_per_epoch=128,
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=[
                            checkpoint_periodical,
                            tensorboard,
                            lr_sched
                        ],
                        validation_data=data.generate_test(batch_size=validation_batch_size),
                        validation_steps=data.test_batches(validation_batch_size),
                        initial_epoch=0)


def check(fold=1):
    # data = Dataset(fold, load_only_video_ids=['00WK7DR6FyPZ5u3A'])
    data = Dataset(fold)
    model = model_unet_cumsum()
    model.load_weights('../output/checkpoints/sequence/model_squence_unet_cumsum_fold_1/checkpoint-028-0.0681.hdf5')

    for batch_x, yy in data.generate_test(batch_size=1, verbose=True):
        batch_y = yy['current_values']
        batch_y_cumsum = yy['cumsum_values']
        prediction, prediction_cumsum = model.predict_on_batch(batch_x)
        gt = batch_y[0, :]
        gt_cumsum = batch_y_cumsum[0, :]
        # print('gt:', batch_y[0, :, 0])
        # print('pr:', prediction[:, 0])
        # plt.plot(prediction)
        # plt.plot(gt*0.5)

        plt.figure()
        frames = range(data.last_offset, data.last_offset+NB_RES_STEPS)
        # plt.plot(frames, np.cumsum(prediction))
        # plt.plot(frames, np.cumsum(gt))

        plt.plot(frames, prediction[0], label='prediction')
        plt.plot(frames, gt, label='gt')

        plt.plot(frames, prediction_cumsum[0], label='prediction cumsum')
        plt.plot(frames, gt_cumsum, label='gt cumsum')

        plt.legend()
        # non_no_fish = 1.0 - batch_x[0, NB_STEPS_CROP:-NB_STEPS_CROP, data.columns.index('species__')]
        # plt.plot(frames, non_no_fish)

        fish_clear = batch_x[0, NB_STEPS_CROP:-NB_STEPS_CROP, data.columns.index('fish clear')]
        plt.plot(frames, fish_clear)

        plt.show()




if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='ruler masks')
    # parser.add_argument('action', type=str, default='check')
    # parser.add_argument('--weights', type=str, default='')
    # parser.add_argument('--fold', type=int, default=0)
    #
    # args = parser.parse_args()
    #
    # action = args.action
    # train()
    check()
