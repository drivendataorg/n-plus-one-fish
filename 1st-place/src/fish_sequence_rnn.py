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
from keras.layers import LSTM, GRU, Bidirectional, Dense
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

DETECTION_MODELS = ['resnet_53', 'resnet_62']
CLASSIFICATION_MODEL = 'densenet'

NB_FEATURES = 17 * len(DETECTION_MODELS)
NB_STEPS = 256
NB_STEPS_CROP = 32
NB_RES_STEPS = NB_STEPS - NB_STEPS_CROP * 2
INPUT_SHAPE = (NB_STEPS, NB_FEATURES)

USE_CUMSUM = True


def model_gru1(input_shape=INPUT_SHAPE):
    inputs = Input(shape=input_shape)
    x = Bidirectional(GRU(64, return_sequences=True, unroll=True))(inputs)
    x = Bidirectional(GRU(16, return_sequences=True, unroll=True))(x)
    x = Dense(1, activation='sigmoid')(x)
    x = Cropping1D(cropping=(NB_STEPS_CROP, NB_STEPS_CROP))(x)
    x = Flatten()(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def model_gru1_cumsum(input_shape=INPUT_SHAPE, unroll=True):
    inputs = Input(shape=input_shape)
    x = Bidirectional(GRU(64, return_sequences=True, unroll=unroll))(inputs)
    x = Bidirectional(GRU(16, return_sequences=True, unroll=unroll))(x)
    x = Dense(1, activation='sigmoid')(x)
    x = Cropping1D(cropping=(NB_STEPS_CROP, NB_STEPS_CROP))(x)
    x = Flatten(name='current_values')(x)

    cumsum_value = Lambda(lambda a: K.cumsum(a, axis=1), name='cumsum_values')(x)

    model = Model(inputs=inputs, outputs=[x, cumsum_value])
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'current_values': 'binary_crossentropy', 'cumsum_values': 'mse'},
                  loss_weights={'current_values': 1.0, 'cumsum_values': 0.001})

    return model


def model_lstm2_cumsum(input_shape=INPUT_SHAPE):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(16, return_sequences=True))(x)
    x = Dense(1, activation='sigmoid')(x)
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
                 train_on_all_dataset=True,
                 load_only_video_ids=None,
                 is_test=False):

        self.video_clips = dataset.video_clips(is_test=is_test)

        if load_only_video_ids is not None:
            all_video_ids = set(load_only_video_ids)
        else:
            all_video_ids = set(self.video_clips.keys())

        if train_on_all_dataset:
            self.test_video_ids = []
            self.train_video_ids = all_video_ids
        else:
            self.test_video_ids = set(dataset.fold_test_video_ids(fold))
            self.train_video_ids = all_video_ids.difference(self.test_video_ids)

        self.gt = pd.read_csv('../input/N1_fish_N2_fish_-_Training_set_annotations.csv')
        self.gt.dropna(axis=0, inplace=True)
        self.gt['have_frame'] = 1.0

        self.video_frames_count = {}
        self.video_data = {}
        self.video_data_gt = {}

        print('load video data...')
        cache_fn = '../output/sequence_rnn_test.pkl' if is_test else '../output/sequence_rnn_train.pkl'
        try:
            self.video_frames_count, self.video_data, self.video_data_gt, self.columns = utils.load_data(cache_fn)
        except FileNotFoundError:
            self.video_frames_count, self.video_data, self.video_data_gt, self.columns = self.load(all_video_ids,
                                                                          detection_results_dir,
                                                                          classification_results_dir)
            utils.save_data((self.video_frames_count, self.video_data, self.video_data_gt, self.columns), cache_fn)
        print('loaded')

    def load(self, all_video_ids, detection_results_dir, classification_results_dir):
        print('generate video data...')
        video_frames_count = {}
        video_data = {}
        video_data_gt = {}
        columns = {}

        for video_id in sorted(all_video_ids):
            results = []
            nb_frames = len(self.video_clips[video_id])
            video_frames_count[video_id] = nb_frames

            for detection_model in DETECTION_MODELS:
                # try:
                ds_detection = pd.read_csv(
                    os.path.join(detection_results_dir, detection_model, video_id + '_ssd_detection.csv'))
                ds_classification = pd.read_csv(
                    os.path.join(classification_results_dir, detection_model, CLASSIFICATION_MODEL, video_id + '_categories.csv'))
                # except FileNotFoundError:
                #     continue

                ds_combined = ds_classification.join(ds_detection, on='frame', how='left', rsuffix='_det')
                ds_combined.x /= IMG_WITH
                ds_combined.y /= IMG_HEIGHT
                ds_combined.w /= IMG_WITH
                ds_combined.h /= IMG_HEIGHT
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

                columns = ['species__', 'species_fourspot', 'species_grey sole', 'species_other', 'species_plaice',
                                'species_summer', 'species_windowpane', 'species_winter', 'no fish', 'hand over fish',
                                'fish clear', 'x', 'y', 'w', 'h', 'detection_conf', 'detection_species']
                results.append(ds_combined.as_matrix(columns=columns))

            video_data[video_id] = np.hstack(results)

            all_frames = pd.DataFrame({'frame': list(range(nb_frames))})
            gt_combined = all_frames.merge(self.gt.loc[self.gt.video_id == video_id], on='frame', how='left').fillna(
                0.0)
            video_data_gt[video_id] = gt_combined.as_matrix(columns=['have_frame'])

        return video_frames_count, video_data, video_data_gt, columns

    def generate_x(self, video_id, offset, nb_steps=NB_STEPS):
        res = np.zeros((nb_steps, NB_FEATURES))
        nb_res_steps = nb_steps - NB_STEPS_CROP * 2

        nb_frames = self.video_frames_count[video_id]
        steps_before = min(NB_STEPS_CROP, offset)
        steps_after = min(NB_STEPS_CROP, nb_frames - offset - nb_res_steps)

        res[NB_STEPS_CROP - steps_before:nb_steps - NB_STEPS_CROP + steps_after, :] = \
            self.video_data[video_id][offset - steps_before:offset + nb_res_steps + steps_after, :]
        return res

    def generate_y(self, video_id, offset):
        res = np.zeros((NB_RES_STEPS,))
        nb_frames = self.video_frames_count[video_id]
        frames_used = min(NB_RES_STEPS, nb_frames - offset)
        res[0:frames_used] = self.video_data_gt[video_id][offset:offset + frames_used, 0]
        return res

    def generate(self, batch_size, use_cumsum=False):
        valid_video_ids = list(self.train_video_ids.intersection(self.video_data.keys()))

        batch_x = np.zeros((batch_size,) + INPUT_SHAPE, dtype=np.float32)
        batch_y = np.zeros((batch_size, NB_RES_STEPS), dtype=np.float32)
        # batch_y_sum = np.zeros((batch_size, NB_RES_STEPS), dtype=np.float32)
        while True:
            for batch_idx in range(batch_size):
                video_id = random.choice(valid_video_ids)

                if self.video_frames_count[video_id] < NB_RES_STEPS:
                    offset = 0
                else:
                    offset = random.randrange(0, self.video_frames_count[video_id] - NB_RES_STEPS)

                batch_x[batch_idx] = self.generate_x(video_id, offset)
                batch_y[batch_idx] = self.generate_y(video_id, offset)

            if use_cumsum:
                yield batch_x, {'current_values': batch_y, 'cumsum_values': np.cumsum(batch_y, axis=1)}
            else:
                yield batch_x, batch_y

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

    def generate_test(self, batch_size, verbose=False, use_cumsum=False):
        valid_video_ids = sorted(set(self.test_video_ids).intersection(self.video_data.keys()))
        print('test valid_video_ids:', len(valid_video_ids))

        batch_x = np.zeros((batch_size,) + INPUT_SHAPE, dtype=np.float32)
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
                        if use_cumsum:
                            yield batch_x, {'current_values': batch_y, 'cumsum_values': np.cumsum(batch_y, axis=1)}
                        else:
                            yield batch_x, batch_y


def train(fold=1, use_cumsum=USE_CUMSUM, train_on_all_dataset=True):
    data = Dataset(fold, train_on_all_dataset=train_on_all_dataset)

    if use_cumsum:
        model = model_gru1_cumsum()
        model_name = 'model_squence_gru1_cumsum'
    else:
        model = model_gru1()
        model_name = 'model_squence_gru1'

    if train_on_all_dataset:
        model_name += '_all'

    model.summary()

    checkpoints_dir = '../output/checkpoints/sequence/{}_fold_{}'.format(model_name, fold)
    tensorboard_dir = '../output/tensorboard/sequence/{}_fold_{}'.format(model_name, fold)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    def cheduler(epoch):
        if epoch < 10:
            return 5e-4
        if epoch < 20:
            return 2e-4
        if epoch < 50:
            return 1e-4
        if epoch < 100:
            return 5e-5
        return 2e-5

    validation_batch_size = 16


    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=False, write_images=True)
    lr_sched = LearningRateScheduler(schedule=cheduler)

    nb_epoch = 800
    batch_size = 128

    if train_on_all_dataset:
        checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{loss:.5f}.hdf5",
                                                verbose=1,
                                                save_weights_only=True,
                                                period=1)
        validation_data = None
        validation_steps = None
    else:
        checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.5f}.hdf5",
                                                verbose=1,
                                                save_weights_only=True,
                                                period=1)
        validation_data = data.generate_test(batch_size=validation_batch_size, use_cumsum=use_cumsum),
        validation_steps = data.test_batches(validation_batch_size),

    model.fit_generator(data.generate(batch_size=batch_size, use_cumsum=use_cumsum),
                        steps_per_epoch=512,
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=[
                            checkpoint_periodical,
                            tensorboard,
                            lr_sched
                        ],
                        validation_data=validation_data,
                        validation_steps=validation_steps,
                        initial_epoch=0)


def check(fold=1, use_cumsum=USE_CUMSUM):
    # data = Dataset(fold, load_only_video_ids=['00WK7DR6FyPZ5u3A'])
    data = Dataset(fold, train_on_all_dataset=False)
    if use_cumsum:
        model = model_gru1_cumsum()
    else:
        model = model_gru1()

    # model.load_weights('../output/checkpoints/sequence/model_squence_gru1_cumsum_fold_1/checkpoint-201-0.0484.hdf5')
    # model.load_weights('../output/checkpoints/sequence/model_squence_gru1_cumsum_fold_1/checkpoint-146-0.0509.hdf5')
    # model.load_weights('../output/checkpoints/sequence/model_squence_gru1_cumsum_fold_1/checkpoint-547-0.0439.hdf5')
    model.load_weights('../output/checkpoints/sequence/model_squence_gru1_cumsum_all_fold_0/checkpoint-110-0.06723.hdf5')

    for batch_x, yy in data.generate_test(batch_size=1, verbose=True, use_cumsum=use_cumsum):
        if use_cumsum:
            batch_y = yy['current_values']
            batch_y_cumsum = yy['cumsum_values']
            prediction, prediction_cumsum = model.predict_on_batch(batch_x)
            gt = batch_y[0, :]
            gt_cumsum = batch_y_cumsum[0, :]
        else:
            batch_y = yy

            prediction = model.predict_on_batch(batch_x)
            prediction_cumsum = np.cumsum(prediction, axis=1)

            gt = batch_y[0, :]
            gt_cumsum = np.cumsum(batch_y[0, :])

        # print('gt:', batch_y[0, :, 0])
        # print('pr:', prediction[:, 0])
        # plt.plot(prediction)
        # plt.plot(gt*0.5)

        plt.figure()
        frames = range(data.last_offset, data.last_offset + NB_RES_STEPS)
        # plt.plot(frames, np.cumsum(prediction))
        # plt.plot(frames, np.cumsum(gt))

        plt.plot(frames, prediction[0], label='prediction lstm')
        plt.plot(frames, gt, label='gt')

        plt.plot(frames, prediction_cumsum[0], label='prediction cumsum')
        plt.plot(frames, gt_cumsum, label='gt cumsum')

        plt.legend()
        # non_no_fish = 1.0 - batch_x[0, NB_STEPS_CROP:-NB_STEPS_CROP, data.columns.index('species__')]
        # plt.plot(frames, non_no_fish)

        fish_clear = batch_x[0, NB_STEPS_CROP:-NB_STEPS_CROP, data.columns.index('fish clear')]
        plt.plot(frames, fish_clear)

        plt.show()


def predict_test(output_dir):
    data = Dataset(fold=0,
                   detection_results_dir='../output/detection_results_test',
                   classification_results_dir='../output/classification_results_test_combined',
                   train_on_all_dataset=False,
                   is_test=True)
    max_size = 10000
    model = model_gru1_cumsum(input_shape=(max_size, NB_FEATURES), unroll=False)
    model.load_weights('../output/checkpoints/sequence/model_squence_gru1_cumsum_all_fold_0/checkpoint-110-0.06723.hdf5')
    os.makedirs(output_dir, exist_ok=True)

    video_ids = sorted(list(data.video_clips.keys()))
    X = np.array([data.generate_x(video_id, offset=0, nb_steps=max_size) for video_id in video_ids])
    y = model.predict(X, batch_size=64, verbose=1)
    print(y)
    print(np.array(y).shape)
    np.save(os.path.join(output_dir, 'key_fish_prob.npy'), np.array(y))
    np.save(os.path.join(output_dir, 'key_fish_ids.npy'), np.array(video_ids))
    #
    # for video_id in utils.chunks(data.video_clips.keys(), n=128):
    #     print(video_id)
    #     X = data.generate_x(video_id, offset=0, nb_steps=max_size)
    #     res = model.predict(np.array([X]), batch_size=1)[0]
    #     print(res.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find key fish with rnn')
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--detection_model', type=str, default='')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--initial_epoch', type=int, default=0)

    args = parser.parse_args()

    action = args.action

    if action == 'train_full':
        train(fold=0, train_on_all_dataset=True)
    if action == 'check':
        check(fold=args.fold)
    if action == 'predict_test':
        predict_test(output_dir='../output/sequence_results_test')
    # parser = argparse.ArgumentParser(description='ruler masks')
    # parser.add_argument('action', type=str, default='check')
    # parser.add_argument('--weights', type=str, default='')
    # parser.add_argument('--fold', type=int, default=0)
    #
    # args = parser.parse_args()
    #
    # action = args.action
    # train()
    # check()
