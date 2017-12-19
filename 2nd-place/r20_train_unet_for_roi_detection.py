# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- This code train 5KFold models for segmentation of fishes (e.g. find exact fish location)
- It requires 2-3 days to complete. Can be run in parallel for 5 Folds on 5 GPUs
- You can skip this part if you already have models with name 'ZF_UNET_1280_720_V2_SINGLE_OUTPUT_SMALL_fold_*.h5'
  in ../models/ directory
'''

import os
import sys
import platform
from a00_augmentation_functions import *

gpu_use = 1
FOLD_TO_CALC = [1, 2, 3, 4, 5]
if platform.processor() == 'Intel64 Family 6 Model 63 Stepping 2, GenuineIntel' or platform.processor() == 'Intel64 Family 6 Model 79 Stepping 1, GenuineIntel':
    os.environ["THEANO_FLAGS"] = "device=gpu{},lib.cnmem=0.81,,base_compiledir='C:\\\\Users\\\\user\\\\AppData\\\\Local\\\\Theano{}'".format(gpu_use, gpu_use)
    if sys.version_info[1] > 4:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import datetime
from a00_common_functions import *
from a02_zf_unet_model import *


INPUT_PATH = "../input/train_images/"
MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
HISTORY_FOLDER_PATH = "../models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)


def bbox1(img):
    a = np.where(img > 0)
    if a[0].shape[0] > 0:
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    else:
        bbox = 0, 0, 0, 0
    return bbox


def batch_generator_train_full_frame(files, batch_size):
    border_limit = 100

    while True:
        batch_images = []
        batch_masks = []
        batch_files = np.random.choice(files, batch_size)
        for f in batch_files:
            im_full = cv2.imread(f)
            mask_path = f[:-9] + '_mask.png'
            im_mask = cv2.imread(mask_path, 0)
            im_mask[im_mask > 0] = 255

            if 1:
                im_full, im_mask = random_rotate_with_mask(im_full, im_mask, 5)
                bbox = bbox1(im_mask)

                # Random crop and resize
                if bbox == (0, 0, 0, 0):
                    start_0 = random.randint(0, border_limit)
                    start_1 = random.randint(0, border_limit)
                    end_0 = random.randint(im_full.shape[0] - border_limit, im_full.shape[0])
                    end_1 = random.randint(im_full.shape[1] - border_limit, im_full.shape[1])
                else:
                    start_0 = random.randint(0, min(border_limit, bbox[0]))
                    start_1 = random.randint(0, min(border_limit, bbox[2]))
                    end_0 = random.randint(max(im_full.shape[0] - border_limit, bbox[1]), im_full.shape[0])
                    end_1 = random.randint(max(im_full.shape[1] - border_limit, bbox[3]), im_full.shape[1])

                im_full = im_full[start_0:end_0, start_1:end_1, :]
                im_mask = im_mask[start_0:end_0, start_1:end_1]

                im_full = cv2.resize(im_full, (1280, 720), cv2.INTER_LANCZOS4)
                im_mask = cv2.resize(im_mask, (1280, 720), cv2.INTER_LANCZOS4)

                im_full = random_intensity_change(im_full, 10)
                if random.randint(0, 1) == 0:
                    # fliplr
                    im_full = im_full[:, ::-1, :]
                    im_mask = im_mask[:, ::-1]

                im_mask[im_mask < 255] = 0

            # show_image(im_full)
            # show_image(im_mask)

            batch_images.append(im_full)
            batch_masks.append(im_mask)

        batch_images = np.array(batch_images, dtype=np.float32)
        batch_masks = np.array(batch_masks, dtype=np.float32)
        batch_masks[batch_masks > 0] = 1

        batch_images = preprocess_batch(batch_images)
        batch_images = batch_images.transpose((0, 3, 1, 2))
        batch_masks = np.expand_dims(batch_masks, axis=1)

        yield batch_images, batch_masks


def train_single_model_full_frame(num_fold, train_index, test_index, files):
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.optimizers import Adam, SGD

    print('Creating and compiling UNET...')
    restore = 1
    patience = 50
    epochs = 300
    optim_type = 'Adam'
    learning_rate = 0.001
    cnn_type = 'ZF_UNET_1280_720_V2_SINGLE_OUTPUT_SMALL'
    model = ZF_UNET_1280_720_V2_SINGLE_OUTPUT_SMALL(dropout_val=0, batch_norm=True)

    final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
    if os.path.isfile(final_model_path) and restore == 0:
        print('Model already exists for fold {}.'.format(final_model_path))
        return 0.0

    cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
    if os.path.isfile(cache_model_path) and restore:
        print('Load model from last point: ', cache_model_path)
        model.load_weights(cache_model_path)
    else:
        print('Start training from begining')

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[dice_coef])

    print('Fitting model...')
    train_files = files[train_index]
    test_files = files[test_index]

    batch_size = 12
    print('Batch size: {}'.format(batch_size))
    samples_train_per_epoch = batch_size*100
    samples_valid_per_epoch = batch_size*100
    print('Samples train: {}, Samples valid: {}'.format(samples_train_per_epoch, samples_valid_per_epoch))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(generator=batch_generator_train_full_frame(train_files, batch_size),
                  nb_epoch=epochs,
                  samples_per_epoch=samples_train_per_epoch,
                  validation_data=batch_generator_train_full_frame(test_files, batch_size),
                  nb_val_samples=samples_valid_per_epoch,
                  verbose=2, max_q_size=20,
                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}.csv'.format(cnn_type, num_fold, min_loss, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


def run_cross_validation_create_models_unet2(nfolds=5):
    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(nfolds)
    num_fold = 0
    sum_score = 0
    for train_index, test_index in kfold_images_split:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split frames train: ', len(train_index))
        print('Split frames valid: ', len(test_index))

        if 'FOLD_TO_CALC' in globals():
            if num_fold not in FOLD_TO_CALC:
                continue

        score = train_single_model_full_frame(num_fold, train_index, test_index, np.array(files))
        sum_score += score

    print('Avg loss: {}'.format(sum_score/nfolds))


if __name__ == '__main__':
    num_folds = 5
    run_cross_validation_create_models_unet2(num_folds)
