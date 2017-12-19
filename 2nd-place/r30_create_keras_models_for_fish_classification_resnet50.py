# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- This code train 5KFold models for classification of fishes (e.g. find type of fish by crop from frame)
- Neural net based on ResNet50. Pretrained weights are the part of Keras module
- Can be run in parallel for 5 Folds on 5 GPUs (use FOLD_TO_CALC constant).
- You can skip this part if you already have models with name 'RESNET50_fold_*.h5' in ../models/ directory
'''

import platform
import sys
import os

gpu_use = 1
# FOLD_TO_CALC = [5]
if platform.processor() == 'Intel64 Family 6 Model 63 Stepping 2, GenuineIntel' or platform.processor() == 'Intel64 Family 6 Model 79 Stepping 1, GenuineIntel':
    os.environ["THEANO_FLAGS"] = "device=gpu{},lib.cnmem=0.81,base_compiledir='C:\\\\Users\\\\user\\\\AppData\\\\Local\\\\Theano{}'".format(gpu_use, gpu_use)
    if sys.version_info[1] > 4:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
from a00_augmentation_functions import *
import datetime
import shutil
import random
from a02_zoo import *
from keras.utils import np_utils

random.seed(2016)
np.random.seed(2016)

PATIENCE = 50
NB_EPOCH = 1000
RESTORE_FROM_LAST_CHECKPOINT = 0
UPDATE_BEST_MODEL = 0

INPUT_PATH = "../input/"
MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
HISTORY_FOLDER_PATH = "../models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)


TRAIN_TABLE = None


def get_labels(files):
    labels = []
    for f in files:
        flag = 0
        for i in range(len(FISH_TABLE)):
            name = FISH_TABLE[i]
            if name in f:
                labels.append(i)
                flag = 1
                break
        if flag == 0:
            if '__' not in f:
                print('Strange')
                exit()
            labels.append(7)

    # convert class vectors to binary class matrices
    labels = np_utils.to_categorical(labels, 8)
    return labels


def get_random_boxes(batch_files, train_table):
    boxes = []
    for i in range(len(batch_files)):
        f = batch_files[i]
        row_id = int(os.path.basename(f).split('_')[0])
        row = train_table[train_table['row_id'] == row_id]
        if np.isnan(row['x1'].values[0]):
            if random.randint(0, 1) == 0:
                # get random box from picture
                box_size = random.randint(100, 500)
                x1 = random.randint(0, 1280-box_size)
                y1 = random.randint(0, 720-box_size)
                x2 = x1 + box_size
                y2 = y1 + box_size
            else:
                # get box from some fish place in same video
                video_id = row['video_id'].values[0]
                row = train_table[(train_table['video_id'] == video_id) & (~train_table['x1'].isnull())]
                if len(row) > 0:
                    random_row = random.randint(0, len(row)-1)
                    x1 = int(row['x1'].values[random_row])
                    y1 = int(row['y1'].values[random_row])
                    x2 = int(row['x2'].values[random_row])
                    y2 = int(row['y2'].values[random_row])
                else:
                    # get random box from picture
                    box_size = random.randint(100, 500)
                    x1 = random.randint(0, 1280 - box_size)
                    y1 = random.randint(0, 720 - box_size)
                    x2 = x1 + box_size
                    y2 = y1 + box_size
        else:
            x1 = int(row['x1'].values[0])
            y1 = int(row['y1'].values[0])
            x2 = int(row['x2'].values[0])
            y2 = int(row['y2'].values[0])

        if x2 < x1:
            c = x2
            x2 = x1
            x1 = c
        if y2 < y1:
            c = y2
            y2 = y1
            y1 = c

        delta = 50
        if abs(x2 - x1) < delta:
            x1 -= delta
            x2 += delta
            if x1 < 0:
                x1 = 0
        if abs(y2 - y1) < delta:
            y1 -= delta
            y2 += delta
            if y1 < 0:
                y1 = 0

        # Add random border
        y1 += random.randint(-200, 5)
        if y1 < 0:
            y1 = 0
        if y1 > 720:
            y1 = 720

        y2 += random.randint(-5, 200)
        if y2 < 0:
            y2 = 0
        if y2 > 720:
            y2 = 720

        x1 += random.randint(-200, 5)
        if x1 < 0:
            x1 = 0
        if x1 > 1280:
            x1 = 1280

        x2 += random.randint(-5, 200)
        if x2 < 0:
            x2 = 0
        if x2 > 1280:
            x2 = 1280

        boxes.append((y1, y2, x1, x2))

    return boxes


def batch_generator_train(cnn_type, files, augment=False):
    global TRAIN_TABLE
    import keras.backend as K

    files_by_class = get_dict_of_files_by_classes(files)
    dim_ordering = K.image_dim_ordering()
    in_shape = get_input_shape(cnn_type)
    batch_size = get_batch_size(cnn_type)

    if TRAIN_TABLE is None:
        TRAIN_TABLE = pd.read_csv(INPUT_PATH + "training.csv")

    while True:
        batch_files = np.empty((0,), dtype=np.str)
        for el in files_by_class:
            batch_files = np.concatenate((batch_files, np.random.choice(files_by_class[el], ((batch_size - 1) // 8) + 1)), axis=0)
        if len(batch_files) > batch_size:
            batch_files = np.random.choice(batch_files, batch_size, replace=False)
        batch_labels = get_labels(batch_files)
        batch_boxes = get_random_boxes(batch_files, TRAIN_TABLE)

        image_list = []
        labels_list = []
        for i in range(len(batch_files)):
            image = cv2.imread(batch_files[i])
            sh0_start, sh0_end, sh1_start, sh1_end = batch_boxes[i]
            label = batch_labels[i]

            image = image[sh0_start:sh0_end, sh1_start:sh1_end]
            # print(label)
            # print(sh0_start, sh0_end, sh1_start, sh1_end)
            # show_resized_image(image, 224, 224)
            # show_image(image)
            image = cv2.resize(image, in_shape, cv2.INTER_LANCZOS4)

            if augment:
                image = random_rotate(image.copy(), 45)
                image = get_random_mirror(image)
                image = random_intensity_change(image, 20)

            image_list.append(image.astype(np.float32))
            labels_list.append(label)
        image_list = np.array(image_list)
        if dim_ordering == 'th':
            image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input_overall(cnn_type, image_list)
        labels_list = np.array(labels_list)
        yield image_list, labels_list


def train_single_classification_model_full_frame(cnn_type, num_fold, train_index, test_index, files):
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    print('Creating and compiling model [{}]...'.format(cnn_type))
    model = get_pretrained_model(cnn_type, 8, final_layer_activation='softmax')

    final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
    cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
    if os.path.isfile(cache_model_path) and RESTORE_FROM_LAST_CHECKPOINT:
        print('Load model from last point: ', cache_model_path)
        model.load_weights(cache_model_path)
    elif os.path.isfile(final_model_path) and UPDATE_BEST_MODEL:
        print('Load model from best point: ', final_model_path)
        model.load_weights(final_model_path)
    else:
        print('Start training from begining')

    print('Fitting model...')
    train_files = files[train_index]
    valid_files = files[test_index]

    batch_size = get_batch_size(cnn_type)
    print('Batch size: {}'.format(batch_size))
    print('Learning rate: {}'.format(get_learning_rate(cnn_type)))
    samples_train_per_epoch = 1600
    samples_valid_per_epoch = 1600
    print('Samples train: {}, Samples valid: {}'.format(samples_train_per_epoch, samples_valid_per_epoch))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(generator=batch_generator_train(cnn_type, train_files, True),
                  nb_epoch=NB_EPOCH,
                  samples_per_epoch=samples_train_per_epoch,
                  validation_data=batch_generator_train(cnn_type, valid_files, True),
                  nb_val_samples=samples_valid_per_epoch,
                  verbose=2, max_q_size=20,
                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}_weather.csv'.format(cnn_type, num_fold, min_loss, get_learning_rate(cnn_type), now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


def run_cross_validation_create_models(nfolds, cnn_type):
    global FOLD_TO_CALC

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

        score = train_single_classification_model_full_frame(cnn_type, num_fold, train_index, test_index, np.array(files))
        sum_score += score

    print('Avg loss: {}'.format(sum_score / nfolds))


if __name__ == '__main__':
    run_cross_validation_create_models(5, 'RESNET50')
