# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import bz2
import pickle
import os
import glob
import time
import cv2
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
from multiprocessing import Process, Manager
import random


random.seed(2016)
np.random.seed(2016)

INPUT_PATH = '../input/'
OUTPUT_PATH = '../modified_data/'
FISH_TABLE = ['species_fourspot', 'species_grey_sole', 'species_other', 'species_plaice', 'species_summer', 'species_windowpane', 'species_winter']
COLOR_TABLE = {
    '': 0,
    'species_fourspot': 36,
    'species_grey_sole': 72,
    'species_other': 108,
    'species_plaice': 144,
    'species_summer': 180,
    'species_windowpane': 216,
    'species_winter': 255,
}


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3), protocol=4)


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def rle_encode(pixels):
    pixels = np.concatenate(([0], pixels, [0]))
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def dice(im1, im2, empty_score=1.0):
    im1 = im1.astype(np.bool)
    im2 = im2.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum


def get_dict_of_files_by_classes(files):
    ret = dict()
    for c in FISH_TABLE:
        ret[c] = []
        for f in files:
           if c in f:
               ret[c].append(f)

    ret['empty'] = []
    for f in files:
        if '__' in f:
            ret['empty'].append(f)

    sum = 0
    for el in sorted(ret.keys()):
        print(el, len(ret[el]))
        sum += len(ret[el])
    if sum != len(files):
        print('Unexpected!')
        exit()

    return ret


def get_kfold_split(nfolds):
    from sklearn.model_selection import KFold
    kfold_cache_path = OUTPUT_PATH + 'kfold_cache.pklz'
    if not os.path.isfile(kfold_cache_path):
        files = glob.glob(INPUT_PATH + "train_images/*.jpg")
        print('Unique files found: {}'.format(len(files)))
        videos = dict()
        for f in files:
            video_id = os.path.basename(f).split('_')[1]
            videos[video_id] = 1
        unique_videos = np.array(list(sorted(videos.keys())))
        print('Different videos: {}'.format(len(unique_videos)))
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=66)
        ret = []
        ret_videos = []
        for train_vid, test_vid in kf.split(range(len(unique_videos))):
            train_index = []
            test_index = []
            total = 0
            for f in files:
                video_id = os.path.basename(f).split('_')[1]
                if video_id in unique_videos[train_vid]:
                    train_index.append(total)
                elif video_id in unique_videos[test_vid]:
                    test_index.append(total)
                else:
                    print('Strange...')
                    exit()
                total += 1
            ret.append((train_index, test_index))
            ret_videos.append((train_vid, test_vid))
        save_in_file((files, ret, unique_videos, ret_videos), kfold_cache_path)
    else:
        files, ret, unique_videos, ret_videos = load_from_file(kfold_cache_path)

    return files, ret, unique_videos, ret_videos


def get_kfold_split_based_on_groups(nfolds):
    from sklearn.model_selection import KFold
    groups = pd.read_csv("../modified_data/groups.csv")
    # print(groups['group'].value_counts())
    unique_groups = groups['group'].unique()
    print('Different groups:', len(unique_groups))

    kfold_cache_path = OUTPUT_PATH + 'kfold_cache_groups.pklz'
    if not os.path.isfile(kfold_cache_path):
        files = glob.glob(INPUT_PATH + "train_images/*.jpg")
        print('Unique files found: {}'.format(len(files)))
        videos = dict()
        for f in files:
            video_id = os.path.basename(f).split('_')[1]
            videos[video_id] = 1
        unique_videos = np.array(list(sorted(videos.keys())))
        print('Different videos: {}'.format(len(unique_videos)))
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=66)
        ret = []
        ret_videos = []
        for train_group, test_group in kf.split(range(len(unique_groups))):
            total = 0
            train_vid = []
            test_vid = []
            for v in unique_videos:
                gr = groups.loc[groups['video_id'] == v, 'group'].values[0]
                if gr in unique_groups[train_group]:
                    train_vid.append(total)
                elif gr in unique_groups[test_group]:
                    test_vid.append(total)
                else:
                    print('Strange {}...'.format(gr))
                    exit()
                total += 1

            total = 0
            train_index = []
            test_index = []
            for f in files:
                video_id = os.path.basename(f).split('_')[1]
                if video_id in unique_videos[train_vid]:
                    train_index.append(total)
                elif video_id in unique_videos[test_vid]:
                    test_index.append(total)
                else:
                    print('Strange...')
                    exit()
                total += 1
            ret.append((train_index, test_index))
            ret_videos.append((train_vid, test_vid))
        save_in_file((files, ret, unique_videos, ret_videos), kfold_cache_path)
    else:
        files, ret, unique_videos, ret_videos = load_from_file(kfold_cache_path)

    if 0:
        print(unique_videos[ret_videos[0][1]])
        dt = []
        for v in unique_videos[ret_videos[0][1]]:
            print(groups.loc[groups['video_id'] == v, 'group'].values[0])
            dt.append(groups.loc[groups['video_id'] == v, 'group'].values[0])
        print(set(dt))
        print(ret_videos)
        print(len(ret_videos[0][1]))
        print(len(ret_videos[1][1]))
        print(len(ret_videos[2][1]))
        print(len(ret_videos[3][1]))
        print(len(ret_videos[4][1]))

    return files, ret, unique_videos, ret_videos


def get_bbox_dict():
    bbox = dict()
    # Order: start_0, start_1, end_0, end_1
    f = open('../modified_data/bounding_boxes_train.csv')
    f.readline()
    while 1:
        line = f.readline()
        if line == '':
            break
        arr = line.split(',')
        bbox[arr[0]] = [int(arr[1]), int(arr[3]), int(arr[2]), int(arr[4])]
    f = open('../modified_data/bounding_boxes_test.csv')
    f.readline()
    while 1:
        line = f.readline()
        if line == '':
            break
        arr = line.split(',')
        bbox[arr[0]] = [int(arr[1]), int(arr[3]), int(arr[2]), int(arr[4])]

    # print('Finish reading BBoxes!')
    return bbox


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes
