# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


'''
- Get length of fish for each frame, based on ROI matrices
'''

import os
import sys
import platform
import math
import shutil

from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from a00_common_functions import *

random.seed(2016)
np.random.seed(2016)

INPUT_PATH = "../input/"
DIV_POINT_V1 = 200
DIV_POINT_V2 = 70


CACHE_PATH_TRAIN = "../cache_roi_train/"
CACHE_PATH_TEST = "../cache_roi_test/"
CACHE_LENGTH_VALID = "../cache_length_train/"
if not os.path.isdir(CACHE_LENGTH_VALID):
    os.mkdir(CACHE_LENGTH_VALID)
CACHE_LENGTH_TEST = "../cache_length_test/"
if not os.path.isdir(CACHE_LENGTH_TEST):
    os.mkdir(CACHE_LENGTH_TEST)
ADD_PATH = '../modified_data/'


def get_length_from_mask(msk, bbox, div_point):
    avg = msk.copy()
    avg[avg > div_point] = 255
    avg[avg < div_point] = 0
    _, contours, hierarchy = cv2.findContours(avg.copy(), 1, 2)
    if len(contours) == 0:
        print('Some problem here')
        exit()
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    bbox_height = bbox[1] - bbox[0]
    bbox_width = bbox[3] - bbox[2]

    if bbox_height > bbox_width and bbox[3] < 1000 and bbox[1] < 600:
        length = math.sqrt(w * w + h * h)
    elif bbox_width > bbox_height:
        length = w
    # elif bbox_height > bbox_width:
    else:
        length = h
    return length


def get_length_for_train():
    videos = glob.glob(INPUT_PATH + 'train_videos/*.mp4')
    train = pd.read_csv(INPUT_PATH + 'training.csv')
    bboxes = pd.read_csv(ADD_PATH + "bboxes_train.csv")

    avg_score = 0.0
    counter = 0
    # videos = ['464cHVEP2rxuqYzg.jpg']
    for v in videos[:]:
        name = os.path.basename(v)[:-4]
        cache_path = CACHE_LENGTH_VALID + name + '_length.pklz'
        if os.path.isfile(cache_path):
            print('Length for {} already exists!'.format(name))
            continue
        real = train[train['video_id'] == name].copy()
        real.fillna(0, inplace=True)
        fish = real['length'] > 0
        frames_to_extract = list(real[fish]['frame'].values)
        real_length_values = list(real[fish]['length'].values)
        bbox = list(bboxes.loc[bboxes['id'] == name, ['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']].values[0])

        print('Go for {}'.format(name))
        print('BBox', bbox)
        roi_files = glob.glob(CACHE_PATH_TRAIN + name + '_*')
        masks_list = []
        for f in roi_files:
            mask = load_from_file(f)
            print(mask.shape)
            masks_list.append(mask)
        masks_list = np.concatenate(masks_list, axis=0)
        masks = masks_list

        div_point = DIV_POINT_V1
        if bbox[3] > 1100:
            div_point = DIV_POINT_V2

        print('Initial frames: {}'.format(masks.shape[0]))
        pred_length_values = []
        full_length_values = []
        for i in range(masks.shape[0]):
            if masks[i].max() > div_point:
                l = get_length_from_mask(masks[i], bbox, div_point)
                # show_image(masks[i])
                full_length_values.append(l)
            else:
                full_length_values.append(0.0)
            if i in frames_to_extract:
                pred_length_values.append(full_length_values[-1])

        print(real_length_values)
        print(pred_length_values)

        score = r2_score(real_length_values, pred_length_values)
        print('Score:', score)
        save_in_file(full_length_values, cache_path)
        if score < 0:
            score = 0
        avg_score += score
        counter += 1

    print('AVG score: {}'.format(avg_score/counter))


def get_length_for_test():
    videos = glob.glob(INPUT_PATH + 'test_videos/*.mp4')
    bboxes = pd.read_csv(ADD_PATH + "bboxes_test.csv")

    avg_score = 0.0
    counter = 0
    # videos = ['7xW9n6yQcGgx6c2b.mp4']
    for v in videos:
        name = os.path.basename(v)[:-4]
        cache_path = CACHE_LENGTH_TEST + name + '_length.pklz'
        if os.path.isfile(cache_path):
            print('Length for {} already exists!'.format(name))
            continue
        print('Go for {}'.format(name))
        bbox = list(bboxes.loc[bboxes['id'] == name, ['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']].values[0])
        print('BBox', bbox)
        roi_files = glob.glob(CACHE_PATH_TEST + name + '_*')
        if len(roi_files) == 0:
            print('ROI files absent. Skip it!')
            continue
        masks_list = []
        for f in roi_files:
            mask = load_from_file(f)
            print(mask.shape)
            masks_list.append(mask)
        masks_list = np.concatenate(masks_list, axis=0)
        masks = masks_list

        div_point = DIV_POINT_V1
        if bbox[3] > 1100:
            div_point = DIV_POINT_V2

        print('Initial frames: {}'.format(masks.shape[0]))
        full_length_values = []
        for i in range(masks.shape[0]):
            if masks[i].max() > div_point:
                l = get_length_from_mask(masks[i], bbox, div_point)
                # show_image(masks[i])
                full_length_values.append(l)
            else:
                full_length_values.append(0.0)

        save_in_file(full_length_values, cache_path)

    print('Complete!')


if __name__ == '__main__':
    get_length_for_train()
    get_length_for_test()
