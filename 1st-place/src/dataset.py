import numpy as np
import pandas as pd
from typing import List
import pickle
import os

VIDEOS_DIR = '../input/train_videos'
IMAGES_DIR = '../input/train_videos/img'
MASKS_DIR = '../output/ruler_masks'
AVG_MASKS_DIR = '../output/ruler_masks_avg'
AVG_MASKS_DIR_TEST = '../output/ruler_masks_avg_test'
RULER_CROPS_DIR = '../output/ruler_crops'
RULER_CROPS_DIR_TEST = '../output/ruler_crops_test'

IMAGES_DIR_TEST = '../input/test_videos/img'
MASKS_DIR_TEST = '../output/ruler_masks_test'

SPECIES = ['fourspot', 'grey sole', 'other', 'plaice', 'summer', 'windowpane', 'winter']
CLASSES = ['_'] + SPECIES
NB_FOLDS = 4

ASPECT_RATIO_TABLE = {
    # '_': 0.1,
    'fourspot': 0.55,
    'grey sole': 0.55,
    'other': 0.4,
    'plaice': 0.55,
    'summer': 0.5,
    'windowpane': 0.55,
    'winter': 0.5
}


def video_clips(is_test=False) -> {str}:
    if is_test:
        return video_clips_test()

    cache_fn = '../output/video_clips.pkl'
    try:
        clips = pickle.load(open(cache_fn, 'rb'))
    except FileNotFoundError:
        clips = {}
        for dir_name in os.listdir(IMAGES_DIR):
            clip_dir = os.path.join(IMAGES_DIR, dir_name)
            frames = []
            for frame_name in os.listdir(clip_dir):
                if not frame_name.endswith('.jpg'):
                    continue
                frames.append(frame_name[:-len('.jpg')])
            clips[dir_name] = frames

        pickle.dump(clips, open(cache_fn, 'wb'))
    return clips


def video_clips_test() -> {str}:
    cache_fn = '../output/video_clips_test.pkl'
    try:
        clips = pickle.load(open(cache_fn, 'rb'))
    except FileNotFoundError:
        clips = {}
        for dir_name in os.listdir(IMAGES_DIR_TEST):
            clip_dir = os.path.join(IMAGES_DIR_TEST, dir_name)
            frames = []
            for frame_name in os.listdir(clip_dir):
                if not frame_name.endswith('.jpg'):
                    continue
                frames.append(frame_name[:-len('.jpg')])
            clips[dir_name] = frames

        pickle.dump(clips, open(cache_fn, 'wb'))
    return clips


def fold_test_video_ids(fold: int) -> List[str]:
    all_video_ids = sorted(video_clips().keys())

    if fold == 0:
        return []
    return all_video_ids[(fold-1)::NB_FOLDS]


def image_fn(video_id, frame, is_test=False):
    if is_test:
        return '{}/{}/{:04}.jpg'.format(IMAGES_DIR_TEST, video_id, int(frame)+1)
    else:
        return '{}/{}/{:04}.jpg'.format(IMAGES_DIR, video_id, int(frame) + 1)


def image_crop_fn(video_id, frame, is_test=False):
    if is_test:
        return '{}/{}/{:04}.jpg'.format(RULER_CROPS_DIR_TEST, video_id, int(frame)+1)
    else:
        return '{}/{}/{:04}.jpg'.format(RULER_CROPS_DIR, video_id, int(frame) + 1)
