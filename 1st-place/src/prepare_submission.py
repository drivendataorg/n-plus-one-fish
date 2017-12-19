import numpy as np
import pandas as pd
import argparse
import math
import os
import os.path
import pickle
import matplotlib.pyplot as plt
import sys

import dataset
import utils
import fish_detection
from typing import List
from numpy.testing import assert_array_equal

MAX_ROWS = 10000

SPECIES_COLS = ['species_' + species for species in dataset.SPECIES]
COVER_COLS = ['no fish', 'hand over fish', 'fish clear']
CLS_COLS = SPECIES_COLS + COVER_COLS


def load_key_frames(sequence_res_dir='../output/sequence_results_test'):
    predictions = np.load(os.path.join(sequence_res_dir, 'key_fish_prob.npy'))
    video_ids = np.load(os.path.join(sequence_res_dir, 'key_fish_ids.npy'))

    steps_before = 8
    steps_after = 8
    peak_threashold = 0.16

    res = {}
    for i, video_id in enumerate(video_ids):
        # print(video_id)
        res[video_id] = []

        items = predictions[0, i, :].copy()
        while True:
            top_idx = np.argmax(items)
            if items[top_idx] < 0.05:
                break

            step_from = max(0, top_idx-steps_before)
            step_to = top_idx + steps_after

            peak_area = np.sum(items[step_from:step_to])
            if peak_area > peak_threashold:
                res[video_id].append(top_idx)

            items[step_from:step_to] = 0

        # plt.plot(predictions[0, i, :])
        # plt.scatter(x=res[video_id], y=[0.0]*len(res[video_id]))
        # plt.show()

    return res


def key_frames_to_frame_numbers(key_frames: List[int], res_size):
    res = np.ones((res_size,), dtype=np.int) * len(key_frames)

    for fish_num in range(len(key_frames)-1):

        to_idx = int(round(key_frames[fish_num] * 0.5 + key_frames[fish_num+1] * 0.5))

        if fish_num == 0:
            res[:to_idx] = fish_num+1
        else:
            from_idx = int(round(key_frames[fish_num - 1] * 0.5 + key_frames[fish_num] * 0.5))
            res[from_idx:to_idx] = fish_num+1
    return res


def test_key_frames_to_frame_numbers():
    assert_array_equal(key_frames_to_frame_numbers([], res_size=4), np.array([0, 0, 0, 0]))
    assert_array_equal(key_frames_to_frame_numbers([1], res_size=4), np.array([1, 1, 1, 1]))
    assert_array_equal(key_frames_to_frame_numbers([0, 3], res_size=4), np.array([1, 1, 2, 2]))
    assert_array_equal(key_frames_to_frame_numbers([0, 4, 8], res_size=10), np.array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3]))


FRAME_IDX = 1
VIDEO_ID_IDX = 2
FISH_NUMBER_IDX = 3
LENGTH_IDX = 4
SPECIES_START_IDX = 5


def combine_group_species(orig_submission_array: np.ndarray):
    fish_num = orig_submission_array[0, FISH_NUMBER_IDX]
    fish_start_row = 0
    for res_row in range(orig_submission_array.shape[0]):
        row = orig_submission_array[res_row]

        if row[FISH_NUMBER_IDX] != fish_num:
            fish_end_row = res_row

            fish_rows = orig_submission_array[fish_start_row:fish_end_row, :]
            fish_prob = np.sum(fish_rows[:, SPECIES_START_IDX:], axis=1)
            max_prob_fish = np.max(fish_prob)

            fish_prob_clip = fish_prob.copy()
            fish_prob_clip[fish_prob_clip < max_prob_fish * 0.75] = 0.0
            fish_rows_weights = fish_prob_clip ** 2
            fish_rows_weights /= np.sum(fish_rows_weights)

            species = np.average(fish_rows[:, SPECIES_START_IDX:],
                                 weights=fish_rows_weights,
                                 axis=0)
            species /= np.sum(species)
            updated_probs = np.outer(fish_prob, species)
            orig_submission_array[fish_start_row:fish_end_row, SPECIES_START_IDX:] = updated_probs

            fish_start_row = res_row
            fish_num = row[FISH_NUMBER_IDX]
    return orig_submission_array



def prepare_submission():
    orig_submission = pd.read_csv('../input/submission_format_zeros.csv')
    key_frames = load_key_frames()
    detection_ds = fish_detection.FishDetectionDataset(is_test=True)

    print('load detections:')
    try:
        detections, classifications, fish_numbers = utils.load_data('../output/cache_submission_det_csl.pkl')
        raise FileNotFoundError
    except FileNotFoundError:
        detections = {}
        for video_id in orig_submission.video_id.unique():
            df = pd.read_csv('../output/detection_results_test/resnet_53/{}_ssd_detection.csv'.format(video_id))
            df_full = pd.DataFrame({'frame': range(MAX_ROWS)})
            detection_res = df_full.merge(df, on='frame', how='left')
            detections[video_id] = detection_res['w'].fillna(0.0).as_matrix()

        print('load classifications:')
        classifications = {}

        cls_models = [
            ('inception', 1.0)
        ]

        fn_mask = '../output/classification_results_test_combined/{}/{}/{}_categories.csv'
        for video_id in orig_submission.video_id.unique():
            cls_res = np.zeros((MAX_ROWS, len(CLS_COLS)), dtype=np.float32)
            for det_id, det_weight in [('resnet_53', 0.5), ('resnet_62', 0.5)]:
                for cls_model_name, cls_model_weight in cls_models:
                    df = pd.read_csv(fn_mask.format(det_id, cls_model_name, video_id))
                    df_full = pd.DataFrame({'frame': range(MAX_ROWS)})
                    cls_res += df_full.merge(df, on='frame', how='left').fillna(0.0).as_matrix(columns=CLS_COLS) * det_weight * cls_model_weight
            classifications[video_id] = cls_res

        print('load fish numbers:')
        fish_numbers = {}
        for video_id in orig_submission.video_id.unique():
            fish_numbers[video_id] = key_frames_to_frame_numbers(sorted(key_frames[video_id]), res_size=MAX_ROWS)

        utils.save_data([detections, classifications, fish_numbers], '../output/cache_submission_det_csl.pkl')

    print('generate transforms')
    transforms = {}
    for video_id in orig_submission.video_id.unique():
        transforms[video_id] = detection_ds.transform_for_clip(video_id)

    orig_submission_array = orig_submission.as_matrix()
    for res_row in range(orig_submission_array.shape[0]):
        row = orig_submission_array[res_row]

        video_id = row[VIDEO_ID_IDX]
        frame = row[FRAME_IDX]

        if res_row % 1000 == 0:
            print(res_row, res_row*100.0 / orig_submission_array.shape[0])

        orig_submission_array[res_row, FISH_NUMBER_IDX] = fish_numbers[video_id][frame]
        w = detections[video_id][frame]
        vector_global = transforms[video_id](np.array([[0, 0], [w, 0]]))
        length = np.linalg.norm(vector_global[0] - vector_global[1])
        orig_submission_array[res_row, LENGTH_IDX] = length

        cls = classifications[video_id][frame]
        clear_conf = cls[-1]
        orig_submission_array[res_row, SPECIES_START_IDX:] = cls[:len(SPECIES_COLS)] * (
            0.5 + 0.5 * clear_conf)

    orig_submission_array = combine_group_species(orig_submission_array)
    orig_submission['fish_number'] = orig_submission_array[:, FISH_NUMBER_IDX].astype(np.float32)
    orig_submission['length'] = orig_submission_array[:, LENGTH_IDX].astype(np.float32)
    for species_idx, species in enumerate(SPECIES_COLS):
        orig_submission[species] = orig_submission_array[:, SPECIES_START_IDX+species_idx].astype(np.float32)

    orig_submission.to_csv('../output/submission30_inception_only.csv', index=False, float_format='%.8f')


res_cols = ['fish_number', 'length',
            'species_fourspot', 'species_grey sole', 'species_other',
            'species_plaice', 'species_summer', 'species_windowpane', 'species_winter']


def check_corr(sub1, sub2):
    print(sub1, sub2)
    s1 = pd.read_csv('../output/' + sub1)
    s2 = pd.read_csv('../output/' + sub2)
    for col in res_cols:
        print(col, s1[col].corr(s2[col]))

    print('mean ', sub1, sub2, 'sub2-sub1')
    for col in res_cols:
        print('{:20}  {:.6} {:.6} {:.6}'.format(col, s1[col].mean(), s2[col].mean(), s2[col].mean() - s1[col].mean()))


if __name__ == '__main__':
    # check_corr('submission{}.csv'.format(sys.argv[1]), 'submission{}.csv'.format(sys.argv[2]))
    prepare_submission()

