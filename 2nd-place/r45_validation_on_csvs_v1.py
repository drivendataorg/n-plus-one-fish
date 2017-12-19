# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Validation v1. Try to predict score
- Doesn't work as expected. Always predicts more optimistic values. Probably because of non-random train/test split.
'''

import editdistance
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from a00_common_functions import *


INPUT_PATH = "../input/"
CACHE_PATH_VALID = "../cache_csv_train/"
CACHE_PATH_TEST = "../cache_csv_test/"


def get_sequence_score_v1_incorrect(true, pred):
    true_seq = true[FISH_TABLE].apply(np.argmax, axis=1).tolist()
    frame_loc = true['frame'].tolist()
    max_prob_fish = np.array(pred[FISH_TABLE].apply(np.argmax, axis=1).tolist())
    pred_seq = max_prob_fish[frame_loc]
    edist = editdistance.eval(true_seq, pred_seq)
    edist = min(edist / len(true_seq), 1)
    score = 1 - edist
    return score


def get_sequence_score(true, pred):
    true_seq = true[FISH_TABLE].apply(np.argmax, axis=1).tolist()

    pred['fs'] = pred[FISH_TABLE].apply(np.argmax, axis=1).tolist()
    pred_seq = pred[['fs', 'fish_number']].groupby('fish_number')
    d = pred_seq.agg(lambda x: x.value_counts().index[0])
    pred_seq = d['fs'].tolist()

    edist = editdistance.eval(true_seq, pred_seq)
    edist = min(edist / len(true_seq), 1)
    score = 1 - edist
    return score


def get_auc_score(true, pred):
    score = []
    for spc in FISH_TABLE:
        try:
            sc = roc_auc_score(true[spc], pred[spc])
            score.append(sc)
        except:
            continue
    if len(score) > 0:
        score = 2 * np.mean(score) - 1
    else:
        score = 1 - np.sum([mean_absolute_error(true[spc], pred[spc]) for spc in FISH_TABLE])
    return max(0, score)


def get_length_score(true, pred):
    fish = true['length'] > 0
    score = r2_score(true['length'][fish], pred['length'][fish])
    return max(score, 0)


def find_score_for_name(train, video, video_id):
    real = train[train['video_id'] == video_id].copy()
    frames_to_extract = list(real['frame'].values)
    sequence_score = get_sequence_score(real, video)

    pred = video[video['frame'].isin(frames_to_extract)]
    if len(real) != len(pred):
        print('Some problem here: {}'.format(video_id))
        exit()
    real.reset_index(drop=True, inplace=True)
    pred.reset_index(drop=True, inplace=True)

    auc_score = get_auc_score(real, pred)
    length_score = get_length_score(real, pred)
    print("Edit score: {:.3f} AUC score: {:.3f} Length score: {:.3f}".format(sequence_score, auc_score, length_score))
    weighted_score = 0.6 * sequence_score + 0.3 * auc_score + 0.1 * length_score
    print("Score:", weighted_score)
    if weighted_score < 0.5:
        print('Low score for video: {}'.format(video_id))
        # exit()

    return weighted_score


def get_score_for_validation():
    train = pd.read_csv(INPUT_PATH + 'training.csv')
    files = glob.glob(CACHE_PATH_VALID + '*.csv')
    # files = [CACHE_PATH_VALID + '0Vn7LRp72VjFggGy.csv']
    avg_score = 0
    total = 0
    for f in files:
        video_id = os.path.basename(f)[:-4]
        video = pd.read_csv(f)
        if len(video) == 0:
            continue
        score = find_score_for_name(train, video, video_id)
        avg_score += score
        total += 1
    avg_score /= total
    print('Final score: {}'.format(round(avg_score, 6)))


if __name__ == '__main__':
    get_score_for_validation()
