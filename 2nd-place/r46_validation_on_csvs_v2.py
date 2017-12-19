'''
- Validation v2. Based on code provided on forum
- Doesn't work as expected. Always predicts more optimistic values. Probably because of non-random train/test split.
'''

import editdistance
import numpy as np
import os
import glob
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score

INPUT_PATH = "../input/"
CSV_PATH_TRAIN = "../cache_csv_train/"
SUBM_PATH = '../subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)

# defined for the n+1 fish, n+2 fish competition
COLUMNS = ['frame',
           'video_id',
           'fish_number',
           'length',
           'species_fourspot',
           'species_grey sole',
           'species_other',
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']

SPECIES = COLUMNS[4:]
SPECIES_COL_IDX = np.arange(4, len(COLUMNS))

VIDEO_ID_IDX = 1
FISH_NUMBER_IDX = 2
LENGTH_IDX = 3


def multi_multi_log_loss(predicted, actual, class_column_indices, eps=1e-15):
    """Multi class, multi-label version of Logarithmic Loss metric.
    :param predicted: a 2d numpy array of the predictions that are probabilities [0, 1]
    :param actual: a 2d numpy array of the same shape as your predictions. 1 for the actual labels, 0 elsewhere
    :return: The multi-multi log loss score for this set of predictions
    """
    class_scores = np.ones(len(class_column_indices), dtype=np.float64)

    # calculate log loss for each set of columns that belong to a class:
    for k, this_class_indices in enumerate(class_column_indices):
        # get just the columns for this class
        preds_k = predicted[:, this_class_indices]

        # normalize so probabilities sum to one (unless sum is zero, then we clip)
        preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.inf)

        actual_k = actual[:, this_class_indices]

        # shrink predictions
        y_hats = np.clip(preds_k, eps, 1 - eps)
        sum_logs = np.sum(actual_k * np.log(y_hats))
        class_scores[k] = (-1.0 / actual.shape[0]) * sum_logs

    return np.average(class_scores)


def weighted_rmsle(predicted, actual, weights=None):
        """ Calculates RMSLE weighted by a vector of weights.
        :param predicted: the predictions
        :param actual: the actual true data
        :param weights: how "important" each column is (if None, assume equal)
        :return: WRMSLE
        """
        # force floats
        predicted = predicted.astype(np.float64)
        actual = actual.astype(np.float64)

        # if no weights, assume equal weighting
        if weights is None:
            weights = np.ones(predicted.shape[1], dtype=np.float64)

        # reshape as a column matrix
        weights = weights.reshape(-1, 1).astype(np.float64)

        # make sure that there are the right number of weights
        if weights.shape[0] != predicted.shape[1]:
            error_msg = "Weight matrix {} must have same number of entries as columns in predicted ({})."
            raise Exception(error_msg.format(weights.shape, predicted.shape[1]))

        # calculate weighted scores
        predicted_score = predicted.dot(weights)
        actual_score = actual.dot(weights)

        # calculate log error
        log_errors = np.log(predicted_score + 1) - np.log(actual_score + 1)

        # return RMSLE
        return np.sqrt((log_errors ** 2).mean())


def adjusted_mean_absolute_percent_error(predicted, actual, error_weights):
    """Calculates the mean absolute percent error.
    :param predicted: The predicted values.
    :param actual: The actual values.
    :param error_weights: Available as `e_n` and as a standalone file for nests
        in the competition materials.
    """
    not_nan_mask = ~np.isnan(actual)

    # calculate absolute error
    abs_error = (np.abs(actual[not_nan_mask] - predicted[not_nan_mask]))

    # calculate the percent error (replacing 0 with 1
    # in order to avoid divide-by-zero errors).
    pct_error = abs_error / np.maximum(1, actual[not_nan_mask])

    # adjust error by count accuracies
    adj_error = pct_error / error_weights[not_nan_mask]

    # return the mean as a percentage
    return np.mean(adj_error)


def fish_metric(actual_all_vid, predicted_all_vid, a_l=0.1, a_n=0.6, a_s=0.3, species_prefix='species_'):
    """ Reference implementation for the N+1 fish, N+2 fish competition
        evaluation metric.
        Implemented in pure numpy for performance gains over pandas.
    """

    def get_fish_order(fish_numbers, species_probs):
        """ Gets a sequence of fish from the ordering of fish numbers
            and the species probabilities
        """
        sequence = []

        unique_fish = np.unique(fish_numbers[~np.isnan(fish_numbers)])

        for fishy in unique_fish:
            mask = (fish_numbers == fishy)
            this_fish = species_probs[mask, :]

            col_maxes = np.nanmax(this_fish, axis=0)

            species = SPECIES[np.argmax(col_maxes)]
            sequence.append(species)

        return sequence

    def levenfish(act_fish_numbers, act_species, pred_fish_numbers, pred_species):
        """ Edit distance for a sequence of fishes in the competition
            submission format.
        """
        # print(act_species)
        # print(pred_species)
        s1 = get_fish_order(act_fish_numbers, act_species)
        s2 = get_fish_order(pred_fish_numbers, pred_species)
        # print(s1, s2)
        return editdistance.eval(s1, s2)

    video_ids = actual_all_vid[:, VIDEO_ID_IDX].ravel()

    actual_fish_numbers = actual_all_vid[:, FISH_NUMBER_IDX].astype(np.float64)
    pred_fish_numbers = predicted_all_vid[:, FISH_NUMBER_IDX].astype(np.float64)

    actual_lengths = actual_all_vid[:, LENGTH_IDX].astype(np.float64)
    pred_lengths = predicted_all_vid[:, LENGTH_IDX].astype(np.float64)

    actual_species = actual_all_vid[:, SPECIES_COL_IDX].astype(np.float64)
    pred_species = predicted_all_vid[:, SPECIES_COL_IDX].astype(np.float64)

    uniq_video_ids = np.unique(video_ids)
    per_video_scores = np.zeros_like(uniq_video_ids, dtype=np.float64)

    # uniq_video_ids = ['0EmM5wsVVNqaKNaM']
    for ix, vid_idx in enumerate(uniq_video_ids):
        print('Video:', vid_idx)
        this_vid_mask = (video_ids == vid_idx)

        # edit distance scoring
        n_fish = np.nanmax(actual_fish_numbers[this_vid_mask])
        # print('NFish: ', n_fish)

        actual_fn = actual_fish_numbers[this_vid_mask]
        pred_fn = pred_fish_numbers[this_vid_mask]

        actual_spec = actual_species[this_vid_mask, :]
        pred_spec = pred_species[this_vid_mask, :]

        edit_error = 1 - (levenfish(actual_fn, actual_spec, pred_fn, pred_spec) / n_fish)
        edit_error = np.clip(edit_error, 0, 1)
        edit_component = a_n * edit_error

        # only test length and species against frames where we
        # have actual fish labeled
        annotated_frames = ~np.isnan(actual_fn)

        # species identification scoring
        def _auc(a, p):
            try:
                return roc_auc_score(a, p)
            except ValueError:
                mae = np.mean(np.abs(a - p))
                return ((1 - mae) / 2) + 0.5

        aucs = [_auc(actual_spec[annotated_frames, c],
                     pred_spec[annotated_frames, c])
                for c in range(actual_species.shape[1])]

        # normalize to 0-1
        species_auc = 2 * (np.mean(aucs) - 0.5)
        species_auc = np.clip(species_auc, 0, 1)
        species_component = a_s * species_auc

        # we have "no-fish" annotations where all of the species are zero
        # these are only relevant for the species classification task. We'll
        # ignore these for the length task.
        only_fish_annotations = (np.nan_to_num(actual_species.sum(axis=1)) > 0) & this_vid_mask

        # length scoring
        length_r2 = r2_score(actual_lengths[only_fish_annotations],
                             pred_lengths[only_fish_annotations])

        length_r2 = np.clip(length_r2, 0, 1)
        length_component = a_l * length_r2

        per_video_scores[ix] = length_component + edit_component + species_component
        print("Edit score: {:.3f} AUC score: {:.3f} Length score: {:.3f}".format(edit_error, species_auc, length_r2))
        print('Score: {:.3f}'.format(per_video_scores[ix]))

    return np.mean(per_video_scores)


def merge_train_csv_in_single_file(out_file):
    real = pd.read_csv(INPUT_PATH + 'training.csv')
    files = glob.glob(CSV_PATH_TRAIN + '*.csv')
    s = []
    for video_id in list(real['video_id'].unique()):
        f = CSV_PATH_TRAIN + video_id + '.csv'
        print('Go for {}'.format(f))
        tbl = pd.read_csv(f)
        video_id = os.path.basename(f)[:-4]
        tbl['video_id'] = video_id
        frames = list(real.loc[real['video_id'] == video_id, 'frame'].values)
        tbl = tbl[tbl['frame'].isin(frames)].copy()
        if len(frames) != len(tbl):
            print('Problem: {} != {}'.format(len(frames), len(tbl)))
            exit()
        s.append(tbl)
    subm = pd.concat(s)
    subm.rename(columns={"species_grey_sole": "species_grey sole"}, inplace=True)
    subm.to_csv(out_file, index=False)


def get_score():
    out_file = SUBM_PATH + 'valid1.csv'
    if not os.path.isfile(out_file) or 1:
        merge_train_csv_in_single_file(SUBM_PATH + 'valid1.csv')
    pred = pd.read_csv(out_file)
    real = pd.read_csv(INPUT_PATH + 'training.csv')

    # Validate only on subset of videos
    if 0:
        boat_ids = pd.read_csv('../modified_data/boat_ids_train.csv')
        needed_ids = list(boat_ids[boat_ids['boat_id'] == 4]['name'].values)
        pred = pred[pred['video_id'].isin(needed_ids)].copy()
        real = real[real['video_id'].isin(needed_ids)].copy()

    real.rename(columns={"species_grey_sole": "species_grey sole"}, inplace=True)
    print('Length real: {}'.format(len(real)))
    print('Length pred: {}'.format(len(pred)))
    score = fish_metric(real[COLUMNS].as_matrix(), pred[COLUMNS].as_matrix())
    print('Score: {}'.format(score))


if __name__ == '__main__':
    get_score()
