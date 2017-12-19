# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Merge independent CSVs for each video in single submission CSV
'''

from a00_common_functions import *

INPUT_PATH = "../input/"
CSV_PATH_TEST = "../cache_csv_test/"
SUBM_PATH = '../subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)


def merge_test_csv_in_submission(subm_path):
    subm = pd.read_csv(INPUT_PATH + 'submission_format_zeros.csv')
    subm = subm[['row_id', 'frame', 'video_id']]
    len1 = len(subm)
    uvideos = subm['video_id'].unique()
    # uvideos = ['P3QkoeOjxoM6pDKb']
    full = None
    for u in uvideos:
        print('Go for {}'.format(u))
        path = CSV_PATH_TEST + u + '.csv'
        dt = pd.read_csv(path)
        dt['video_id'] = u
        if full is None:
            full = dt.copy()
        else:
            full = full.append(dt, ignore_index=True)
    subm = pd.merge(subm, full, how='left', on=['video_id', 'frame'], left_index=True)
    len2 = len(subm)
    if len1 != len2:
        print('Some error here!')
        exit()
    subm.reset_index(drop=True, inplace=True)

    if 1:
        # Fix errors with framerate in sample submission (
        # subm = subm.loc[subm['video_id'] == 'P3QkoeOjxoM6pDKb'].copy()
        failed_ids = subm.loc[subm['length'].isnull()].copy()['video_id'].unique()
        print('Failed ids: {}'.format(failed_ids))
        cols = list(subm.columns.values)
        cols.remove('frame')
        cols.remove('row_id')
        for f in failed_ids:
            dt = subm.loc[((subm['video_id'] == f) & (~subm['length'].isnull()))][-1:].copy()
            print(dt)
            subm.loc[((subm['video_id'] == f) & (subm['length'].isnull())), cols] = dt[cols].values

    subm.rename(columns={"species_grey_sole": "species_grey sole"}, inplace=True)
    subm.to_csv(subm_path, index=False)


if __name__ == '__main__':
    subm_path = SUBM_PATH + 'subm.csv'
    merge_test_csv_in_submission(subm_path)
