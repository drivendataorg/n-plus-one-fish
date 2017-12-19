# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Try to predict with XGBoost length of fish on given frame. Works worse than default
  values obtained by heuristic algorithm. Probably need to add more features.
- It uses data about current frame and 5 previous and 5 next frames.
- Features for XGboost created from predictions of neural nets
- This predictions is not used in final pipeline
'''

from a00_common_functions import *
import xgboost as xgb
from operator import itemgetter
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error
from math import sqrt

INPUT_PATH = "../input/"

CACHE_PATH_VALID = "../cache_dense_net_train/"
CACHE_PATH_TEST = "../cache_dense_net_test/"
CACHE_PATH_VALID_2 = "../cache_resnet50_train/"
CACHE_PATH_TEST_2 = "../cache_resnet50_test/"
CACHE_LENGTH_VALID = "../cache_length_train/"
CACHE_LENGTH_TEST = "../cache_length_test/"
CACHE_ROI_VALID = "../cache_roi_train/"
CACHE_ROI_TEST = "../cache_roi_test/"
CSV_PATH_VALID = "../cache_vector_train/"
if not os.path.isdir(CSV_PATH_VALID):
    os.mkdir(CSV_PATH_VALID)
CSV_PATH_TEST = "../cache_vector_test/"
if not os.path.isdir(CSV_PATH_TEST):
    os.mkdir(CSV_PATH_TEST)
MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
ADD_PATH = '../modified_data/'


def create_feature_map(features):
    outfile = open('xgb_length.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb_length.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def create_xgboost_model(train, features):
    start_time = time.time()

    # Only left needed values!
    train = train[train['target'] != -1]

    num_folds = 5
    eta = 0.25
    max_depth = 2
    subsample = 0.9
    colsample_bytree = 0.9
    eval_metric = 'rmse'
    unique_target = np.array(sorted(train['target'].unique()))
    print('Target length: {}: {}'.format(len(unique_target), unique_target))

    log_str = 'XGBoost iter {}. FOLDS: {} METRIC: {} ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(1,
                                                                                                           num_folds,
                                                                                                           eval_metric,
                                                                                                           eta,
                                                                                                           max_depth,
                                                                                                           subsample,
                                                                                                           colsample_bytree)
    print(log_str)
    params = {
        "objective": "reg:linear",
        "booster": "gbtree",
        "eval_metric": eval_metric,
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": 2018,
        "nthread": 6,
        # 'gpu_id': 0,
        # 'updater': 'grow_gpu_hist',
    }
    num_boost_round = 10000
    early_stopping_rounds = 50

    print('Train shape:', train.shape)
    print('Features:', features)

    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(num_folds)
    num_fold = 0
    train['pred_length'] = -1


    model_list = []
    for train_index, test_index in kfold_videos_split:
        num_fold += 1
        train_videos = videos[train_index]
        test_videos = videos[test_index]
        print('Start fold {} from {}'.format(num_fold, num_folds))
        X_train = train.loc[train['video_id'].isin(train_videos)].copy()
        X_valid = train.loc[train['video_id'].isin(test_videos)].copy()
        y_train = X_train['target'].copy()
        y_valid = X_valid['target'].copy()

        print('Train data:', X_train[features].shape)
        print('Valid data:', X_valid[features].shape)
        print('Target train shape:', y_train.shape)
        print('Valid train shape:', y_valid.shape)

        dtrain = xgb.DMatrix(X_train[features], y_train)
        dvalid = xgb.DMatrix(X_valid[features], y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

        imp = get_importance(gbm, features)
        print('Importance: {}'.format(imp))

        print("Validating...")
        preds = gbm.predict(dvalid, ntree_limit=gbm.best_iteration + 1)
        print(preds.shape)
        train.loc[train['video_id'].isin(test_videos), 'pred_length'] = preds
        model_list.append(gbm)

    rmse = sqrt(mean_squared_error(list(train['target'].values), list(train['pred_length'].values)))
    print('Predicted score RMSE: {}'.format(rmse))
    print("Time XGBoost: %s sec" % (round(time.time() - start_time, 0)))
    return model_list


def get_train_tst_tables_for_length():
    train_full = pd.read_csv(INPUT_PATH + 'training.csv')
    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(5)
    bboxes_train = pd.read_csv(ADD_PATH + 'bboxes_train.csv')
    bboxes_test = pd.read_csv(ADD_PATH + 'bboxes_test.csv')
    roi_train_stat = pd.read_csv(ADD_PATH + "roi_stat_train.csv")
    roi_test_stat = pd.read_csv(ADD_PATH + "roi_stat_test.csv")
    pred_fish_num = 5
    next_fish_num = 5
    features = []

    num_fold = 0
    train_full.fillna(0, inplace=True)
    train_full['target'] = train_full['length']

    train = train_full[['row_id', 'video_id', 'frame', 'target']].copy()

    features.append('length_center')
    train[features[-1]] = 0
    for j in range(1, pred_fish_num + 1):
        features.append('length_prev_{}'.format(j))
        train[features[-1]] = 0
    for j in range(1, next_fish_num + 1):
        features.append('length_next_{}'.format(j))
        train[features[-1]] = 0

    # We try to add all frames to train table !!!
    train_csv_path = ADD_PATH + "fish_length_train_big.csv"
    if not os.path.isfile(train_csv_path):
        overall_list = []
        for train_index, test_index in kfold_videos_split:
            num_fold += 1
            for i in test_index:
                name = os.path.basename(videos[i])
                print('Go for {}'.format(name))
                length_path = CACHE_LENGTH_VALID + name + '_length.pklz'
                length_arr = np.array(load_from_file(length_path))
                frames = list(range(length_arr.shape[0]))
                videos_list = [name].copy() * length_arr.shape[0]

                table = [frames, videos_list]
                df = pd.DataFrame(table)
                df = df.transpose()
                df.columns = ['frame', 'video_id']
                for f in features:
                    df[f] = -1

                frames = np.array(frames)
                subdf = df.index
                df.loc[subdf, 'length_center'] = length_arr.copy()
                for j in range(1, pred_fish_num + 1):
                    frames_prev = frames - j
                    pred_arr_prev = length_arr[frames_prev]
                    pred_arr_prev[frames_prev < 0] = -1
                    df.loc[subdf, 'length_prev_{}'.format(j)] = pred_arr_prev.copy()
                for j in range(1, next_fish_num + 1):
                    frames_next = frames + j
                    frames_next[frames_next >= length_arr.shape[0]] = -1
                    pred_arr_next = length_arr[frames_next]
                    pred_arr_next[frames_next < 0] = -1
                    df.loc[subdf, 'length_next_{}'.format(j)] = pred_arr_next.copy()

                df['target'] = -1
                sub_train = train[train['video_id'] == name]
                frames = sub_train['frame'].values
                targets = sub_train['target'].values
                for j in range(len(frames)):
                    if targets[j] != 0:
                        df.loc[df['frame'] == frames[j] - 1, 'target'] = targets[j]
                        df.loc[df['frame'] == frames[j], 'target'] = targets[j]
                        df.loc[df['frame'] == frames[j] + 1, 'target'] = targets[j]
                overall_list.append(df)
        train = pd.concat(overall_list)
        train.to_csv(train_csv_path, index=False)
    else:
        train = pd.read_csv(train_csv_path)

    test_csv_path = ADD_PATH + "fish_length_test.csv"
    if not os.path.isfile(test_csv_path):
        test = pd.read_csv(INPUT_PATH + 'submission_format_zeros.csv')
        test = test[['row_id', 'frame', 'video_id']]
        videos = glob.glob(INPUT_PATH + 'test_videos/*.mp4')

        for f in features:
            test[f] = 0

        for v in videos:
            name = os.path.basename(v)[:-4]
            print('Go for {}'.format(name))
            length_path = CACHE_LENGTH_TEST + name + '_length.pklz'
            length_arr = np.array(load_from_file(length_path))
            frames = list(range(length_arr.shape[0]))
            subtest = (test['video_id'] == name) & (test['frame'].isin(frames))
            frames = np.array(frames)

            test.loc[subtest, 'length_center'] = length_arr.copy()
            for j in range(1, pred_fish_num + 1):
                frames_prev = frames - j
                pred_arr_prev = length_arr[frames_prev]
                pred_arr_prev[frames_prev < 0] = -1
                test.loc[subtest, 'length_prev_{}'.format(j)] = pred_arr_prev.copy()
            for j in range(1, next_fish_num + 1):
                frames_next = frames + j
                frames_next[frames_next >= length_arr.shape[0]] = -1
                pred_arr_next = length_arr[frames_next]
                pred_arr_next[frames_next < 0] = -1
                test.loc[subtest, 'length_next_{}'.format(j)] = pred_arr_next.copy()

        test.to_csv(test_csv_path, index=False)
    else:
        test = pd.read_csv(test_csv_path)

    # ROI stat add
    print('Merge with ROI data...')
    print(len(train))
    train = pd.merge(train, roi_train_stat, on=['frame', 'video_id'], how='left')
    test = pd.merge(test, roi_test_stat, on=['frame', 'video_id'], how='left')
    print(len(train))
    features += ['masks_mx', 'masks_mean', 'masks_bbox_mx', 'masks_bbox_mean']

    # BBox data
    print('Merge with BBox data...')
    bboxes_train.rename(columns={'id': 'video_id'}, inplace=True)
    bboxes_test.rename(columns={'id': 'video_id'}, inplace=True)
    train = pd.merge(train, bboxes_train, on=['video_id'], how='left')
    test = pd.merge(test, bboxes_test, on=['video_id'], how='left')
    features += ['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']

    # More data from classifiers
    print('Merge data from classifiers')
    train_type = pd.read_csv(ADD_PATH + "fish_type_train_big.csv")
    col_to_use = list(train_type.columns.values)
    col_to_use.remove('target')
    train = pd.merge(train, train_type[col_to_use], on=['frame', 'video_id'], how='left')
    test_type = pd.read_csv("../modified_data/fish_type_test.csv")
    test = pd.merge(test, test_type[col_to_use], on=['frame', 'video_id'], how='left')
    col_to_use.remove('frame')
    col_to_use.remove('video_id')
    features += col_to_use

    return train, test, features


def predict_frames_on_full_train(model_list, train, features):
    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(5)
    num_fold = 0

    fnames = []
    train['length_pred'] = -1
    fnames.append('length_pred')

    for train_index, test_index in kfold_videos_split:
        test_videos = videos[test_index]
        print('Start fold {} from {}'.format(num_fold+1, 5))
        X_valid = train.loc[train['video_id'].isin(test_videos)].copy()
        print('Valid data:', X_valid[features].shape)

        dvalid = xgb.DMatrix(X_valid[features])
        preds = model_list[num_fold].predict(dvalid, ntree_limit=model_list[num_fold].best_iteration + 1)
        train.loc[train['video_id'].isin(test_videos), fnames] = preds
        num_fold += 1

    train.loc[train['length_pred'] < 0, 'length_pred'] = 0
    train[['video_id', 'frame'] + fnames].to_csv(ADD_PATH + 'fish_length_prediction_xgboost_train.csv', index=False)


def predict_frames_on_full_tst(model_list, test, features):
    fnames = []
    test['length_pred'] = -1
    fnames.append('length_pred')

    dvalid = xgb.DMatrix(test[features])
    p = []
    for m in model_list:
        preds = m.predict(dvalid, ntree_limit=m.best_iteration + 1)
        p.append(preds)
    p = np.array(p)
    p = p.mean(axis=0)
    test[fnames] = p

    test.loc[test['length_pred'] < 0, 'length_pred'] = 0
    test[['video_id', 'frame'] + fnames].to_csv(ADD_PATH + 'fish_length_prediction_xgboost_test.csv', index=False)


if __name__ == '__main__':
    train, test, features = get_train_tst_tables_for_length()
    models_cache = MODELS_PATH + "xgboost.length.models.pklz"
    if 1:
        model_list = create_xgboost_model(train, features)
        save_in_file(model_list, models_cache)
    else:
        print('Restore models from cache!')
        model_list = load_from_file(models_cache)
    predict_frames_on_full_train(model_list, train, features)
    predict_frames_on_full_tst(model_list, test, features)
