# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Try to predict with XGBoost type of fish on given frame. Works better than just average of predictions from CNNs.
- It uses data about current frame and 3 previous and 3 next frames.
- Features for XGboost created from predictions of neural nets
- It probably would be better to rewrite code with LightGBM instead of XGBoost to increase speed
'''


from a00_common_functions import *
import xgboost as xgb
from operator import itemgetter
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, f1_score


INPUT_PATH = "../input/"

CACHE_PATH_VALID = "../cache_dense_net_train/"
CACHE_PATH_TEST = "../cache_dense_net_test/"
CACHE_PATH_VALID_2 = "../cache_resnet50_train/"
CACHE_PATH_TEST_2 = "../cache_resnet50_test/"
CACHE_PATH_VALID_3 = "../cache_inception_v3_train/"
CACHE_PATH_TEST_3 = "../cache_inception_v3_test/"
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
    outfile = open('xgb_type.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb_type.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def create_xgboost_model(train, features):
    start_time = time.time()

    num_folds = 5
    eta = 0.25
    max_depth = 2
    subsample = 0.9
    colsample_bytree = 0.9
    eval_metric = 'mlogloss'
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
        "objective": "multi:softprob",
        "num_class": 8,
        "booster": "gbtree",
        "eval_metric": eval_metric,
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": 2017,
        "nthread": 6,
        # 'gpu_id': 3,
        # 'updater': 'grow_gpu_hist',
    }
    num_boost_round = 10000
    early_stopping_rounds = 100

    print('Train shape:', train.shape)
    print('Features:', features)

    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(num_folds)
    num_fold = 0
    pred_names = []
    for i in range(8):
        train['prediction_{}'.format(i)] = -1
        pred_names.append('prediction_{}'.format(i))

    model_list = []
    for train_index, test_index in kfold_videos_split:
        num_fold += 1
        if num_fold < 5:
            continue
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
        train.loc[train['video_id'].isin(test_videos), pred_names] = preds
        model_list.append(gbm)

    for i in range(8):
        rw = (train['target'] == i).astype(np.uint8)
        roc_auc = roc_auc_score(rw, list(train[pred_names[i]].values))
        print('Predicted score ROC AUC {}: {}'.format(i, roc_auc))
    print("Time XGBoost: %s sec" % (round(time.time() - start_time, 0)))
    return model_list


def get_train_tst_tables_for_type():
    train_full = pd.read_csv(INPUT_PATH + 'training.csv')
    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(5)
    bboxes = pd.read_csv(ADD_PATH + 'bboxes_train.csv')
    roi_train_stat = pd.read_csv(ADD_PATH + "roi_stat_train.csv")
    roi_test_stat = pd.read_csv(ADD_PATH + "roi_stat_test.csv")
    pred_fish_num = 3
    next_fish_num = 3
    features = []

    num_fold = 0
    train_full.fillna(-1, inplace=True)
    train_full['target'] = 7
    for i in range(7):
        train_full.loc[train_full[FISH_TABLE[i]] == 1, 'target'] = i

    train = train_full[['row_id', 'video_id', 'frame', 'fish_number', 'target']].copy()

    feat = dict()
    for available_nets in ['densenet121', 'resnet50', 'inception_v3']:
        feat[available_nets] = []
        for i in FISH_TABLE:
            fname = i + '_' + available_nets
            train[fname] = -1
            feat[available_nets].append(fname)
        fname = 'no_fish_' + available_nets
        train[fname] = -1
        feat[available_nets].append(fname)
        features += feat[available_nets]
        for j in range(1, pred_fish_num + 1):
            tmp = (available_nets, 'prev', j)
            feat[tmp] = []
            for i in FISH_TABLE:
                fname = i + '_' + available_nets + '_prev_' + str(j)
                train[fname] = -1
                feat[tmp].append(fname)
            fname = 'no_fish_{}_prev_{}'.format(available_nets, j)
            train[fname] = -1
            feat[tmp].append(fname)
            features += feat[tmp]
        for j in range(1, next_fish_num + 1):
            tmp = (available_nets, 'next', j)
            feat[tmp] = []
            for i in FISH_TABLE:
                fname = i + '_' + available_nets + '_next_' + str(j)
                train[fname] = -1
                feat[tmp].append(fname)
            fname = 'no_fish_{}_next_{}'.format(available_nets, j)
            train[fname] = -1
            feat[tmp].append(fname)
            features += feat[tmp]


    if 1:
        # We try to add all frames to train table !!!
        train_csv_path = ADD_PATH + "fish_type_train_big.csv"
        pred_arr = dict()
        if not os.path.isfile(train_csv_path):
            overall_list = []
            for train_index, test_index in kfold_videos_split:
                num_fold += 1
                for i in test_index:
                    name = os.path.basename(videos[i])
                    print('Go for {}'.format(name))
                    prediction_path1 = CACHE_PATH_VALID + name + '_prediction.pklz'
                    pred_arr['densenet121'] = load_from_file(prediction_path1)
                    prediction_path2 = CACHE_PATH_VALID_2 + name + '_prediction.pklz'
                    pred_arr['resnet50'] = load_from_file(prediction_path2)
                    prediction_path3 = CACHE_PATH_VALID_3 + name + '_prediction.pklz'
                    pred_arr['inception_v3'] = load_from_file(prediction_path3)
                    frames = list(range(pred_arr['densenet121'].shape[0]))
                    videos_list = [name].copy() * pred_arr['densenet121'].shape[0]

                    table = [frames, videos_list]
                    df = pd.DataFrame(table)
                    df = df.transpose()
                    df.columns = ['frame', 'video_id']
                    for f in features:
                        df[f] = -1

                    frames = np.array(frames)
                    subdf = df.index
                    for available_nets in ['densenet121', 'resnet50', 'inception_v3']:
                        df.loc[subdf, feat[available_nets]] = pred_arr[available_nets].copy()
                        for j in range(1, pred_fish_num + 1):
                            frames_prev = frames - j
                            pred_arr_prev = pred_arr[available_nets][frames_prev]
                            pred_arr_prev[frames_prev < 0, :] = -1
                            df.loc[subdf, feat[(available_nets, 'prev', j)]] = pred_arr_prev.copy()
                        for j in range(1, next_fish_num + 1):
                            frames_next = frames + j
                            frames_next[frames_next >= pred_arr[available_nets].shape[0]] = -1
                            pred_arr_next = pred_arr[available_nets][frames_next]
                            pred_arr_next[frames_next < 0, :] = -1
                            df.loc[subdf, feat[(available_nets, 'next', j)]] = pred_arr_next.copy()

                    df['target'] = 7
                    sub_train = train[train['video_id'] == name]
                    frames = sub_train['frame'].values
                    targets = sub_train['target'].values
                    for j in range(len(frames)):
                        if targets[j] != 7:
                            df.loc[df['frame'] == frames[j] - 1, 'target'] = targets[j]
                            df.loc[df['frame'] == frames[j], 'target'] = targets[j]
                            df.loc[df['frame'] == frames[j] + 1, 'target'] = targets[j]
                    overall_list.append(df)
            train = pd.concat(overall_list)
            train.to_csv(train_csv_path, index=False)
        else:
            train = pd.read_csv(train_csv_path)

    test_csv_path = ADD_PATH + "fish_type_test.csv"
    if not os.path.isfile(test_csv_path):
        pred_arr = dict()
        test = pd.read_csv(INPUT_PATH + 'submission_format_zeros.csv')
        test = test[['row_id', 'frame', 'video_id']]
        bboxes = pd.read_csv('../modified_data/bboxes_test.csv')
        videos = glob.glob(INPUT_PATH + 'test_videos/*.mp4')

        for f in features:
            test[f] = -1

        for v in videos:
            name = os.path.basename(v)[:-4]
            print('Go for {}'.format(name))
            prediction_path1 = CACHE_PATH_TEST + name + '_prediction.pklz'
            pred_arr['densenet121'] = load_from_file(prediction_path1)
            prediction_path2 = CACHE_PATH_TEST_2 + name + '_prediction.pklz'
            pred_arr['resnet50'] = load_from_file(prediction_path2)
            prediction_path3 = CACHE_PATH_TEST_3 + name + '_prediction.pklz'
            pred_arr['inception_v3'] = load_from_file(prediction_path3)
            frames = list(range(pred_arr['densenet121'].shape[0]))
            subtest = (test['video_id'] == name) & (test['frame'].isin(frames))
            frames = np.array(frames)

            for available_nets in ['densenet121', 'resnet50', 'inception_v3']:
                test.loc[subtest, feat[available_nets]] = pred_arr[available_nets].copy()
                for j in range(1, pred_fish_num + 1):
                    frames_prev = frames - j
                    pred_arr_prev = pred_arr[available_nets][frames_prev]
                    pred_arr_prev[frames_prev < 0, :] = -1
                    test.loc[subtest, feat[(available_nets, 'prev', j)]] = pred_arr_prev.copy()
                for j in range(1, next_fish_num + 1):
                    frames_next = frames + j
                    frames_next[frames_next >= pred_arr[available_nets].shape[0]] = -1
                    pred_arr_next = pred_arr[available_nets][frames_next]
                    pred_arr_next[frames_next < 0, :] = -1
                    test.loc[subtest, feat[(available_nets, 'next', j)]] = pred_arr_next.copy()

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

    return train, test, features


def predict_frames_on_full_train(model_list, train, features):
    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(5)
    num_fold = 0
    fnames = []
    for f in FISH_TABLE:
        train[f] = -1
        fnames.append(f)
    train['no_fish'] = -1
    fnames.append('no_fish')

    for train_index, test_index in kfold_videos_split:
        test_videos = videos[test_index]
        print('Start fold {} from {}'.format(num_fold+1, 5))
        X_valid = train.loc[train['video_id'].isin(test_videos)].copy()
        print('Valid data:', X_valid[features].shape)

        dvalid = xgb.DMatrix(X_valid[features])
        preds = model_list[num_fold].predict(dvalid, ntree_limit=model_list[num_fold].best_iteration + 1)
        train.loc[train['video_id'].isin(test_videos), fnames] = preds
        num_fold += 1

    train[['video_id', 'frame'] + fnames].to_csv(ADD_PATH + 'fish_type_prediction_xgboost_train.csv', index=False)


def predict_frames_on_full_tst(model_list, test, features):
    fnames = []
    for f in FISH_TABLE:
        test[f] = -1
        fnames.append(f)
    fnames.append('no_fish')
    test['no_fish'] = -1

    dvalid = xgb.DMatrix(test[features])
    p = []
    for m in model_list:
        preds = m.predict(dvalid, ntree_limit=m.best_iteration + 1)
        p.append(preds)
    p = np.array(p)
    p = p.mean(axis=0)
    test[fnames] = p

    test[['video_id', 'frame'] + fnames].to_csv(ADD_PATH + 'fish_type_prediction_xgboost_test.csv', index=False)


if __name__ == '__main__':
    train, test, features = get_train_tst_tables_for_type()
    models_cache = MODELS_PATH + "xgboost.type.models.pklz"
    if 1:
        model_list = create_xgboost_model(train, features)
        save_in_file(model_list, models_cache)
    else:
        print('Restore models from cache!')
        model_list = load_from_file(models_cache)
    predict_frames_on_full_train(model_list, train, features)
    predict_frames_on_full_tst(model_list, test, features)
