# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Use all data to create CSV with predictions for each train video and each test video
- There are 2 different methods, chosen based on boat ID.
'''

from a00_common_functions import *

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
CSV_PATH_VALID = "../cache_csv_train/"
if not os.path.isdir(CSV_PATH_VALID):
    os.mkdir(CSV_PATH_VALID)
CSV_PATH_TEST = "../cache_csv_test/"
if not os.path.isdir(CSV_PATH_TEST):
    os.mkdir(CSV_PATH_TEST)
ADD_PATH = '../modified_data/'
ADD_PATH_MASKS = '../modified_data/tmp_msk/'
if not os.path.isdir(ADD_PATH_MASKS):
    os.mkdir(ADD_PATH_MASKS)


def process_single_video_from_predictions_type_1(type, name, bboxes, boat_id, vector_predictions, fish_type_predictions, length_predictions):
    if type == 'train':
        prediction_path1 = CACHE_PATH_VALID + name + '_prediction.pklz'
        prediction_path2 = CACHE_PATH_VALID_2 + name + '_prediction.pklz'
        prediction_path3 = CACHE_PATH_VALID_3 + name + '_prediction.pklz'
        length_path = CACHE_LENGTH_VALID + name + '_length.pklz'
        roi_files = glob.glob(CACHE_ROI_VALID + name + '_*')
    else:
        prediction_path1 = CACHE_PATH_TEST + name + '_prediction.pklz'
        prediction_path2 = CACHE_PATH_TEST_2 + name + '_prediction.pklz'
        prediction_path3 = CACHE_PATH_TEST_3 + name + '_prediction.pklz'
        length_path = CACHE_LENGTH_TEST + name + '_length.pklz'
        roi_files = glob.glob(CACHE_ROI_TEST + name + '_*')

    thresh = 0.5

    pred_arr = load_from_file(prediction_path1)
    length_arr = np.array(load_from_file(length_path))

    if 1:
        cache_mask_mx = ADD_PATH_MASKS + name + '.pklz'
        if not os.path.isfile(cache_mask_mx):
            masks_list = []
            for f in roi_files:
                mask = load_from_file(f)
                masks_list.append(mask)
            masks_list = np.concatenate(masks_list, axis=0)
            masks = masks_list
            masks_mx = masks.max(axis=(1, 2))
            save_in_file(masks_mx, cache_mask_mx)
        else:
            masks_mx = load_from_file(cache_mask_mx)

    # If predict for test
    if len(pred_arr.shape) == 3:
        pred_arr = pred_arr.mean(axis=0)

    frames_arr = []
    prob_arr = pred_arr[:, :7].copy()
    vector = []
    for i in range(pred_arr.shape[0]):
        frames_arr.append(str(i))
        if pred_arr[i][7] < thresh:
            # print('Frame {} Fish exists'.format(i))
            vector.append(1)
        else:
            # print('Frame {} No fish'.format(i))
            vector.append(0)

        # Fixes with ROI!
        if vector[-1] == 0:
            if masks_mx[i] >= 255:
                vector[-1] = 1
                # print('Fixed 0 -> 1')

        if vector[-1] == 1:
            if masks_mx[i] <= 20:
                vector[-1] = 0
                # print('Fixed 1 -> 0')

    vector = np.array(vector)

    # Fill 1 from beginning
    start = 0
    for i in range(len(vector)):
        if vector[i] == 1:
            start = i+1
            break
        vector[i] = 1

    end = len(vector)
    # Fill 1 from end
    for i in range(len(vector)-1, -1, -1):
        if vector[i] == 1:
            end = i
            break
        vector[i] = 1

    str_to_fix = ''.join(str(x) for x in vector)
    str_to_fix = str_to_fix.replace('1110111', '1111111')
    str_to_fix = str_to_fix.replace('0001000', '0000000')
    str_to_fix = str_to_fix.replace('11011', '11111')
    str_to_fix = str_to_fix.replace('00100', '00000')
    str_to_fix = str_to_fix.replace('101', '111')
    str_to_fix = str_to_fix.replace('010', '000')
    str_to_fix = str_to_fix.replace('101', '111')
    str_to_fix = str_to_fix.replace('010', '000')
    if boat_id != 1:
        str_to_fix = str_to_fix.replace('1001', '1111')
        str_to_fix = str_to_fix.replace('0110', '0000')
    vector = np.array(list(str_to_fix))

    rle = rle_encode(vector)

    result = np.zeros(vector.shape, dtype=np.uint32)
    counter = 1
    for i in range(len(rle) // 2):
        start = rle[2*i]
        end = rle[2*i] + rle[2*i+1]
        result[start:end] = counter
        if i < len(rle) // 2 - 1:
            start = rle[2 * i] + rle[2 * i + 1]
            end = rle[2 * (i + 1)]
            divide_pos = (start+end) // 2
            result[start:divide_pos] = counter
            result[divide_pos:end] = counter + 1
        counter += 1

    print('Fish found: {}'.format(max(result)))

    # Recalc probability array
    fish_type_part = fish_type_predictions[fish_type_predictions['video_id'] == name].copy()
    prob_arr = fish_type_part[FISH_TABLE].as_matrix()

    if type == 'train':
        out = open(CSV_PATH_VALID + name + '.csv', 'w')
    else:
        out = open(CSV_PATH_TEST + name + '.csv', 'w')
    out.write('frame,fish_number,length,species_fourspot,species_grey_sole,species_other,species_plaice,species_summer,species_windowpane,species_winter\n')
    for i in range(len(result)):
        out.write(str(frames_arr[i]))
        out.write(',' + str(result[i]))
        out.write(',' + str(length_arr[i]))
        for j in range(7):
            out.write(',' + str(prob_arr[i][j]))
        out.write('\n')
    out.close()
    return max(result)


def process_single_video_from_predictions_exp1(type, name, bboxes, boat_id, vector_predictions, fish_type_predictions, length_predictions):
    if type == 'train':
        prediction_path1 = CACHE_PATH_VALID + name + '_prediction.pklz'
        prediction_path2 = CACHE_PATH_VALID_2 + name + '_prediction.pklz'
        prediction_path3 = CACHE_PATH_VALID_3 + name + '_prediction.pklz'
        length_path = CACHE_LENGTH_VALID + name + '_length.pklz'
        roi_files = glob.glob(CACHE_ROI_VALID + name + '_*')
    else:
        prediction_path1 = CACHE_PATH_TEST + name + '_prediction.pklz'
        prediction_path2 = CACHE_PATH_TEST_2 + name + '_prediction.pklz'
        prediction_path3 = CACHE_PATH_TEST_3 + name + '_prediction.pklz'
        length_path = CACHE_LENGTH_TEST + name + '_length.pklz'
        roi_files = glob.glob(CACHE_ROI_TEST + name + '_*')

    pred_arr = load_from_file(prediction_path1)
    length_arr = np.array(load_from_file(length_path))

    # If predict for test
    if len(pred_arr.shape) == 3:
        pred_arr = pred_arr.mean(axis=0)

    frames_arr = list(range(pred_arr.shape[0]))
    prob_arr = pred_arr[:, :7].copy()
    vec_part = vector_predictions[vector_predictions['video_id'] == name].copy()
    vector = np.array(list(vec_part['prediction'].values))
    if boat_id == 1:
        thr = 0.6
    else:
        thr = 0.6
    vector[vector > thr] = 1
    vector[vector <= thr] = 0
    vector = list(vector.astype(np.uint8))

    # Fill 1 from beginning
    start = 0
    for i in range(len(vector)):
        if vector[i] == 1:
            start = i+1
            break
        vector[i] = 1

    end = len(vector)
    # Fill 1 from end
    for i in range(len(vector)-1, -1, -1):
        if vector[i] == 1:
            end = i
            break
        vector[i] = 1

    str_to_fix = ''.join(str(x) for x in vector)
    vector = np.array(list(str_to_fix))
    rle = rle_encode(vector)

    result = np.zeros(vector.shape, dtype=np.uint32)
    counter = 1
    for i in range(len(rle) // 2):
        start = rle[2*i]
        end = rle[2*i] + rle[2*i+1]
        result[start:end] = counter
        if i < len(rle) // 2 - 1:
            start = rle[2 * i] + rle[2 * i + 1]
            end = rle[2 * (i + 1)]
            divide_pos = (start+end) // 2
            result[start:divide_pos] = counter
            result[divide_pos:end] = counter + 1
        counter += 1

    print('Fish found: {}'.format(max(result)))

    # Recalc probability array
    fish_type_part = fish_type_predictions[fish_type_predictions['video_id'] == name].copy()
    prob_arr = fish_type_part[FISH_TABLE].as_matrix()

    if type == 'train':
        out = open(CSV_PATH_VALID + name + '.csv', 'w')
    else:
        out = open(CSV_PATH_TEST + name + '.csv', 'w')
    out.write('frame,fish_number,length,species_fourspot,species_grey_sole,species_other,species_plaice,species_summer,species_windowpane,species_winter\n')
    for i in range(len(result)):
        try:
            out.write(str(frames_arr[i]))
            out.write(',' + str(result[i]))
        except:
            out.write(str(i))
            out.write(',0')

        try:
            out.write(',' + str(length_arr[i]))
        except:
            out.write(str(0.0))
        for j in range(7):
            try:
                out.write(',' + str(prob_arr[i][j]))
            except:
                out.write(',0.0')
        out.write('\n')
    out.close()
    return max(result)


def process_all_train_videos(nfolds):
    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(nfolds)
    bboxes = pd.read_csv(ADD_PATH + 'bboxes_train.csv')
    vector_predictions = pd.read_csv(ADD_PATH + 'fish_exists_prediction_xgboost_train.csv')
    fish_type_predictions = pd.read_csv(ADD_PATH + 'fish_type_prediction_xgboost_train.csv')
    length_predictions = pd.read_csv(ADD_PATH + 'fish_length_prediction_xgboost_train.csv')
    boat_ids_train = pd.read_csv(ADD_PATH + "boat_ids_train.csv")
    train = pd.read_csv(INPUT_PATH + 'training.csv')

    num_fold = 0
    avg_error = 0
    pos_cases = 0
    neg_cases = 0
    zero_cases = 0

    avg_error_per_boat = {1: 0, 2: 0, 3: 0, 4: 0}
    pos_cases_per_boat = {1: 0, 2: 0, 3: 0, 4: 0}
    neg_cases_per_boat = {1: 0, 2: 0, 3: 0, 4: 0}
    zero_cases_per_boat = {1: 0, 2: 0, 3: 0, 4: 0}
    total_per_boat = {1: 0, 2: 0, 3: 0, 4: 0}

    total = 0
    for train_index, test_index in kfold_videos_split:
        num_fold += 1
        for i in test_index:
            # i = 1329
            name = os.path.basename(videos[i])
            print('Go for {}'.format(name))
            boat_id = boat_ids_train.loc[boat_ids_train['video_id'] == name, 'boat_id'].values[0]
            if boat_id == 1 or boat_id == 3:
                fish_found = process_single_video_from_predictions_type_1('train', name, bboxes, boat_id, vector_predictions, fish_type_predictions, length_predictions)
            else:
                fish_found = process_single_video_from_predictions_exp1('train', name, bboxes, boat_id, vector_predictions, fish_type_predictions, length_predictions)
            # fish_found = process_single_video_from_predictions('train', name, bboxes)
            tbl = train[train['video_id'] == name].copy()
            mx = tbl['fish_number'].max()
            print('Real fish number: {} Error: {}'.format(mx, mx-fish_found))
            avg_error += abs(mx-fish_found)
            avg_error_per_boat[boat_id] += abs(mx-fish_found)
            if mx < fish_found:
                neg_cases += 1
                neg_cases_per_boat[boat_id] += 1
            elif mx > fish_found:
                pos_cases += 1
                pos_cases_per_boat[boat_id] += 1
            else:
                zero_cases += 1
                zero_cases_per_boat[boat_id] += 1
            total += 1
            total_per_boat[boat_id] += 1
    print('Average fish error: {} No error: {} Pos errors: {} Neg errors: {}'.format(
        round(avg_error / total, 3), zero_cases, pos_cases, neg_cases))
    for boat_id in sorted(list(avg_error_per_boat.keys())):
        print('Boat: {} Average fish error: {} No error: {} Pos errors: {} Neg errors: {}'.format(
            boat_id, round(avg_error_per_boat[boat_id] / total_per_boat[boat_id], 3), zero_cases_per_boat[boat_id],
            pos_cases_per_boat[boat_id], neg_cases_per_boat[boat_id]))


def process_all_test_videos():
    bboxes = pd.read_csv(ADD_PATH + 'bboxes_test.csv')
    vector_predictions = pd.read_csv(ADD_PATH + 'fish_exists_prediction_xgboost_test.csv')
    fish_type_predictions = pd.read_csv(ADD_PATH + 'fish_type_prediction_xgboost_test.csv')
    length_predictions = pd.read_csv(ADD_PATH + 'fish_length_prediction_xgboost_test.csv')
    boat_ids_test = pd.read_csv(ADD_PATH + "boat_ids_test.csv")
    videos = glob.glob(INPUT_PATH + 'test_videos/*.mp4')
    # videos = glob.glob(INPUT_PATH + 'test_videos/jOP8cvFOQ9WmUlvB.mp4')
    for v in videos:
        name = os.path.basename(v)[:-4]
        print('Go for {}'.format(name))
        boat_id = boat_ids_test.loc[boat_ids_test['video_id'] == name, 'boat_id'].values[0]
        if boat_id == 1 or boat_id == 3:
            process_single_video_from_predictions_type_1('test', name, bboxes, boat_id, vector_predictions, fish_type_predictions, length_predictions)
        else:
            process_single_video_from_predictions_exp1('test', name, bboxes, boat_id, vector_predictions, fish_type_predictions, length_predictions)


if __name__ == '__main__':
    num_folds = 5
    process_all_train_videos(num_folds)
    process_all_test_videos()
