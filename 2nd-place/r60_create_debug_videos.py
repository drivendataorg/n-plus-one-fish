# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Code provide functions to create visualisation of predictions directly on train/test videos
- It was used to generate video: https://www.youtube.com/watch?v=OlDPPF_0lWY
'''

from a00_common_functions import *

INPUT_PATH = "../input/"
OUTPUT_PATH = "../modified_data/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


def create_video(image_list, out_file):
    height, width = image_list[0].shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = -1
    fps = 5.0
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height), True)
    for im in image_list:
        video.write(im.astype(np.uint8))
    cv2.destroyAllWindows()
    video.release()


def get_main_fish_text(length, train, pred_arr):
    full_text_pred = []
    full_text_real = []
    prev_text = ''
    prev_text_frames = 0
    for current_frame in range(length):
        index = np.argmax(pred_arr[current_frame])
        if index == 7:
            text = 'Pred: No fish'
        else:
            text = 'Pred: Fish (' + str(index) + '): ' + FISH_TABLE[index]
        full_text_pred.append(text)

        row_exists = train[train['frame'] == current_frame].copy()
        if len(row_exists) > 0:
            index = np.argmax(list(row_exists[FISH_TABLE].values))
            if list(row_exists[FISH_TABLE].values[0])[index] == 0:
                text = 'Real: No fish'
            else:
                text = 'Real: Fish (' + str(index) + '): ' + FISH_TABLE[index]
            prev_text = text
            prev_text_frames = 4
        else:
            if prev_text_frames <= 0:
                text = 'Real: -----'
            else:
                text = prev_text
                prev_text_frames -= 1
        full_text_real.append(text)
    return full_text_pred, full_text_real


def get_current_fish_text(length, train, train_predictions):
    full_text_pred = []
    full_text_real = ([0]*length).copy()
    train = train.copy()
    train = train.fillna(-1)
    for current_frame in range(length):
        fish = train_predictions.loc[current_frame, 'fish_number']
        text = 'Pred fish number: {}'.format(fish)
        full_text_pred.append(text)

    prev_frame = -1
    current_frame = 0
    last_fish = 0
    for index, row in train.iterrows():
        current_frame = int(row['frame'])
        if row['fish_number'] == -1:
            continue
        # print(current_frame, row['fish_number'])
        if prev_frame == -1:
            full_text_real[:current_frame] = [row['fish_number']]*len(full_text_real[:current_frame])
        else:
            avg = (prev_frame + current_frame) // 2
            full_text_real[prev_frame:avg] = [row['fish_number'] - 1]*len(full_text_real[prev_frame:avg])
            full_text_real[avg:current_frame] = [row['fish_number']]*len(full_text_real[avg:current_frame])
        prev_frame = current_frame
        last_fish = row['fish_number']
    full_text_real[current_frame:] = [last_fish]*len(full_text_real[current_frame:])
    for i in range(length):
        full_text_real[i] = 'Real fish number: {}'.format(full_text_real[i])

    return full_text_pred, full_text_real


def get_length_data(length, train, train_predictions):
    full_pred = []
    full_real = ([0]*length).copy()
    train = train.copy()
    train = train.fillna(-1)
    for current_frame in range(length):
        sm = np.sum(list(train_predictions.loc[current_frame, FISH_TABLE].values))
        if sm > 0.5:
            fish = round(train_predictions.loc[current_frame, 'length'], 3)
        else:
            fish = 0
        full_pred.append(fish)

    for index, row in train.iterrows():
        current_frame = int(row['frame'])
        if row['length'] == -1:
            continue
        # print(current_frame, row['frame'])
        full_real[current_frame:min(current_frame+4, len(full_real))] = [row['length']]*len(full_real[current_frame:current_frame+4])

    return full_pred, full_real


def create_background(frame):
    frame = frame.astype(np.uint32)
    frame[3:80, 10: 600, :] += 120
    frame[93:170, 10: 600, :] += 120
    frame[183:270, 10: 600, :] += 120

    frame[frame > 255] = 255
    frame = frame.astype(np.uint8)
    return frame


def create_background_test(frame):
    frame = frame.astype(np.uint32)
    frame[3:40, 10: 600, :] += 120
    frame[53:90, 10: 600, :] += 120
    frame[103:150, 10: 600, :] += 120

    frame[frame > 255] = 255
    frame = frame.astype(np.uint8)
    return frame


def check_fish_detection_train(name):
    start_time = time.time()
    train = pd.read_csv(INPUT_PATH + 'training.csv')
    train = train[train['video_id'] == name].copy()
    train_fish_loc = train.fillna(0)
    cache_path = "../cache_dense_net_train/"
    prediction_path = cache_path + name + '_prediction.pklz'
    pred_arr = load_from_file(prediction_path)
    prediction_path = "../cache_csv_train/" + name + '.csv'
    train_predictions = pd.read_csv(prediction_path)
    bboxes = pd.read_csv("../modified_data/bboxes_train.csv")
    bbox = list(bboxes.loc[bboxes['id'] == name, ['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']].values[0])
    sh0_start, sh0_end, sh1_start, sh1_end = bbox

    video_path = INPUT_PATH + 'train_videos/' + name + '.mp4'
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    current_fish = 0
    frame_list = []

    roi_files = glob.glob("../cache_roi_train/" + name + '_*')
    masks_list = []
    for f in roi_files:
        mask = load_from_file(f)
        masks_list.append(mask)
    masks_list = np.concatenate(masks_list, axis=0)
    masks = masks_list

    full_text_pred, full_text_real = get_main_fish_text(length, train, pred_arr)
    cf_text_pred, cf_text_real = get_current_fish_text(length, train, train_predictions)
    length_pred, length_real = get_length_data(length, train, train_predictions)
    prev_train_fish_location = (-1, -1, -1, -1)
    prev_train_fish_location_count = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break

        # Place where fish located by prediction in red channel
        frame = frame.astype(np.float32)
        # frame[:, :, 0] = frame[:, :, 0] - (masks[current_frame] * (255. - frame[:, :, 0])) / 255.
        # frame[:, :, 1] = frame[:, :, 1] - (masks[current_frame] * (255. - frame[:, :, 1])) / 255.
        frame[:, :, 0] = frame[:, :, 0] - (masks[current_frame] * (255. - 0.5*frame[:, :, 0])) / 255.
        frame[:, :, 1] = frame[:, :, 1] - (masks[current_frame] * (255. - 0.5*frame[:, :, 1])) / 255.
        frame[frame > 255] = 255
        frame[frame < 0] = 0
        frame = frame.astype(np.uint8)

        # print(pred_arr[current_frame])
        sm = create_background(frame)
        sm = cv2.rectangle(sm, (sh1_start, sh0_start), (sh1_end, sh0_end), (0, 0, 255), 2)
        sm = cv2.rectangle(sm, (max(sh1_start - 30, 0), max(sh0_start - 30, 0)), (min(sh1_end + 30, sm.shape[1]), min(sh0_end + 30, sm.shape[0])), (0, 0, 200), 1)
        sm = cv2.rectangle(sm, (max(sh1_start - 50, 0), max(sh0_start - 50, 0)), (min(sh1_end + 50, sm.shape[1]), min(sh0_end + 50, sm.shape[0])), (0, 0, 150), 1)

        # Predicted fish
        cv2.putText(sm, full_text_pred[current_frame], (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(sm, full_text_real[current_frame], (20, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        # Predicted fish nimber
        cv2.putText(sm, cf_text_pred[current_frame], (20, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(sm, cf_text_real[current_frame], (20, 160), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        # Predicted fish length
        cv2.putText(sm, 'Pred fish length: {}'.format(length_pred[current_frame]), (20, 210), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(sm, (20, 220), (20 + int(length_pred[current_frame]), 225), (0, 0, 255), -1)
        cv2.putText(sm, 'Real fish length: {}'.format(length_real[current_frame]), (20, 250), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(sm, (20, 260), (20 + int(length_real[current_frame]), 265), (255, 0, 0), -1)

        # Fish location from train
        row_exists = train_fish_loc[train_fish_loc['frame'] == current_frame].copy()
        if len(row_exists) > 0:
            x1, y1, x2, y2 = row_exists[['x1', 'y1', 'x2', 'y2']].values[0]
            cv2.line(sm, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            prev_train_fish_location = (x1, y1, x2, y2)
            prev_train_fish_location_count = 2
        else:
            if prev_train_fish_location_count > 0:
                x1, y1, x2, y2 = prev_train_fish_location
                cv2.line(sm, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                prev_train_fish_location_count -= 1

        # sm = cv2.resize(sm, (sm.shape[1] // 2, sm.shape[0] // 2), cv2.INTER_LINEAR)
        frame_list.append(sm)
        # show_image(sm)
        current_frame += 1
    create_video(frame_list, OUTPUT_PATH + 'train_videos_debug/' + name + '.avi')


def check_fish_detection_test(name):
    start_time = time.time()
    cache_path = "../cache_dense_net_test/"
    prediction_path = cache_path + name + '_prediction.pklz'
    pred_arr = load_from_file(prediction_path)
    prediction_path = "../cache_csv_test/" + name + '.csv'
    test_predictions = pd.read_csv(prediction_path)
    bboxes = pd.read_csv("../modified_data/bboxes_test.csv")
    bbox = list(bboxes.loc[bboxes['id'] == name, ['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']].values[0])
    sh0_start, sh0_end, sh1_start, sh1_end = bbox

    roi_files = glob.glob("../cache_roi_test/" + name + '_*')
    masks_list = []
    for f in roi_files:
        mask = load_from_file(f)
        masks_list.append(mask)
    masks_list = np.concatenate(masks_list, axis=0)
    masks = masks_list

    video_path = INPUT_PATH + 'test_videos/' + name + '.mp4'
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    frame_list = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break

        # Place where fish located by prediction in red channel
        frame = frame.astype(np.float32)
        frame[:, :, 0] = frame[:, :, 0] - (masks[current_frame] * (255. - 0.5 * frame[:, :, 0])) / 255.
        frame[:, :, 1] = frame[:, :, 1] - (masks[current_frame] * (255. - 0.5 * frame[:, :, 1])) / 255.
        frame[frame > 255] = 255
        frame[frame < 0] = 0
        frame = frame.astype(np.uint8)

        # print(pred_arr[current_frame])
        sm = create_background_test(frame)
        sm = cv2.rectangle(sm, (sh1_start, sh0_start), (sh1_end, sh0_end), (0, 0, 255), 2)
        sm = cv2.rectangle(sm, (max(sh1_start - 30, 0), max(sh0_start - 30, 0)),
                           (min(sh1_end + 30, sm.shape[1]), min(sh0_end + 30, sm.shape[0])), (0, 0, 200), 1)
        sm = cv2.rectangle(sm, (max(sh1_start - 50, 0), max(sh0_start - 50, 0)),
                           (min(sh1_end + 50, sm.shape[1]), min(sh0_end + 50, sm.shape[0])), (0, 0, 150), 1)

        # Predicted fish number
        fish_number = test_predictions.loc[current_frame, 'fish_number']
        text = 'Pred fish number: {}'.format(fish_number)
        cv2.putText(sm, text, (20, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        # Predicted fish length
        sm1 = np.sum(list(test_predictions.loc[current_frame, FISH_TABLE].values))
        if sm1 > 0.5:
            fish_len = round(test_predictions.loc[current_frame, 'length'], 3)
        else:
            fish_len = 0
        cv2.putText(sm, 'Pred fish length: {}'.format(fish_len), (20, 130), cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(sm, (20, 140), (20 + int(fish_len), 145), (0, 0, 255), -1)

        index = np.argmax(pred_arr[current_frame])
        if index == 7:
            text = 'No fish'
        else:
            text = 'Fish (' + str(index) + '): ' + FISH_TABLE[index]
        cv2.putText(sm, text, (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        # sm = cv2.resize(sm, (sm.shape[1] // 2, sm.shape[0] // 2), cv2.INTER_LINEAR)
        frame_list.append(sm)
        # show_image(sm)

        current_frame += 1
    create_video(frame_list, OUTPUT_PATH + 'test_videos_debug/' + name + '.avi')


def gen_all_test_videos():
    videos = glob.glob(INPUT_PATH + 'test_videos/*.mp4')
    for v in videos:
        name = os.path.basename(v)[:-4]
        print('Go for {}'.format(name))
        check_fish_detection_test(name)


if __name__ == '__main__':
    check_fish_detection_train('7h0Z0CIk6rwLXFCk')
    check_fish_detection_train('4dUnLM3j6Td9LIip')
    check_fish_detection_train('C9uc9V1SHNpSgpTP')
    check_fish_detection_train('KX6y4acwIZsfEZkH')
    check_fish_detection_test('ACkrSixZy57Grkqz')
    check_fish_detection_test('PcCYQkmXWEph69DK')
    check_fish_detection_test('QQd1LxoFTtA6R5jW')
    check_fish_detection_test('rQphXiUQqVFUxi0o')
    gen_all_test_videos()