# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Extract region of interest for all videos based on UNET predictions
- Find bounding boxes (which contains fish) for all videos
'''

import os
import sys
import platform
import math
import shutil

gpu_use = 1
if platform.processor() == 'Intel64 Family 6 Model 63 Stepping 2, GenuineIntel' or platform.processor() == 'Intel64 Family 6 Model 79 Stepping 1, GenuineIntel':
    os.environ["THEANO_FLAGS"] = "device=gpu{},lib.cnmem=0.81,,base_compiledir='C:\\\\Users\\\\user\\\\AppData\\\\Local\\\\Theano{}'".format(gpu_use, gpu_use)
    if sys.version_info[1] > 4:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

from a00_common_functions import *
from a02_zf_unet_model import *

INPUT_PATH = "../input/"
AUGMENTATION_SIZE = 2


CACHE_PATH_VALID = "../cache_roi_train/"
if not os.path.isdir(CACHE_PATH_VALID):
    os.mkdir(CACHE_PATH_VALID)
CACHE_PATH_TEST = "../cache_roi_test/"
if not os.path.isdir(CACHE_PATH_TEST):
    os.mkdir(CACHE_PATH_TEST)
OUTPUT_PATH = "../modified_data/"
STORE_IMAGE_TEST = "../modified_data/debug_test/"
if not os.path.isdir(STORE_IMAGE_TEST):
    os.mkdir(STORE_IMAGE_TEST)
STORE_IMAGE_TRAIN = "../modified_data/debug_train/"
if not os.path.isdir(STORE_IMAGE_TRAIN):
    os.mkdir(STORE_IMAGE_TRAIN)

MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)


def get_image_augm_v3(tf):

    im_big = np.zeros((AUGMENTATION_SIZE, 720, 1280, 3), dtype=np.float32)
    # Orig image
    im_big[0] = tf.copy()
    # Flip image
    im_big[1] = np.fliplr(im_big[0].copy())

    im_big[im_big > 255] = 255
    im_big[im_big < 0] = 0

    return im_big


def restore_mask_v3(mask_list):

    mask = np.zeros((1, 720, 1280), dtype=np.float32)

    # Orig image restore
    mask += mask_list[0].copy()
    # Flip image restore
    mask += np.flip(mask_list[1].copy(), axis=2)

    mask /= AUGMENTATION_SIZE
    return mask


def get_masks_from_models_batch(model_list, image_list):
    augm_image_list = []
    for i in range(image_list.shape[0]):
        augm_image_list.append(get_image_augm_v3(image_list[i]))
    augm_image_list = np.concatenate(augm_image_list, axis=0)
    image_list = preprocess_batch(augm_image_list)
    image_list = image_list.transpose((0, 3, 1, 2))

    mask_list = []
    for model in model_list:
        mask_list.append(model.predict(image_list, batch_size=20))
    mask_list = np.array(mask_list).mean(axis=0)

    mask_restored = []
    for i in range(image_list.shape[0]//AUGMENTATION_SIZE):
        mask_restored.append(restore_mask_v3(mask_list[i*AUGMENTATION_SIZE:(i+1)*AUGMENTATION_SIZE]))
    mask_restored = np.array(mask_restored)
    mask_restored = (255*mask_restored[:, 0, :, :]).astype(np.uint8)
    return mask_restored


def get_roi_for_single_video(model_list, store_path, video_path):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('Video: {} Length: {} Resolution: {}x{} FPS: {}'.format(video_path, length, width, height, fps))
    current_frame = 0
    frames_arr = []
    masks_arr = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        frames_arr.append(frame)
        if len(frames_arr) >= 120:
            masks = get_masks_from_models_batch(model_list, np.array(frames_arr))
            frames_arr = []
            masks_arr.append(masks)
        current_frame += 1

    if len(frames_arr) > 0:
        masks = get_masks_from_models_batch(model_list, np.array(frames_arr))
        masks_arr.append(masks)

    masks_arr = np.concatenate(masks_arr, axis=0)
    # print(masks_arr.shape)
    split_num = 1000
    for i in range(0, masks_arr.shape[0], split_num):
        store1 = store_path
        if i > 0:
            store1 = store_path + '_part_{}.pklz'.format(i)
        if i + split_num <= masks_arr.shape[0]:
            save_in_file(masks_arr[i:i+split_num], store1)
        else:
            save_in_file(masks_arr[i:], store1)

    print('Total frames read: {} in {} sec'.format(current_frame, round(time.time() - start_time, 2)))
    if current_frame != length:
        print('Check some problem {} != {}'.format(current_frame, length))
        exit()
    cap.release()


def get_roi_for_train(nfolds):
    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(nfolds)

    num_fold = 0
    for train_index, test_index in kfold_videos_split:
        num_fold += 1

        cnn_type = 'ZF_UNET_1280_720_V2_SINGLE_OUTPUT_SMALL'

        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
        if not os.path.isfile(final_model_path):
            print('Model for fold {} doesnt exists. Skip it'.format(num_fold))
            continue
        model = ZF_UNET_1280_720_V2_SINGLE_OUTPUT_SMALL(dropout_val=0, batch_norm=True)
        model.load_weights(final_model_path)

        for i in test_index:
            name = os.path.basename(videos[i])
            video_path = INPUT_PATH + 'train_videos/' + name + '.mp4'
            store_path = CACHE_PATH_VALID + name + '_roi.pklz'
            if not os.path.isfile(store_path):
                get_roi_for_single_video([model], store_path, video_path)
            else:
                print('ROI file already exists: {}'.format(store_path))


def get_roi_for_test(nfolds):

    model_list = []
    for num_fold in range(1, nfolds+1):
        cnn_type = 'ZF_UNET_1280_720_V2_SINGLE_OUTPUT_SMALL'
        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
        if os.path.isfile(final_model_path):
            model = ZF_UNET_1280_720_V2_SINGLE_OUTPUT_SMALL(dropout_val=0, batch_norm=True)
            print('Read model: {}'.format(final_model_path))
            model.load_weights(final_model_path)
            model_list.append(model)

    videos = glob.glob(INPUT_PATH + 'test_videos/*.mp4')
    if gpu_use == 3:
        videos = videos[::-1]
    for v in videos:
        name = os.path.basename(v)[:-4]
        store_path = CACHE_PATH_TEST + name + '_roi.pklz'
        if not os.path.isfile(store_path):
            get_roi_for_single_video(model_list, store_path, v)
        else:
            print('ROI file already exists: {}'.format(store_path))


def get_roi_bounding_box(cache_path, name):
    roi_files = glob.glob(cache_path + name + '_*')
    masks_list = []
    for f in roi_files:
        mask = load_from_file(f)
        print(mask.shape)
        masks_list.append(mask)
    masks_list = np.concatenate(masks_list, axis=0)

    masks = masks_list
    print('Initial frames: {}'.format(masks.shape[0]))
    masks_reduced = []
    for i in range(masks.shape[0]):
        if masks[i].max() >= 200:
            masks_reduced.append(masks[i])
    masks_reduced = np.array(masks_reduced)
    print('Fixed frames: {}'.format(masks_reduced.shape[0]))
    if masks_reduced.shape[0] == 0:
        masks_reduced = masks

    avg = masks_reduced.mean(axis=0)
    div_point = (avg.max() + avg.min()) / 4
    print(avg.min(), avg.max(), div_point)
    avg[avg > div_point] = 255
    avg[avg < div_point] = 0
    avg = avg.astype(np.uint8)
    # show_image(avg)

    _, contours, hierarchy = cv2.findContours(avg.copy(), 1, 2)
    if len(contours) == 0:
        print('Some problem here')
        exit()
    c = max(contours, key=cv2.contourArea)
    # print(cv2.contourArea(c))
    x, y, w, h = cv2.boundingRect(c)
    # max_plane = cv2.rectangle(avg, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # show_image(max_plane)
    # print(y, y + h, x, x + w)

    return y, y + h, x, x + w


def check_roi_for_test_and_gen_bboxes(bbox_path):
    out = open(bbox_path, "w")
    out.write('id,sh0_start,sh0_end,sh1_start,sh1_end\n')
    videos = glob.glob(INPUT_PATH + 'test_videos/*.mp4')
    for v in videos:
        name = os.path.basename(v)[:-4]

        print('Go for {}'.format(name))
        out_path = STORE_IMAGE_TEST + name + '_{}.png'.format(0)
        if os.path.isfile(out_path):
            print('Skip!')
            continue

        try:
            bbox = get_roi_bounding_box(CACHE_PATH_TEST, name)
            sh0_start, sh0_end, sh1_start, sh1_end = bbox
        except:
            roi_path = CACHE_PATH_TEST + name + '_roi.pklz'
            print('Error test {}!'.format(name))
            if os.path.isfile(roi_path):
                shutil.copy(roi_path, roi_path + '_error')
            else:
                print('File is absent!')
            continue
        if 0:
            roi = load_from_file(roi_path)
            avg = roi.mean(axis=0)
            div_point = (avg.max() + avg.min()) / 2
            # print(avg.min(), avg.max(), div_point)
            # show_image(avg)
            avg[avg > div_point] = 255
            avg[avg < div_point] = 0
            # show_image(avg)

        cap = cv2.VideoCapture(v)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i in range(3):
            get_frame = int(i*length/3)
            print('Frame: {}'.format(get_frame))
            cap.set(1, get_frame)
            ret, frame = cap.read()

            # avg1 = np.expand_dims(avg, 2)
            # avg1 = np.concatenate((avg1, avg1, avg1), axis=2)
            # sm = ((avg1.astype(np.float32).copy() + frame.astype(np.float32).copy())/2).astype(np.uint8)
            out_path = STORE_IMAGE_TEST + name + '_{}.png'.format(i)
            sm = cv2.rectangle(frame, (sh1_start, sh0_start), (sh1_end, sh0_end), (0, 0, 255), 2)
            cv2.imwrite(out_path, sm)

        out.write(str(name))
        out.write(',' + str(sh0_start))
        out.write(',' + str(sh0_end))
        out.write(',' + str(sh1_start))
        out.write(',' + str(sh1_end))
        out.write('\n')
        out.flush()

    out.close()


def check_roi_for_train_and_gen_bboxes(bbox_path):
    out = open(bbox_path, "w")
    out.write('id,sh0_start,sh0_end,sh1_start,sh1_end\n')
    videos = glob.glob(INPUT_PATH + 'train_videos/*.mp4')
    for v in videos:
        name = os.path.basename(v)[:-4]

        print('Go for {}'.format(name))
        out_path = STORE_IMAGE_TRAIN + name + '_{}.png'.format(0)
        if os.path.isfile(out_path):
            print('Skip!')
            continue

        try:
            bbox = get_roi_bounding_box(CACHE_PATH_VALID, name)
            sh0_start, sh0_end, sh1_start, sh1_end = bbox
        except:
            roi_path = CACHE_PATH_VALID + name + '_roi.pklz'
            print('Error test {}!'.format(name))
            if os.path.isfile(roi_path):
                shutil.copy(roi_path, roi_path + '_error')
            else:
                print('File is absent!')
            continue
        if 0:
            roi = load_from_file(roi_path)
            avg = roi.mean(axis=0)
            div_point = (avg.max() + avg.min()) / 2
            # print(avg.min(), avg.max(), div_point)
            # show_image(avg)
            avg[avg > div_point] = 255
            avg[avg < div_point] = 0
            # show_image(avg)

        cap = cv2.VideoCapture(v)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i in range(3):
            get_frame = int(i*length/3)
            print('Frame: {}'.format(get_frame))
            cap.set(1, get_frame)
            ret, frame = cap.read()

            # avg1 = np.expand_dims(avg, 2)
            # avg1 = np.concatenate((avg1, avg1, avg1), axis=2)
            # sm = ((avg1.astype(np.float32).copy() + frame.astype(np.float32).copy())/2).astype(np.uint8)
            out_path = STORE_IMAGE_TRAIN + name + '_{}.png'.format(i)
            sm = cv2.rectangle(frame, (sh1_start, sh0_start), (sh1_end, sh0_end), (0, 0, 255), 2)
            cv2.imwrite(out_path, sm)

        out.write(str(name))
        out.write(',' + str(sh0_start))
        out.write(',' + str(sh0_end))
        out.write(',' + str(sh1_start))
        out.write(',' + str(sh1_end))
        out.write('\n')
        out.flush()

    out.close()


def get_roi_statistics_for_each_frame_train():
    bboxes = pd.read_csv(OUTPUT_PATH + 'bboxes_train.csv')
    videos = glob.glob(INPUT_PATH + 'train_videos/*.mp4')
    csv_path = OUTPUT_PATH  + 'roi_stat_train.csv'
    overall_list = []
    for v in videos:
        name = os.path.basename(v)[:-4]
        print('Go for {}'.format(name))
        roi_files = glob.glob(CACHE_PATH_VALID + name + '_*')
        masks_list = []
        for f in roi_files:
            mask = load_from_file(f)
            masks_list.append(mask)
        masks_list = np.concatenate(masks_list, axis=0)
        masks = masks_list
        print('Shape: {}'.format(masks.shape))
        masks_mx = masks.max(axis=(1, 2))
        masks_mean = masks.mean(axis=(1, 2))
        bbox = list(bboxes.loc[bboxes['id'] == name, ['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']].values[0])
        part = masks[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]
        print('Shape bbox: {}'.format(part.shape))
        masks_part_mx = part.max(axis=(1, 2))
        masks_part_mean = part.mean(axis=(1, 2))

        frames = list(range(masks.shape[0]))
        videos_list = [name].copy() * masks.shape[0]

        table = [frames, videos_list, list(masks_mx), list(masks_mean), list(masks_part_mx), list(masks_part_mean)]
        df = pd.DataFrame(table)
        df = df.transpose()
        df.columns = ['frame', 'video_id', 'masks_mx', 'masks_mean', 'masks_bbox_mx', 'masks_bbox_mean']
        overall_list.append(df)
    train = pd.concat(overall_list)
    train.to_csv(csv_path, index=False)


def get_roi_statistics_for_each_frame_test():
    bboxes = pd.read_csv(OUTPUT_PATH + 'bboxes_test.csv')
    videos = glob.glob(INPUT_PATH + 'test_videos/*.mp4')
    csv_path = OUTPUT_PATH  + 'roi_stat_test.csv'
    overall_list = []
    for v in videos:
        name = os.path.basename(v)[:-4]
        print('Go for {}'.format(name))
        roi_files = glob.glob(CACHE_PATH_TEST + name + '_*')
        masks_list = []
        for f in roi_files:
            mask = load_from_file(f)
            masks_list.append(mask)
        masks_list = np.concatenate(masks_list, axis=0)
        masks = masks_list
        print('Shape: {}'.format(masks.shape))
        masks_mx = masks.max(axis=(1, 2))
        masks_mean = masks.mean(axis=(1, 2))
        bbox = list(bboxes.loc[bboxes['id'] == name, ['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']].values[0])
        part = masks[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]
        print('Shape bbox: {}'.format(part.shape))
        masks_part_mx = part.max(axis=(1, 2))
        masks_part_mean = part.mean(axis=(1, 2))

        frames = list(range(masks.shape[0]))
        videos_list = [name].copy() * masks.shape[0]

        table = [frames, videos_list, list(masks_mx), list(masks_mean), list(masks_part_mx), list(masks_part_mean)]
        df = pd.DataFrame(table)
        df = df.transpose()
        df.columns = ['frame', 'video_id', 'masks_mx', 'masks_mean', 'masks_bbox_mx', 'masks_bbox_mean']
        overall_list.append(df)
    table = pd.concat(overall_list)
    table.to_csv(csv_path, index=False)


if __name__ == '__main__':
    num_folds = 5
    get_roi_for_train(num_folds)
    get_roi_for_test(num_folds)
    check_roi_for_train_and_gen_bboxes(OUTPUT_PATH + "bboxes_train.csv")
    check_roi_for_test_and_gen_bboxes(OUTPUT_PATH + "bboxes_test.csv")
    get_roi_statistics_for_each_frame_train()
    get_roi_statistics_for_each_frame_test()
