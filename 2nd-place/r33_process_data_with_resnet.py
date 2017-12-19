# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Create prediction with probabilities about type of fish or "no fish" for each frame of each video
- This file uses Inception v3 model for inference
- Predictions are cached in separate folder
'''

import os
import sys
import platform
import math

gpu_use = 1
if platform.processor() == 'Intel64 Family 6 Model 63 Stepping 2, GenuineIntel' or platform.processor() == 'Intel64 Family 6 Model 79 Stepping 1, GenuineIntel':
    os.environ["THEANO_FLAGS"] = "device=gpu{},lib.cnmem=0.81,,base_compiledir='C:\\\\Users\\\\user\\\\AppData\\\\Local\\\\Theano{}'".format(gpu_use, gpu_use)
    if sys.version_info[1] > 4:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

from a00_common_functions import *
from a02_zoo import *
from a00_augmentation_functions import *

random.seed(2016)
np.random.seed(2016)

INPUT_PATH = "../input/"
AUGMENTATION_SIZE = 4


CACHE_PATH_VALID = "../cache_resnet50_train/"
if not os.path.isdir(CACHE_PATH_VALID):
    os.mkdir(CACHE_PATH_VALID)
CACHE_PATH_TEST = "../cache_resnet50_test/"
if not os.path.isdir(CACHE_PATH_TEST):
    os.mkdir(CACHE_PATH_TEST)
MODELS_PATH = '../models/'
ADD_PATH = '../modified_data/'


def get_image_augm_dense_net(tf, bbox):
    sh0_start, sh0_end, sh1_start, sh1_end = bbox

    im_big = np.zeros((AUGMENTATION_SIZE, 224, 224, 3), dtype=np.float32)

    # Orig image
    part = tf[sh0_start:sh0_end, sh1_start:sh1_end, :]
    im_big[0] = cv2.resize(part, (224, 224), cv2.INTER_LANCZOS4)
    # Mirrored image
    im_big[1] = np.fliplr(im_big[0].copy())
    part = tf[max(0, sh0_start - 10):min(tf.shape[0], sh0_end + 10), max(0, sh1_start - 10):min(tf.shape[1], sh1_end + 10), :]
    im_big[2] = cv2.resize(part, (224, 224), cv2.INTER_LANCZOS4)
    part = tf[max(0, sh0_start - 20):min(tf.shape[0], sh0_end + 20), max(0, sh1_start - 20):min(tf.shape[1], sh1_end + 20), :]
    im_big[3] = cv2.resize(part, (224, 224), cv2.INTER_LANCZOS4)

    if 0:
        for i in range(im_big.shape[0]):
            show_image(im_big[i])
        exit()

    im_big[im_big > 255] = 255
    im_big[im_big < 0] = 0

    return im_big


def get_masks_from_models_batch(cnn_type, model_list, image_list, bbox):

    augm_image_list = []
    for i in range(image_list.shape[0]):
        augm_image_list.append(get_image_augm_dense_net(image_list[i], bbox))
    augm_image_list = np.concatenate(augm_image_list, axis=0)
    augm_image_list = augm_image_list.transpose((0, 3, 1, 2))
    image_list = preprocess_input_overall(cnn_type, augm_image_list)

    pred_list = []
    for model in model_list:
        pred_list.append(model.predict(image_list, batch_size=32))
    # Mean by models
    pred_list = np.array(pred_list).mean(axis=0)
    # print(pred_list.shape)

    # Mean by augmentations
    pred_restored = []
    for i in range(image_list.shape[0] // AUGMENTATION_SIZE):
        pred_restored.append(pred_list[i * AUGMENTATION_SIZE:(i + 1) * AUGMENTATION_SIZE].mean(axis=0))
    pred_restored = np.array(pred_restored)
    # print(pred_restored.shape)
    # print(pred_restored)

    return pred_restored


def create_predictions_for_single_video(cnn_type, model_list, bboxes, store_path, video_path):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read ROI
    name = os.path.basename(video_path)[:-4]
    bbox = list(bboxes.loc[bboxes['id'] == name, ['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']].values[0])

    # Process video
    print('Video: {} Length: {} Resolution: {}x{} FPS: {}'.format(video_path, length, width, height, fps))
    print('BBox:', bbox)
    current_frame = 0
    frames_arr = []
    masks_arr = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        frames_arr.append(frame)
        if len(frames_arr) >= 32:
            masks = get_masks_from_models_batch(cnn_type, model_list, np.array(frames_arr), bbox)
            frames_arr = []
            masks_arr.append(masks)
        current_frame += 1

    if len(frames_arr) > 0:
        masks = get_masks_from_models_batch(cnn_type, model_list, np.array(frames_arr), bbox)
        masks_arr.append(masks)

    masks_arr = np.concatenate(masks_arr, axis=0)
    # print(masks_arr.shape)
    save_in_file(masks_arr, store_path)

    print('Total frames read: {} in {} sec'.format(current_frame, round(time.time() - start_time, 2)))
    if current_frame != length:
        print('Check some problem {} != {}'.format(current_frame, length))
        exit()
    cap.release()


def create_predictions_with_resnet50_for_validation(nfolds):
    files, kfold_images_split, videos, kfold_videos_split = get_kfold_split(nfolds)
    bboxes = pd.read_csv(ADD_PATH + "bboxes_train.csv")

    num_fold = 0
    for train_index, test_index in kfold_videos_split:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split videos train: ', len(train_index))
        print('Split videos valid: ', len(test_index))

        cnn_type = 'RESNET50'
        model = get_pretrained_model(cnn_type, 8, final_layer_activation='softmax')
        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
        model.load_weights(final_model_path)

        for i in test_index:
            name = os.path.basename(videos[i])
            video_path = INPUT_PATH + 'train_videos/' + name + '.mp4'
            store_path = CACHE_PATH_VALID + name + '_prediction.pklz'

            try:
                bbox = list(bboxes.loc[bboxes['id'] == name, ['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']].values[0])
            except:
                print('No bbox found for video {}'.format(name))
                continue

            if not os.path.isfile(store_path):
                create_predictions_for_single_video(cnn_type, [model], bboxes, store_path, video_path)
            else:
                print('Prediction with ResNet50 file already exists: {}'.format(store_path))


def create_predictions_with_resnet50_for_test(nfolds):
    bboxes = pd.read_csv(ADD_PATH + "bboxes_test.csv")

    model_list = []
    cnn_type = 'RESNET50'
    for i in range(1, nfolds+1):
        model = get_pretrained_model(cnn_type, 8, final_layer_activation='softmax')
        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, i)
        model.load_weights(final_model_path)
        model_list.append(model)

    num_fold = 0
    videos = glob.glob(INPUT_PATH + 'test_videos/*.mp4')
    for v in videos:
        name = os.path.basename(v)[:-4]
        num_fold += 1
        store_path = CACHE_PATH_TEST + name + '_prediction.pklz'
        if not os.path.isfile(store_path):
            create_predictions_for_single_video(cnn_type, model_list, bboxes, store_path, v)
        else:
            print('Prediction with ResNet50 file already exists: {}'.format(store_path))


if __name__ == '__main__':
    num_folds = 5
    create_predictions_with_resnet50_for_validation(num_folds)
    create_predictions_with_resnet50_for_test(num_folds)
