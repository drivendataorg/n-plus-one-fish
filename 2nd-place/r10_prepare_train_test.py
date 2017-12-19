# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Fix column name in training from 'species_grey sole' to 'species_grey_sole'
- Extract frames from video files according to 'training.csv' and create masks
'''

from a00_common_functions import *
import shutil
import cv2
import pandas as pd


INPUT_PATH = "../input/"
OUTPUT_IMAGES = "../input/train_images/"
if not os.path.isdir(OUTPUT_IMAGES):
    os.mkdir(OUTPUT_IMAGES)


def get_fish_name(row):
    for f in FISH_TABLE:
        if row[f].values[0] == 1:
            return f
    return ''

# I replaced ' ' with '_' manually in the beginning of contest

def small_fix_for_training_csv():
    shutil.copy(INPUT_PATH + 'training.csv', INPUT_PATH + 'training_.csv')
    in1 = open(INPUT_PATH + 'training_.csv')
    out = open(INPUT_PATH + 'training.csv', 'w')
    line = in1.readline()
    line = line.replace('species_grey sole', 'species_grey_sole')
    out.write(line)
    while 1:
        line = in1.readline()
        if line == '':
            break
        out.write(line)
    in1.close()
    out.close()


def create_train():
    train = pd.read_csv(INPUT_PATH + 'training.csv')
    unique = train['video_id'].unique()
    print('Unique videos:', len(unique))
    for video_id in unique:
        print('Go for {}'.format(video_id))
        video_path = INPUT_PATH + 'train_videos/' + video_id + '.mp4'
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        subtable = train[train['video_id'] == video_id].copy()
        frames_to_store = list(subtable['frame'].values)

        print('Length: {} Resolution: {}x{} FPS: {}'.format(length, width, height, fps))
        current_frame = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret is False:
                break
            if current_frame in frames_to_store:
                row = subtable[subtable['frame'] == current_frame].copy()
                row_id = row['row_id'].values[0]
                fish_name = get_fish_name(row)
                color = COLOR_TABLE[fish_name]
                out_prefix = OUTPUT_IMAGES + str(row_id) + '_' + video_id + '_' + str(current_frame) + '_' + fish_name
                out_img_path = out_prefix + '_orig.jpg'
                cv2.imwrite(out_img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                out_mask_path = out_prefix + '_mask.png'
                if fish_name != '':
                    x1 = int(row['x1'].values[0])
                    y1 = int(row['y1'].values[0])
                    x2 = int(row['x2'].values[0])
                    y2 = int(row['y2'].values[0])
                    if x2 < x1:
                        c = x2
                        x2 = x1
                        x1 = c
                    if y2 < y1:
                        c = y2
                        y2 = y1
                        y1 = c

                    delta = 50
                    if abs(x2 - x1) < delta:
                        x1 -= delta
                        x2 += delta
                        if x1 < 0:
                            x1 = 0
                    if abs(y2 - y1) < delta:
                        y1 -= delta
                        y2 += delta
                        if y1 < 0:
                            y1 = 0

                    print('Box: {} {} {} {}'.format(x1, y1, x2, y2))
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    mask[y1:y2, x1:x2] = color
                else:
                    print('Box: empty')
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.imwrite(out_mask_path, mask)
            current_frame += 1
        print('Total frames read: {}'.format(current_frame))
        if current_frame != length:
            print('Check some problem {} != {}'.format(current_frame, length))
            exit()
        cap.release()


if __name__ == '__main__':
    small_fix_for_training_csv()
    create_train()
