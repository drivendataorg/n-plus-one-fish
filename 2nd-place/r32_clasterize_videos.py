# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
- Split boats on different types (assign Boat ID for each video)
'''

from a00_common_functions import *
import shutil


INPUT_PATH = "../input/"
ADD_PATH = '../modified_data/'
if not os.path.isdir(ADD_PATH):
    os.mkdir(ADD_PATH)
OUTPUT_PATH = ADD_PATH + 'cluster_frames/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

random.seed(2016)
np.random.seed(2016)


def get_boat_id(sh0_start, sh0_end, sh1_start, sh1_end):
    w = sh1_end - sh1_start
    h = sh0_end - sh0_start
    if sh1_end > 1100:
        return 1
    if sh0_end > 600:
        return 2
    if w > 1.5*h:
        return 3
    return 4


def claseterize_train_test_by_bbox():
    output_debug_images = 0
    bb_train = pd.read_csv(ADD_PATH + 'bboxes_train.csv')
    bb_test = pd.read_csv(ADD_PATH + 'bboxes_test.csv')

    claster_train = dict()
    claster_test = dict()
    for b, claster, path_part in [(bb_train, claster_train, 'train_videos'), (bb_test, claster_test, 'test_videos')]:
        place_mask = np.zeros((720, 1280), dtype=np.float32)

        for index, row in b.iterrows():
            sh0_start, sh0_end, sh1_start, sh1_end = list(row[['sh0_start', 'sh0_end', 'sh1_start', 'sh1_end']])
            place_mask[sh0_start:sh0_end, sh1_start:sh1_end] += 1.0
            boat_id = get_boat_id(sh0_start, sh0_end, sh1_start, sh1_end)
            claster[row['id']] = boat_id
            if output_debug_images:
                v = INPUT_PATH + path_part + '/' + row['id'] + '.mp4'
                cap = cv2.VideoCapture(v)
                cap.set(1, 10)
                ret, frame = cap.read()
                out_dir = OUTPUT_PATH + str(boat_id) + '/'
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)
                out_dir = out_dir + str(path_part) + '/'
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)
                cv2.imwrite(out_dir + row['id'] + '.jpg', frame)

        if 1:
            place_mask /= place_mask.max()
            place_mask *= 255
        else:
            place_mask[place_mask > 0] = 255

        for i in range(0, place_mask.shape[0], 200):
            place_mask[i:i+1, :] = 255

        for i in range(0, place_mask.shape[1], 200):
            place_mask[:, i:i+1] = 255

        place_mask = place_mask.astype(np.uint8)
        # show_image(place_mask)

    print(claster_train)
    print(claster_test)

    train_boat_stat = [0]*5
    out = open(ADD_PATH + 'boat_ids_train.csv', 'w')
    out.write('video_id,boat_id\n')
    for el in claster_train:
        out.write(el + ',' + str(claster_train[el]) + '\n')
        train_boat_stat[claster_train[el]] += 1
    out.close()

    print('Train boats: {}'.format(train_boat_stat[1:]))

    test_boat_stat = [0] * 5
    out = open(ADD_PATH + 'boat_ids_test.csv', 'w')
    out.write('video_id,boat_id\n')
    for el in claster_test:
        out.write(el + ',' + str(claster_test[el]) + '\n')
        test_boat_stat[claster_test[el]] += 1
    out.close()

    print('Test boats: {}'.format(test_boat_stat[1:]))


if __name__ == '__main__':
    claseterize_train_test_by_bbox()

# Train boats: [83, 403, 713, 133]
# Test boats: [182, 107, 361, 17]