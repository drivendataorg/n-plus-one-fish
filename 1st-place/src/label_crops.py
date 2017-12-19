import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import sys
import shutil
import os
import dataset

# RULER_CROPS_DIR = '.'
RULER_CROPS_DIR = '../output/ruler_crops'


class CropsLabeler:
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs
        self.current_dir_id = 0
        self.file_names = []
        self.current_file_id = 0
        self.current_dir = ''

        self.fig, self.axes = plt.subplots(ncols=3)
        self.fig.canvas.mpl_connect('key_press_event', self.key_event)
        print(self.axes)
        self.imgplots = [
            axis.imshow(np.zeros((10, 20)))
            for axis in self.axes
        ]
        self.current_fn = ''
        self.open_dir(0)
        plt.show()

    def img_fn(self, file_id):
        return os.path.join(RULER_CROPS_DIR, self.data_dirs[self.current_dir_id], self.file_names[file_id])

    def load_img_with_label(self, file_id):
        if file_id < 0 or file_id >= len(self.file_names):
            return np.zeros((10, 20)), ''

        fn = self.img_fn(file_id)

        img = np.zeros((10, 20))
        try:
            img = mpimg.imread(fn)
            with open(fn.replace('.jpg', '.txt'), 'r') as f:
                label = f.read()
        except FileNotFoundError:
            label = ''

        return img, label

    def open_file(self, file_id):
        self.current_file_id = file_id
        self.current_fn = self.img_fn(file_id)

        self.fig.suptitle('{} {}/{}  "j" no fish, "k" hand over fish or not on ruler, "l" clear on ruler'.format(self.data_dirs[self.current_dir_id], self.current_file_id+1, len(self.file_names)))

        for ax_idx,cur_file_id in enumerate([file_id-1, file_id, file_id+1]):
            img, label = self.load_img_with_label(cur_file_id)

            self.imgplots[ax_idx].set_data(img)
            self.axes[ax_idx].set_title('{} {}'.format(cur_file_id+1, label))
        self.fig.canvas.draw()

    def open_dir(self, dir_id):
        self.current_dir_id = dir_id
        self.current_dir = os.path.join(RULER_CROPS_DIR, self.data_dirs[dir_id])
        self.file_names = []
        for fn in sorted(os.listdir(self.current_dir)):
            if fn.endswith('.jpg'):
                self.file_names.append(fn)
        self.open_file(0)

    def open_next_file(self):
        if self.current_file_id+1 < len(self.file_names):
            self.open_file(self.current_file_id+1)
        else:
            self.open_dir(self.current_dir_id+1)

    def open_prev_file(self):
        if self.current_file_id > 0:
            self.open_file(self.current_file_id-1)
        else:
            self.open_dir(self.current_dir_id-1)

    def save_label(self, label):
        fn = self.current_fn.replace('.jpg', '.txt')
        with open(fn, 'w+') as f:
            f.write(label)
        self.open_next_file()

    def key_event(self, ev):
        print(ev.key)
        if ev.key == 'down' or ev.key == 'right':
            self.open_next_file()
        elif ev.key == 'up' or ev.key == 'left':
            self.open_prev_file()
        elif ev.key == 'j':
            self.save_label('no fish')
        elif ev.key == 'k':
            self.save_label('hand over fish')
        elif ev.key == 'l':
            self.save_label('fish clear')



def generate_batches(srd_dir, dest_dir):
    from shutil import copyfile

    clips = sorted(os.listdir(srd_dir))
    items = 100
    images = 50
    for start in range(0, len(clips), items):
        batch_dst_dir = os.path.join(dest_dir, str(start))
        os.makedirs(batch_dst_dir, exist_ok=True)
        for clip in clips[start:start+items]:
            print(clip)
            os.makedirs(os.path.join(batch_dst_dir, clip), exist_ok=True)
            clip_src_dir = os.path.join(srd_dir, clip)
            copied_images = 0
            for file in sorted(os.listdir(clip_src_dir)):
                if file.endswith('.jpg'):
                    copied_images += 1
                    copyfile(os.path.join(clip_src_dir, file), os.path.join(batch_dst_dir, clip, file))
                if copied_images >= images:
                    break


# generate_batches('../output/ruler_crops', '../output/ruler_crops_batch')
# label = CropsLabeler(sorted(os.listdir(sys.argv[1:])))


def check_orig_dataset():
    output_dir = '../output/orig_ds_crops'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv('../input/N1_fish_N2_fish_-_Training_set_annotations.csv')
    for _, row in df.iterrows():
        video_id = row.video_id
        frame = row.frame
        length = row.length

        if not np.isnan(length):
            shutil.copy(dataset.image_crop_fn(video_id, frame),
                        os.path.join(output_dir, '{}_{:04}.jpg'.format(video_id, frame)))

check_orig_dataset()
