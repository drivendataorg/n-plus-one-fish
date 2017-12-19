import numpy as np
import pandas as pd
import json
import os
import pickle
from typing import List, Dict
from collections import namedtuple
from skimage.transform import SimilarityTransform
from sklearn.model_selection import train_test_split
import dataset
from dataset import SPECIES, CLASSES

EXTRA_LABELS_BASE_DIR = '../output/ruler_crops_batch_labeled'
EXTRA_LABELS_BATCHES = ['0', '100', '400', '500']

INPUT_ROWS = 720
INPUT_COLS = 360
input_shape = (INPUT_ROWS, INPUT_COLS, 3)
NUM_CLASSES = len(CLASSES)

FishDetection = namedtuple('FishDetection', ['video_id', 'frame', 'fish_number', 'x1', 'y1', 'x2', 'y2', 'class_id'])
RulerPoints = namedtuple('RulerPoints', ['x1', 'y1', 'x2', 'y2'])


class FishDetectionDataset:
    def __init__(self, is_test=False):
        self.is_test = is_test

        if is_test:
            self.video_clips = dataset.video_clips_test()
            self.fn_suffix = '_test'
        else:
            self.video_clips = dataset.video_clips()
            self.fn_suffix = ''

        cache_fn = '../output/fish_detection{}.pkl'.format(self.fn_suffix)
        try:
            # raise FileNotFoundError
            self.detections = pickle.load(open(cache_fn, 'rb'))  # type: Dict[FishDetection]
        except FileNotFoundError:
            self.detections = self.load()  # type: Dict[FishDetection]
            pickle.dump(self.detections, open(cache_fn, 'wb'))

        self.train_clips, self.test_clips = train_test_split(sorted(self.detections.keys()),
                                                             test_size=0.05,
                                                             random_state=12)

        self.nb_train_samples = sum([len(self.detections[clip]) for clip in self.train_clips])
        self.nb_test_samples = sum([len(self.detections[clip]) for clip in self.test_clips])

        self.ruler_points = {}
        ruler_points = pd.read_csv('../output/ruler_points{}.csv'.format(self.fn_suffix))
        for _, row in ruler_points.iterrows():
            self.ruler_points[row.video_id] = RulerPoints(x1=row.ruler_x0, y1=row.ruler_y0, x2=row.ruler_x1, y2=row.ruler_y1)

    def load(self):
        detections = {}
        ds = pd.read_csv('../input/N1_fish_N2_fish_-_Training_set_annotations.csv')

        species = ds.as_matrix(columns=['species_'+s for s in SPECIES])
        cls_column = np.argmax(species, axis=1)+1
        cls_column[np.max(species, axis=1) == 0] = 0

        for row_id, row in ds.iterrows():
            video_id = row.video_id
            if video_id not in detections:
                detections[video_id] = []

            detections[video_id].append(
                FishDetection(
                    video_id=video_id,
                    frame=row.frame,
                    fish_number=row.fish_number,
                    x1=row.x1, y1=row.y1,
                    x2=row.x2, y2=row.y2,
                    class_id=int(cls_column[row_id])
                )
            )
        # load labeled no fish images
        for extra_batch in EXTRA_LABELS_BATCHES:
            cover_class = 'no fish'
            for fn in os.listdir(os.path.join(EXTRA_LABELS_BASE_DIR, extra_batch, cover_class)):
                if not fn.endswith('.jpg'):
                    continue
                # file name format: video_frame.jpg
                fn = fn[:-len('.jpg')]
                video_id, frame = fn.split('_')
                frame = int(frame) - 1

                if video_id not in detections:
                    detections[video_id] = []

                detections[video_id].append(
                    FishDetection(
                        video_id=video_id,
                        frame=frame,
                        fish_number=0,
                        x1=np.nan, y1=np.nan,
                        x2=np.nan, y2=np.nan,
                        class_id=0
                    )
                )
        return detections

    def transform_for_clip(self, video_id, dst_w=720, dst_h=360, points_random_shift=0):
        points = self.ruler_points[video_id]

        ruler_points = np.array([[points.x1, points.y1], [points.x2, points.y2]])
        img_points = np.array([[dst_w * 0.1, dst_h / 2], [dst_w * 0.9, dst_h / 2]])

        if points_random_shift > 0:
            img_points += np.random.uniform(-points_random_shift, points_random_shift, (2, 2))

        tform = SimilarityTransform()
        tform.estimate(dst=ruler_points, src=img_points)

        return tform
