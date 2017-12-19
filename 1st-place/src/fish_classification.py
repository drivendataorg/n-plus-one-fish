import argparse
from collections import namedtuple

import numpy as np
import pandas as pd
import skimage
import skimage.transform
from typing import Union
import sys
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from scipy.misc import imread, imresize
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.layers import Input, Activation, BatchNormalization, UpSampling2D
from keras.layers.merge import concatenate, multiply
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.applications import ResNet50, Xception, InceptionV3
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
import keras.backend

import matplotlib.pyplot as plt
from PIL import Image
from copy import copy
import concurrent.futures

from multiprocessing.pool import ThreadPool
from typing import List, Dict

import pickle, os, random
import utils
import scipy.misc

import img_augmentation

import dataset
from dataset import SPECIES, CLASSES
import fish_detection

import densenet161
import densenet121
from fish_masks import model_unet

EXTRA_LABELS_BASE_DIR = '../output/ruler_crops_batch_labeled'
EXTRA_LABELS_BATCHES = ['0', '100', '400', '500']

CROP_WIDTH = 720
CROP_HEIGHT = 360

# INPUT_ROWS = 224
# INPUT_COLS = 224
INPUT_ROWS = 300
INPUT_COLS = 300
INPUT_SHAPE = (INPUT_ROWS, INPUT_COLS, 3)
SPECIES_CLASSES = CLASSES

COVER_CLASSES = ['no fish', 'hand over fish', 'fish clear']
CLASS_NO_FISH_ID = 0
CLASS_HAND_OVER_ID = 1
CLASS_FISH_CLEAR_ID = 2


def build_model_densenet_161():
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = densenet161.DenseNet(
        img_input=img_input,
        reduction=0.5,
        weights_path='../input/densenet161_weights_tf.h5',
        classes=1000)
    base_model.layers.pop()
    base_model.layers.pop()

    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model.layers[-1].output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model.layers[-1].output)
    # output = concatenate([species_dense, cover_dense], axis=0)

    model = Model(input=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model_resnet50():
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = ResNet50(input_tensor=img_input, include_top=False, pooling='avg')

    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model.layers[-1].output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model.layers[-1].output)

    model = Model(input=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model_xception():
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = Xception(input_tensor=img_input, include_top=False, pooling='avg')

    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model.layers[-1].output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model.layers[-1].output)

    model = Model(input=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model_inception():
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = InceptionV3(input_tensor=img_input, include_top=False, pooling='avg')

    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model.layers[-1].output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model.layers[-1].output)

    model = Model(input=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model_resnet50_with_mask():
    img_input = Input(INPUT_SHAPE, name='data')
    mask_model = model_unet(INPUT_SHAPE)
    mask_model.load_weights('../output/checkpoints/fish_mask_unet/model_fish_unet2/checkpoint-best-064-0.0476.hdf5')

    mask = mask_model(img_input)
    mask3 = concatenate([mask, mask, mask], axis=3)
    masked_image = multiply([img_input, mask3])

    base_model = ResNet50(input_shape=INPUT_SHAPE, include_top=False, pooling='avg')
    base_model_output = base_model(masked_image)
    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model_output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model_output)

    model = Model(inputs=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model_densenet_with_mask():
    img_input = Input(INPUT_SHAPE, name='data')
    mask_model = model_unet(INPUT_SHAPE)
    mask_model.load_weights('../output/checkpoints/fish_mask_unet/model_fish_unet2/checkpoint-best-064-0.0476.hdf5')

    mask = mask_model(img_input)
    mask3 = concatenate([mask, mask, mask], axis=3)
    masked_image = multiply([img_input, mask3])

    base_model = densenet161.DenseNet(
        img_input=img_input,
        reduction=0.5,
        weights_path='../input/densenet161_weights_tf.h5',
        classes=1000)
    base_model.layers.pop()
    base_model.layers.pop()
    base_model_output = base_model(masked_image)
    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model_output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model_output)

    model = Model(inputs=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model_densenet121_with_mask():
    img_input = Input(INPUT_SHAPE, name='data')
    mask_model = model_unet(INPUT_SHAPE)
    mask_model.load_weights('../output/checkpoints/fish_mask_unet/model_fish_unet2/checkpoint-best-064-0.0476.hdf5')

    mask = mask_model(img_input)
    mask3 = concatenate([mask, mask, mask], axis=3)
    masked_image = multiply([img_input, mask3])

    base_model = densenet121.DenseNet(
        img_input=img_input,
        reduction=0.5,
        weights_path='../input/densenet121_weights_tf.h5',
        classes=1000)
    base_model.layers.pop()
    base_model.layers.pop()
    base_model_output = base_model(masked_image)
    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model_output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model_output)

    model = Model(inputs=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model_densenet_121():
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = densenet121.DenseNet(
        img_input=img_input,
        reduction=0.5,
        weights_path='../input/densenet121_weights_tf.h5',
        classes=1000)
    base_model.layers.pop()
    base_model.layers.pop()

    species_dense = Dense(len(SPECIES_CLASSES), activation='softmax', name='cat_species')(base_model.layers[-1].output)
    cover_dense = Dense(len(COVER_CLASSES), activation='softmax', name='cat_cover')(base_model.layers[-1].output)
    # output = concatenate([species_dense, cover_dense], axis=0)

    model = Model(input=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


FishClassification = namedtuple('FishClassification', ['video_id',
                                                       'frame',
                                                       'x', 'y', 'w',
                                                       'species_class',
                                                       'cover_class'])

SSDDetection = namedtuple('SSDDetection', ['video_id',
                                           'frame',
                                           'x', 'y', 'w', 'h',
                                           'class_id', 'confidence'
                                           ])

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])


class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self,
                 fish_classification: FishClassification,
                 saturation=0.5, contrast=0.5, brightness=0.5, color_shift=0.5,  # 0.5  - no changes, range 0..1
                 scale_rect_x=1.0, scale_rect_y=1.0,
                 shift_x_ratio=0.0, shift_y_ratio=0.0,
                 angle=0.0,
                 hflip=False,
                 vflip=False,
                 blurred_by_downscaling=1,
                 random_pos=False,
                 ssd_detection=None):
        self.color_shift = color_shift
        self.ssd_detection = ssd_detection
        self.angle = angle
        self.shift_x_ratio = shift_x_ratio
        self.shift_y_ratio = shift_y_ratio
        self.scale_rect_y = scale_rect_y
        self.scale_rect_x = scale_rect_x
        self.fish_classification = fish_classification
        self.vflip = vflip
        self.hflip = hflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blurred_by_downscaling = blurred_by_downscaling
        self.cache_img = False

        w = np.clip(fish_classification.w + 64, 200, 360)
        x = fish_classification.x
        y = np.clip(fish_classification.y, CROP_HEIGHT / 2 - 64, CROP_HEIGHT / 2 + 64)

        if random_pos or (fish_classification.cover_class == CLASS_NO_FISH_ID and abs(x) < 0.01):
            w = random.randrange(200, 360)
            x = random.randrange(200, CROP_WIDTH - 200)
            y = random.randrange(CROP_HEIGHT / 2 - 64, CROP_HEIGHT / 2 + 64)

        self.rect = Rect(x=x - w / 2, y=y - w / 2, w=w, h=w)

    def __lt__(self, other):
        return True

    def __str__(self):
        return dataset.CLASSES[self.fish_classification.species_class] + ' ' + str(self.__dict__)


def load_ssd_detection(video_id, frame_id, data_dir='../output/predictions_ssd_roi2/vgg_41') -> SSDDetection:
    fn = os.path.join(data_dir, video_id, '{:04}.npy'.format(frame_id + 1))
    try:
        results = np.load(fn)
    except FileNotFoundError:
        print("ssd prediction not found:", fn)
        return None

    if len(results) == 0:
        return None

    det_label = results[:, 0]
    det_conf = results[:, 1]
    det_xmin = results[:, 2]
    det_ymin = results[:, 3]
    det_xmax = results[:, 4]
    det_ymax = results[:, 5]
    top_indices_conf = sorted([(conf, i) for i, conf in enumerate(det_conf) if conf >= 0.1], reverse=True)
    if len(top_indices_conf) == 0:
        return None

    idx = top_indices_conf[0][1]

    return SSDDetection(
        video_id, frame_id,
        x=(det_xmin[idx] + det_xmax[idx]) / 2 * CROP_WIDTH,
        y=(det_ymin[idx] + det_ymax[idx]) / 2 * CROP_HEIGHT,
        w=(det_xmax[idx] - det_xmin[idx]) * CROP_WIDTH,
        h=(det_ymax[idx] - det_ymin[idx]) * CROP_HEIGHT,
        class_id=det_label[idx],
        confidence=det_conf[idx]
    )


class ClassificationDataset(fish_detection.FishDetectionDataset):
    def __init__(self, fold=0, preprocess_input=preprocess_input):
        super().__init__()
        self.preprocess_input = preprocess_input
        print('build clip transforms')
        self.clip_transforms = {
            video_id: self.transform_for_clip(video_id) for video_id in self.video_clips
        }

        self.data = []  # type: List[FishClassification]
        # video_id->frame->species:
        self.known_species = {}  # type: Dict[str, Dict[int, int]]
        self.data, self.known_species = self.load()

        all_video_ids = set(self.video_clips.keys())
        self.test_video_ids = set(dataset.fold_test_video_ids(fold))
        self.train_video_ids = all_video_ids.difference(self.test_video_ids)

        self.train_data = [d for d in self.data if d.video_id in self.train_video_ids]
        self.test_data_full = [d for d in self.data if d.video_id in self.test_video_ids]
        self.test_data = self.test_data_full[::2]

        self.test_data_for_clip = {}
        for d in self.test_data_full:
            if not d.video_id in self.test_data_for_clip:
                self.test_data_for_clip[d.video_id] = []
            self.test_data_for_clip[d.video_id].append(d)

        self.crops_cache = {}

        print('train samples: {} test samples {}'.format(len(self.train_data), len(self.test_data)))

    def train_batches(self, batch_size):
        return int(len(self.train_data) / 2 // batch_size)

    def test_batches(self, batch_size):
        return int(len(self.test_data) // batch_size)

    def load(self):
        repeat_samples = {
            CLASS_FISH_CLEAR_ID: 1,
            CLASS_HAND_OVER_ID: 4,
            CLASS_NO_FISH_ID: 2
        }
        data = []
        known_species = {}
        used_frames = {video_id: set() for video_id in self.video_clips.keys()}
        # we can use the original dataset for clear and no_fish classes
        # ssd detected boxes for extra labeled dataset for all classes, when can guess species from the main dataset
        for video_id, detections in self.detections.items():
            for detection in detections:
                used_frames[video_id].add(detection.frame)

                if detection.class_id != 0:
                    if detection.video_id not in known_species:
                        known_species[detection.video_id] = {}
                    known_species[detection.video_id][detection.frame] = detection.class_id
                    ssd_detection = load_ssd_detection(detection.video_id, detection.frame)
                    if ssd_detection is not None:
                        data.append(
                            FishClassification(
                                video_id=video_id,
                                frame=detection.frame,
                                x=ssd_detection.x, y=ssd_detection.y, w=ssd_detection.w,
                                species_class=detection.class_id, cover_class=CLASS_FISH_CLEAR_ID
                            )
                        )
                else:
                    data.append(
                        FishClassification(
                            video_id=detection.video_id,
                            frame=detection.frame,
                            x=0, y=0, w=0, species_class=0, cover_class=CLASS_NO_FISH_ID,
                        )
                    )
        print('base size:', len(data))
        # load extra labeled images
        for extra_batch in EXTRA_LABELS_BATCHES:
            for cover_class_id, cover_class in enumerate(COVER_CLASSES):
                for fn in os.listdir(os.path.join(EXTRA_LABELS_BASE_DIR, extra_batch, cover_class)):
                    if not fn.endswith('.jpg'):
                        continue
                    # file name format: video_frame.jpg
                    fn = fn[:-len('.jpg')]
                    video_id, frame = fn.split('_')
                    frame = int(frame) - 1

                    used_frames[video_id].add(frame)

                    for _ in range(repeat_samples[cover_class_id]):
                        if cover_class_id == CLASS_NO_FISH_ID:
                            data.append(
                                FishClassification(
                                    video_id=video_id,
                                    frame=frame,
                                    x=0, y=0, w=0, species_class=0, cover_class=cover_class_id
                                )
                            )
                        # keep no fish also here, so classificator learns to fix detector mistakes
                        ssd_detection = load_ssd_detection(video_id, frame,
                                                           data_dir='../output/predictions_ssd_roi2/resnet_53')
                        species_class = guess_species(known_species[video_id], frame)
                        if ssd_detection is not None and species_class is not None:
                            data.append(
                                FishClassification(
                                    video_id=video_id,
                                    frame=frame,
                                    x=ssd_detection.x, y=ssd_detection.y, w=ssd_detection.w,
                                    species_class=species_class, cover_class=cover_class_id
                                )
                            )

        print('data size:', len(data))
        pickle.dump(used_frames, open('../output/used_frames.pkl', 'wb'))
        return data, known_species

    def generate_x(self, cfg: SampleCfg):
        img = scipy.misc.imread(dataset.image_crop_fn(cfg.fish_classification.video_id, cfg.fish_classification.frame))

        crop = utils.get_image_crop(full_rgb=img, rect=cfg.rect,
                                    scale_rect_x=cfg.scale_rect_x, scale_rect_y=cfg.scale_rect_y,
                                    shift_x_ratio=cfg.shift_x_ratio, shift_y_ratio=cfg.shift_y_ratio,
                                    angle=cfg.angle, out_size=INPUT_ROWS)

        crop = crop.astype('float32')
        if cfg.saturation != 0.5:
            crop = img_augmentation.saturation(crop, variance=0.2, r=cfg.saturation)

        if cfg.contrast != 0.5:
            crop = img_augmentation.contrast(crop, variance=0.25, r=cfg.contrast)

        if cfg.brightness != 0.5:
            crop = img_augmentation.brightness(crop, variance=0.3, r=cfg.brightness)

        if cfg.hflip:
            crop = img_augmentation.horizontal_flip(crop)

        if cfg.vflip:
            crop = img_augmentation.vertical_flip(crop)

        if cfg.blurred_by_downscaling != 1:
            crop = img_augmentation.blurred_by_downscaling(crop, 1.0 / cfg.blurred_by_downscaling)
        return crop * 255.0

    def generate_xy(self, cfg: SampleCfg):
        return self.generate_x(cfg), cfg.fish_classification.species_class, cfg.fish_classification.cover_class

    def generate(self, batch_size, skip_pp=False, verbose=False):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        def rand_or_05():
            if random.random() > 0.5:
                return random.random()
            return 0.5

        while True:
            sample = random.choice(self.train_data)  # type: FishClassification
            cfg = SampleCfg(fish_classification=sample,
                            saturation=rand_or_05(),
                            contrast=rand_or_05(),
                            brightness=rand_or_05(),
                            color_shift=rand_or_05(),
                            shift_x_ratio=random.uniform(-0.2, 0.2),
                            shift_y_ratio=random.uniform(-0.2, 0.2),
                            angle=random.uniform(-20.0, 20.0),
                            hflip=random.choice([True, False]),
                            vflip=random.choice([True, False]),
                            blurred_by_downscaling=np.random.choice([1, 1, 1, 1, 1, 1, 1, 1, 2, 2.5, 3, 4])
                            )
            if verbose:
                print(cfg)
            samples_to_process.append(cfg)

            if len(samples_to_process) == batch_size:
                batch_samples = pool.map(self.generate_xy, samples_to_process)
                # batch_samples = [self.generate_xy(sample) for sample in samples_to_process]
                X_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
                y_batch_species = np.array([batch_sample[1] for batch_sample in batch_samples])
                y_batch_cover = np.array([batch_sample[2] for batch_sample in batch_samples])
                if not skip_pp:
                    X_batch = self.preprocess_input(X_batch)
                    y_batch_species = to_categorical(y_batch_species, num_classes=len(SPECIES_CLASSES))
                    y_batch_cover = to_categorical(y_batch_cover, num_classes=len(COVER_CLASSES))
                samples_to_process = []
                yield X_batch, {'cat_species': y_batch_species, 'cat_cover': y_batch_cover}

    def generate_test(self, batch_size, skip_pp=False, verbose=False):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        while True:
            for sample in self.test_data[:int(len(self.test_data) // batch_size) * batch_size]:
                cfg = SampleCfg(fish_classification=sample)
                if verbose:
                    print(cfg)
                samples_to_process.append(cfg)

                if len(samples_to_process) == batch_size:
                    batch_samples = pool.map(self.generate_xy, samples_to_process)
                    X_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
                    y_batch_species = np.array([batch_sample[1] for batch_sample in batch_samples])
                    y_batch_cover = np.array([batch_sample[2] for batch_sample in batch_samples])
                    if not skip_pp:
                        X_batch = self.preprocess_input(X_batch)
                        y_batch_species = to_categorical(y_batch_species, num_classes=len(SPECIES_CLASSES))
                        y_batch_cover = to_categorical(y_batch_cover, num_classes=len(COVER_CLASSES))
                    samples_to_process = []
                    yield X_batch, {'cat_species': y_batch_species, 'cat_cover': y_batch_cover}

    def generate_full_test_for_clip(self, batch_size, pool, video_id, skip_pp=False):
        all_configs = [SampleCfg(fish_classification=sample) for sample in self.test_data_for_clip[video_id]]
        all_configs = sorted(all_configs, key=lambda x: x.frame)
        for samples_to_process in utils.chunks(all_configs, batch_size):
            batch_samples = pool.map(self.generate_xy, samples_to_process)
            X_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
            if not skip_pp:
                X_batch = self.preprocess_input(X_batch)
            yield X_batch


def guess_species(known_species, frame_id):
    known_frames = sorted(known_species.keys())
    if len(known_frames) == 0:
        return None

    for i, frame in enumerate(known_frames):
        if frame == frame_id:
            return known_species[frame]
        elif frame > frame_id:
            if i == 0:
                return known_species[frame]
            if known_species[frame] == known_species[known_frames[i - 1]]:
                return known_species[frame]
            else:
                return None

    return known_species[known_frames[-1]]


def test_guess_species():
    known_species = {2: 1, 5: 1, 7: 2}

    assert guess_species(known_species, 0) == 1
    assert guess_species(known_species, 1) == 1
    assert guess_species(known_species, 2) == 1
    assert guess_species(known_species, 3) == 1
    assert guess_species(known_species, 5) == 1
    assert guess_species(known_species, 6) is None
    assert guess_species(known_species, 7) == 2
    assert guess_species(known_species, 8) == 2


def check_dataset_generator():
    dataset = ClassificationDataset(fold=1)

    batch_size = 2
    for x_batch, y_batch in dataset.generate(batch_size=batch_size, skip_pp=True, verbose=True):
        print(y_batch)
        for i in range(batch_size):
            print(np.min(x_batch[i]), np.max(x_batch[i]))
            plt.imshow(x_batch[i] / 256.0)
            # print(SPECIES_CLASSES[y_batch['cat_species'][i]], COVER_CLASSES[y_batch['cat_cover'][i]])
            plt.show()


def train(fold, continue_from_epoch=0, weights='', batch_size=8, model_type='densenet'):

    preprocess_input_func = preprocess_input

    if model_type == 'densenet':
        model = build_model_densenet_161()
        model_name = 'model_densenet161_ds3'
        lock_layer1 = 'pool5'
        lock_layer2 = 'pool4'
    elif model_type == 'densenet121':
        model = build_model_densenet_121()
        model_name = 'model_densenet121'
        lock_layer1 = 'pool5'
        lock_layer2 = 'pool4'
    elif model_type == 'densenet121_mask':
        model = build_model_densenet121_with_mask()
        model_name = 'model_densenet121_mask'
        lock_layer1 = 'cat_species'
        lock_layer2 = 'densenet'
    elif model_type == 'densenet2':
        model = build_model_densenet_161()
        model_name = 'model_densenet161_ds3'
        lock_layer1 = 'pool5'
        lock_layer2 = 'pool4'
    elif model_type == 'resnet50':
        model = build_model_resnet50()
        model_name = 'model_resnet50_cat'
        lock_layer1 = 'activation_49'
        lock_layer2 = 'activation_40'
    elif model_type == 'xception':
        model = build_model_xception()
        model_name = 'model_xception'
        lock_layer1 = 'block14_sepconv2_act'
        lock_layer2 = 'block14_sepconv1'
        preprocess_input_func = preprocess_input_xception
    elif model_type == 'inception':
        model = build_model_inception()
        model_name = 'model_inception'
        lock_layer1 = 'mixed10'
        lock_layer2 = 'mixed9'
        preprocess_input_func = preprocess_input_xception
    elif model_type == 'resnet50_mask':
        model = build_model_resnet50_with_mask()
        model_name = 'model_resnet50_mask'
        lock_layer1 = 'cat_species'
        lock_layer2 = 'resnet50'
    else:
        print('Invalid model_type', model_type)
        return

    model.summary()

    dataset = ClassificationDataset(fold=fold, preprocess_input=preprocess_input_func)
    checkpoints_dir = '../output/checkpoints/classification/{}_fold_{}'.format(model_name, fold)
    tensorboard_dir = '../output/tensorboard/classification/{}_fold_{}'.format(model_name, fold)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    if len(weights) > 0:
        model.load_weights(weights)

    def cheduler(epoch):
        if epoch < 1:
            return 5e-4
        if epoch < 5:
            return 3e-4
        if epoch < 10:
            return 1e-4
        if epoch < 20:
            return 5e-5
        return 1e-5

    validation_batch_size = 8

    if continue_from_epoch == 0:
        utils.lock_layers_until(model, lock_layer1)
        model.summary()
        model.fit_generator(dataset.generate(batch_size=batch_size),
                            steps_per_epoch=dataset.train_batches(batch_size),
                            epochs=1,
                            verbose=1,
                            callbacks=[],
                            validation_data=dataset.generate_test(batch_size=validation_batch_size),
                            validation_steps=dataset.test_batches(validation_batch_size),
                            initial_epoch=0)
        continue_from_epoch += 1

    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5",
                                            verbose=1,
                                            save_weights_only=True,
                                            period=1)
    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=False, write_images=True)
    lr_sched = LearningRateScheduler(schedule=cheduler)

    utils.lock_layers_until(model, lock_layer2)
    model.summary()

    nb_epoch = 4
    model.fit_generator(dataset.generate(batch_size=batch_size),
                        steps_per_epoch=dataset.train_batches(batch_size),
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=[
                            checkpoint_periodical,
                            tensorboard,
                            lr_sched
                        ],
                        validation_data=dataset.generate_test(batch_size=validation_batch_size),
                        validation_steps=dataset.test_batches(validation_batch_size),
                        initial_epoch=continue_from_epoch + 1)


def check(fold, weights):
    dataset = ClassificationDataset(fold=fold)

    model = build_model_densenet_161()
    model.load_weights(weights)

    batch_size = 2
    for x_batch, y_batch in dataset.generate(batch_size=batch_size):
        print(y_batch)
        predicted = model.predict_on_batch(x_batch)
        print(predicted)
        for i in range(batch_size):
            plt.imshow(utils.preprocessed_input_to_img_resnet(x_batch[i]))
            true_species = y_batch['cat_species'][i]
            true_cover = y_batch['cat_cover'][i]
            predicted_species = predicted[0][i]
            predicted_cover = predicted[1][i]

            for cls_id, cls in enumerate(SPECIES_CLASSES):
                print('{:12} {:.02f} {:.02f}'.format(cls, true_species[cls_id], predicted_species[cls_id]))

            for cls_id, cls in enumerate(COVER_CLASSES):
                print('{:12} {:.02f} {:.02f}'.format(cls, true_cover[cls_id], predicted_cover[cls_id]))

            print(SPECIES_CLASSES[np.argmax(y_batch['cat_species'][i])],
                  COVER_CLASSES[np.argmax(y_batch['cat_cover'][i])])
            plt.show()


def generate_crops_from_detection_results(crops_dir,
                                          detection_results_dir,
                                          classification_crops_dir,
                                          save_jpegs):
    print('load ssd results:')
    os.makedirs(classification_crops_dir, exist_ok=True)
    configs = []
    video_ids = sorted(os.listdir(detection_results_dir))
    for i, video_id in enumerate(video_ids):
        if i % 10 == 0:
            print('{} / {}, {:.2}%'.format(i, len(video_ids), 100.0 * i / len(video_ids)))
        video_clip_dir = os.path.join(detection_results_dir, video_id)
        for detection_fn in sorted(os.listdir(video_clip_dir)):
            if not detection_fn.endswith('.npy'):
                continue
            frame = int(detection_fn[:-len('.npy')]) - 1
            ssd_detection = load_ssd_detection(video_id, frame, data_dir=detection_results_dir)

            if ssd_detection is None:
                continue

            # reuse the same logic as used during training
            configs.append(
                SampleCfg(
                    fish_classification=FishClassification(
                        video_id=video_id,
                        frame=frame,
                        x=ssd_detection.x, y=ssd_detection.y, w=ssd_detection.w,
                        species_class=ssd_detection.class_id, cover_class=CLASS_FISH_CLEAR_ID
                    )
                )
            )

    def process_sample(cfg: SampleCfg):
        src_fn = os.path.join(crops_dir,
                              cfg.fish_classification.video_id,
                              '{:04}.jpg'.format(int(cfg.fish_classification.frame) + 1))
        img = scipy.misc.imread(src_fn)
        crop = utils.get_image_crop(full_rgb=img, rect=cfg.rect, out_size=INPUT_ROWS)
        # utils.print_stats('crop', crop)

        os.makedirs(os.path.join(classification_crops_dir,
                                 cfg.fish_classification.video_id), exist_ok=True)
        if save_jpegs:
            dst_jpg_fn = os.path.join(classification_crops_dir,
                                      cfg.fish_classification.video_id,
                                      '{:04}.jpg'.format(int(cfg.fish_classification.frame) + 1))
            scipy.misc.imsave(dst_jpg_fn, crop)

        # dst_fn = os.path.join(classification_crops_dir,
        #                       cfg.fish_classification.video_id,
        #                       '{:04}.npy'.format(int(cfg.fish_classification.frame) + 1))
        # utils.print_stats('crop', crop)
        # utils.print_stats('crop conv', np.clip(crop, 0, 255).astype(np.uint8))
        #
        # crop *= 255.0
        # np.save(dst_fn, np.clip(crop, 0, 255).astype(np.uint8))

    print('process samples')
    # process_sample(configs[0])
    pool = ThreadPool(processes=8)
    pool.map(process_sample, configs)


def save_detection_results(detection_results_dir, output_dir):
    print('load ssd results:')
    video_ids = sorted(os.listdir(detection_results_dir))
    os.makedirs(output_dir, exist_ok=True)
    for i, video_id in enumerate(video_ids):
        if i % 10 == 0:
            print('{} / {}, {:.2}%'.format(i, len(video_ids), 100.0 * i / len(video_ids)))
        configs = []
        video_clip_dir = os.path.join(detection_results_dir, video_id)
        for detection_fn in sorted(os.listdir(video_clip_dir)):
            if not detection_fn.endswith('.npy'):
                continue
            frame = int(detection_fn[:-len('.npy')]) - 1
            ssd_detection = load_ssd_detection(video_id, frame, data_dir=detection_results_dir)

            if ssd_detection is None:
                continue

            # reuse the same logic as used during training
            configs.append(
                SampleCfg(
                    fish_classification=FishClassification(
                        video_id=video_id,
                        frame=frame,
                        x=ssd_detection.x, y=ssd_detection.y, w=ssd_detection.w,
                        species_class=ssd_detection.class_id, cover_class=CLASS_FISH_CLEAR_ID
                    ),
                    ssd_detection=ssd_detection
                )
            )
        df = pd.DataFrame({'frame': [cfg.fish_classification.frame for cfg in configs]})
        df['video_id'] = video_id
        df['x'] = [cfg.fish_classification.x for cfg in configs]
        df['y'] = [cfg.fish_classification.y for cfg in configs]
        df['w'] = [cfg.fish_classification.w for cfg in configs]
        df['h'] = [cfg.ssd_detection.h for cfg in configs]
        df['detection_conf'] = [cfg.ssd_detection.confidence for cfg in configs]
        df['detection_species'] = [cfg.ssd_detection.class_id for cfg in configs]
        df.to_csv(os.path.join(output_dir, video_id + '_ssd_detection.csv'), index=False, float_format='%.8f')


def generate_results_from_detection_crops_on_fold(fold, weights, crops_dir, output_dir, video_ids=None,
                                                  hflip=0, vflip=0, model_type=''):

    preprocess_input_func = preprocess_input

    if model_type in ('densenet', 'densenet2'):
        model = build_model_densenet_161()
    elif model_type in ('densenet121',):
        model = build_model_densenet_121()
    elif model_type == 'resnet50':
        model = build_model_resnet50()
    elif model_type == 'resnet50_mask':
        model = build_model_resnet50_with_mask()
    elif model_type == 'resnet50_mask5':
        model = build_model_resnet50_with_mask()
    elif model_type in ('xception', 'xception5'):
        model = build_model_xception()
        preprocess_input_func = preprocess_input_xception
    elif model_type == 'inception':
        model = build_model_inception()
        preprocess_input_func = preprocess_input_inception
    else:
        print('Invalid model_type', model_type)
        return

    model.load_weights(weights)

    os.makedirs(output_dir, exist_ok=True)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

    if video_ids is None:
        video_ids = sorted(dataset.fold_test_video_ids(fold))

    batch_size = 8
    for video_id in video_ids:
        src_dir = os.path.join(crops_dir, video_id)
        files = [f for f in sorted(os.listdir(src_dir)) if f.endswith('.jpg')]

        def load_data():
            for batch_files in utils.chunks(files, batch_size):
                res = []
                for fn in batch_files:
                    # print('load', fn)
                    img = scipy.misc.imread(os.path.join(src_dir, fn))
                    if hflip:
                        img = img[:, ::-1]
                    if vflip:
                        img = img[::-1]
                    # utils.print_stats('img', img)
                    # img = np.load(os.path.join(src_dir, fn))
                    # plt.imshow(img)
                    # plt.show()
                    img = img.astype(np.float32)  # * 255.0
                    res.append(img)
                res = np.array(res)
                yield preprocess_input_func(res)

        results_species = []
        results_cover = []
        for batch_data in utils.parallel_generator(load_data(), executor):
            # for batch_data in load_data():
            res_species, res_cover = model.predict_on_batch(batch_data)
            #
            # for i in range(batch_data.shape[0]):
            #     print(SPECIES_CLASSES[np.argmax(res_species[i])],
            #           COVER_CLASSES[np.argmax(res_cover[i])])
            #     print(res_species[i])
            #     print(res_cover[i])

            results_species.append(res_species)
            results_cover.append(res_cover)

        frames = [int(f[:-len('.npy')]) - 1 for f in files]
        df = pd.DataFrame({'frame': frames})
        df['video_id'] = video_id

        results_species = np.row_stack(results_species)
        results_cover = np.row_stack(results_cover)

        for i, species_cls in enumerate(SPECIES_CLASSES):
            df['species_' + species_cls] = results_species[:, i]

        for i, cover_cls in enumerate(COVER_CLASSES):
            df[cover_cls] = results_cover[:, i]

        df.to_csv(os.path.join(output_dir, video_id + '_categories.csv'), index=False, float_format='%.4f')
        # break


def combine_test_results(classification_results_dir, output_dir):
    video_ids = sorted(list(dataset.video_clips_test().keys()))
    os.makedirs(output_dir, exist_ok=True)

    columns ='species__,species_fourspot,species_grey sole,species_other,species_plaice,species_summer,species_windowpane,species_winter,no fish,hand over fish,fish clear'.split(',')

    fold_suffixes = ['', '_hflip', '_vflip', '_hflip_vflip']
    # to match submission, RNN was trained on densent with only one hflip
    if 'densenet' in classification_results_dir:
        fold_suffixes = ['', '_hflip']

    for video_id in video_ids:
        data_frames = [
            pd.read_csv(os.path.join(classification_results_dir, str(fold)+fold_suffix, video_id+'_categories.csv'))
            for fold in range(1, 5)
            for fold_suffix in fold_suffixes
        ]

        combined = pd.concat(data_frames)
        by_row_index = combined.groupby('frame')
        df_means = by_row_index.mean()
        # print(video_id)
        # print(df_means.head())

        res_df = data_frames[0]
        for col in columns:
            res_df[col] = df_means[col].as_matrix()

        res_df.to_csv(os.path.join(output_dir, video_id + '_categories.csv'), index=False, float_format='%.4f')


def save_crops_from_dataset():
    dataset = ClassificationDataset(fold=1)
    dataset.test_data = dataset.test_data + dataset.train_data
    random.shuffle(dataset.test_data)

    batch_size = 1
    img_num = 0
    for x_batch, y_batch in dataset.generate_test(batch_size=batch_size, skip_pp=True, verbose=False):
        # print(y_batch)
        res_dir = '../output/fish_masks_train/{:02}/'.format(img_num // 100)
        res_fn = '{:03}.jpg'.format(img_num % 100)
        img_num += 1
        if img_num > 10000:
            break

        os.makedirs(res_dir, exist_ok=True)
        Image.fromarray(x_batch[0].astype(np.uint8)).save(res_dir+res_fn, format='JPEG', subsampling=0, quality=100)
        # scipy.misc.imsave(, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ruler masks')
    parser.add_argument('action', type=str, default='combine_results_test')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--detection_model', type=str, default='')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--hflip', type=int, default=0)
    parser.add_argument('--vflip', type=int, default=0)
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument('--classification_model', type=str)

    args = parser.parse_args()
    action = args.action
    detection_model = args.detection_model
    classification_model = args.classification_model

    if action == 'train':
        train(fold=args.fold, weights=args.weights, continue_from_epoch=args.initial_epoch, model_type=classification_model)
    if action == 'check_dataset_generator':
        check_dataset_generator()
    if action == 'check':
        check(fold=args.fold, weights=args.weights)
    if action == 'save_crops_from_dataset':
        save_crops_from_dataset()
    if action == 'generate_train_classification_crops':
        generate_crops_from_detection_results(crops_dir=dataset.RULER_CROPS_DIR,
                                              detection_results_dir='../output/predictions_ssd_roi2/' + detection_model,
                                              classification_crops_dir='../output/classification_crop/' + detection_model,
                                              save_jpegs=True)
    if action == 'generate_test_classification_crops':
        generate_crops_from_detection_results(crops_dir=dataset.RULER_CROPS_DIR_TEST,
                                              detection_results_dir='../output/predictions_ssd_roi2_test/' + detection_model,
                                              classification_crops_dir='../output/classification_crop_test/' + detection_model,
                                              save_jpegs=True)
    if action == 'generate_results_from_detection_crops_on_fold':
        generate_results_from_detection_crops_on_fold(fold=args.fold,
                                                      weights=args.weights,
                                                      crops_dir='../output/classification_crop/' + detection_model,
                                                      output_dir='../output/classification_results/' + detection_model)

    if action == 'generate_test_results_from_detection_crops_on_fold':
        suffix = ''
        if args.hflip:
            suffix += '_hflip'
        if args.vflip:
            suffix += '_vflip'
        generate_results_from_detection_crops_on_fold(
            fold=0,
            weights=args.weights,
            crops_dir='../output/classification_crop_test/' + detection_model,
            output_dir='../output/classification_results_test/{}/{}/{}{}'.format(detection_model, classification_model, args.fold, suffix),
            video_ids=sorted(dataset.video_clips_test().keys()),
            hflip=args.hflip,
            vflip=args.vflip,
            model_type=classification_model
        )

    if action == 'save_detection_results':
        save_detection_results(detection_results_dir='../output/predictions_ssd_roi2/' + detection_model,
                               output_dir='../output/detection_results/' + detection_model)

    if action == 'save_detection_results_test':
        save_detection_results(detection_results_dir='../output/predictions_ssd_roi2_test/' + detection_model,
                               output_dir='../output/detection_results_test/' + detection_model)

    if action == 'combine_results_test':
        combine_test_results(classification_results_dir='../output/classification_results_test/{}/{}'.format(detection_model, classification_model),
                             output_dir='../output/classification_results_test_combined/{}/{}'.format(detection_model, classification_model))
