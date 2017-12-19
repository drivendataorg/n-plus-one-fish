import numpy as np
import skimage
import skimage.transform
import sys
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import concurrent.futures

from ssd.ssd_utils import BBoxUtility
from ssd.ssd_training import MultiboxLoss
from ssd.ssd import SSD300_BN
from ssd.ssd_resnet import SSDResnet50_BN

from scipy.misc import imread, imresize
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
from copy import copy

from multiprocessing.pool import ThreadPool
from typing import List, Dict

import pickle, os, random
import utils
import scipy.misc

import img_augmentation

import dataset
from dataset import SPECIES
import fish_detection


INPUT_ROWS = 360
INPUT_COLS = 720
input_shape = (INPUT_ROWS, INPUT_COLS, 3)
NUM_CLASSES = len(SPECIES)+1


def build_model(input_shape, num_classes=NUM_CLASSES, add_dropout=False):
    model = SSD300_BN(input_shape, num_classes=num_classes, add_dropout=add_dropout)
    model.load_weights('ssd/weights_SSD300.hdf5', by_name=True)

    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

    for L in model.layers:
        if L.name in freeze:
            L.trainable = False

    return model


def build_resnet(input_shape, num_classes=NUM_CLASSES):
    model = SSDResnet50_BN(input_shape, num_classes=num_classes)

    found_trainable = False
    for layer in model.layers:
        if layer.name == 'res3a_branch2a': # 'res5a_branch2a':
            found_trainable = True
        if found_trainable:
            layer.trainable = True
    return model


def priors_from_model(model):
    prior_box_layer_names = [
        'conv4_3_norm_mbox_priorbox',
        'fc7_mbox_priorbox',
        'conv6_2_mbox_priorbox',
        'conv7_2_mbox_priorbox',
        'conv8_2_mbox_priorbox',
        'pool6_mbox_priorbox']

    all_priors = []
    for prior_box_layer_name in prior_box_layer_names:
        layer = model.get_layer(prior_box_layer_name)
        if layer is not None:
            all_priors.append(layer.prior_boxes)

    all_priors = np.vstack(all_priors)
    return all_priors


def display_img_with_rects(img, results, res_idx=0, conf_threshold=0.1):
    if len(results[res_idx]) == 0:
        plt.imshow(img / 255.)
    else:
        det_label = results[res_idx][:, 0]
        det_conf = results[res_idx][:, 1]
        det_xmin = results[res_idx][:, 2]
        det_ymin = results[res_idx][:, 3]
        det_xmax = results[res_idx][:, 4]
        det_ymax = results[res_idx][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_threshold]
        # max_conf = max(0.7, max(det_conf))
        # top_indices_conf = sorted([(conf, i) for i, conf in enumerate(det_conf) if conf >= 0.1], reverse=True)
        # top_indices = [i for c, i in top_indices_conf[:8]]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        plt.imshow(img / 255.)
        currentAxis = plt.gca()

        label_text = ['_'] + dataset.SPECIES

        for i in reversed(range(top_conf.shape[0])):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            #         label_name = voc_classes[label - 1]
            display_txt = '{} {:0.2f}'.format(label_text[label], score)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1

            if i == 0 or score > 0.9:
                color = 'green'
            elif score > 0.6:
                color = 'yellow'
            elif score > 0.5:
                color = 'brown'
            else:
                color = 'gray'

            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})


class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self,
                 detection,
                 transformation,
                 saturation=0.5, contrast=0.5, brightness=0.5,  # 0.5  - no changes, range 0..1
                 blurred_by_downscaling=1,
                 hflip=False,
                 vflip=False):
        self.transformation = transformation
        self.detection = detection
        self.vflip = vflip
        self.hflip = hflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blurred_by_downscaling = blurred_by_downscaling

    def __lt__(self, other):
        return True

    def __str__(self):
        return dataset.CLASSES[self.detection.class_id] + ' ' + str(self.__dict__)


class SSDDataset(fish_detection.FishDetectionDataset):
    def __init__(self, bbox_util, preprocess_input=preprocess_input, is_test=False):
        super().__init__(is_test)
        self.bbox_util = bbox_util
        self.preprocess_input = preprocess_input

    def horizontal_flip(self, img, y):
        img = img[:, ::-1]
        if y.size:
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        img = img[::-1]
        if y.size:
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def generate_xy(self, cfg: SampleCfg):
        img = scipy.misc.imread(dataset.image_fn(cfg.detection.video_id, cfg.detection.frame, is_test=self.is_test))
        crop = skimage.transform.warp(img, cfg.transformation, mode='edge', order=3, output_shape=(INPUT_ROWS, INPUT_COLS))

        detection = cfg.detection

        if detection.class_id > 0:
            coords = np.array([[detection.x1, detection.y1], [detection.x2, detection.y2]])
            coords_in_crop = cfg.transformation.inverse(coords)
            aspect_ratio = dataset.ASPECT_RATIO_TABLE[dataset.CLASSES[detection.class_id]]
            coords_box0, coords_box1 = utils.bbox_for_line(coords_in_crop[0, :], coords_in_crop[1, :], aspect_ratio)
            coords_box0 /= np.array([INPUT_COLS, INPUT_ROWS])
            coords_box1 /= np.array([INPUT_COLS, INPUT_ROWS])
            targets = [coords_box0[0], coords_box0[1], coords_box1[0], coords_box1[1]]

            # print(detection.class_id, dataset.CLASSES[detection.class_id], aspect_ratio, coords_box0, coords_box1)

            cls = [0] * (NUM_CLASSES - 1)
            cls[detection.class_id-1] = 1
            targets = np.array([targets+cls])
        else:
            targets = np.array([])

        crop = crop.astype('float32')
        if cfg.saturation != 0.5:
            crop = img_augmentation.saturation(crop, variance=0.25, r=cfg.saturation)

        if cfg.contrast != 0.5:
            crop = img_augmentation.contrast(crop, variance=0.25, r=cfg.contrast)

        if cfg.brightness != 0.5:
            crop = img_augmentation.brightness(crop, variance=0.3, r=cfg.brightness)

        if cfg.hflip:
            crop, targets = self.horizontal_flip(crop, targets)

        if cfg.vflip:
            crop, targets = self.vertical_flip(crop, targets)

        crop = img_augmentation.blurred_by_downscaling(crop, 1.0/cfg.blurred_by_downscaling)

        return crop*255.0, targets

    def generate_x_from_precomputed_crop(self, cfg: SampleCfg):
        crop = scipy.misc.imread(dataset.image_crop_fn(cfg.detection.video_id, cfg.detection.frame, is_test=self.is_test))
        crop = crop.astype('float32')
        # print('crop max val:', np.max(crop))
        return crop

    def generate_ssd(self, batch_size, is_training, verbose=False, skip_assign_boxes=False, always_shuffle=False):
        pool = ThreadPool(processes=8)

        def rand_or_05():
            if random.random() > 0.5:
                return random.random()
            return 0.5

        detections = []  # type: List[fish_detection.FishDetection]
        if is_training:
            detections += sum([self.detections[video_id] for video_id in self.train_clips], [])
        else:
            detections += sum([self.detections[video_id] for video_id in self.test_clips], [])

        while True:
            points_random_shift = 0
            samples_to_process = []
            if is_training or always_shuffle:
                random.shuffle(detections)
                points_random_shift = 32

            for detection in detections:
                tform = self.transform_for_clip(detection.video_id,
                                                dst_w=INPUT_COLS, dst_h=INPUT_ROWS,
                                                points_random_shift=points_random_shift)
                cfg = SampleCfg(detection=detection, transformation=tform)

                if is_training:
                    cfg.contrast = rand_or_05()
                    cfg.brightness = rand_or_05()
                    cfg.saturation = rand_or_05()
                    cfg.hflip = random.choice([True, False])
                    cfg.vflip = random.choice([True, False])
                    cfg.blurred_by_downscaling = np.random.choice([1, 1, 1, 1, 2, 2.5, 3, 4])

                if verbose:
                    print(str(cfg))

                samples_to_process.append(cfg)

                if len(samples_to_process) >= batch_size:
                    inputs = []
                    targets = []
                    for img, y in pool.map(self.generate_xy, samples_to_process):
                    # for img, y in map(self.generate_xy, samples_to_process):
                        inputs.append(img)
                        if skip_assign_boxes:
                            targets.append(y)
                        else:
                            targets.append(self.bbox_util.assign_boxes(y))

                    tmp_inp = np.array(inputs)
                    inputs.clear()  # lets return some memory earlier
                    samples_to_process = []
                    x = self.preprocess_input(tmp_inp)
                    y = np.array(targets)

                    yield x, y

    def generate_x_for_train_video_id(self, video_id, batch_size, pool, frames=None):
        detections = []  # type: List[fish_detection.FishDetection]
        frames_to_use = frames if frames is not None else range(len(dataset.video_clips(is_test=self.is_test)[video_id]))
        for frame_id in frames_to_use:
            detections.append(
                fish_detection.FishDetection(
                    video_id=video_id,
                    frame=frame_id,
                    fish_number=0,
                    x1=np.nan, y1=np.nan,
                    x2=np.nan, y2=np.nan,
                    class_id=0
                )
            )

        def output_samples(samples_to_process):
            inputs = []
            for img in pool.map(self.generate_x_from_precomputed_crop, samples_to_process):
                inputs.append(img)

            frames = [cfg.detection.frame for cfg in samples_to_process]

            tmp_inp = np.array(inputs)
            inputs.clear()  # lets return some memory earlier
            x = self.preprocess_input(tmp_inp)
            return x, frames

        samples_to_process = []
        for detection in detections:
            cfg = SampleCfg(detection=detection, transformation=None)
            samples_to_process.append(cfg)

            if len(samples_to_process) >= batch_size:
                yield output_samples(samples_to_process)
                samples_to_process = []

        if len(samples_to_process) > 0:
            yield output_samples(samples_to_process)


def check_dataset():
    model = build_model(input_shape)
    model.compile(loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0, pos_cost_multiplier=1.2).compute_loss,
                  optimizer=Adam(lr=1e-4))
    priors = priors_from_model(model)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    dataset = SSDDataset(bbox_util=bbox_util, preprocess_input=lambda x: x)

    batch_size = 2
    for images, ys in dataset.generate_ssd(batch_size=batch_size, skip_assign_boxes=True, is_training=True, verbose=True):
        print('min value:', np.min(images[0]))
        print('max value:', np.max(images[0]))

        img = images[0]
        plt.imshow(img / 255.)
        currentAxis = plt.gca()
        for y in ys[0]:
            xmin = int(round(y[0] * img.shape[1]))
            ymin = int(round(y[1] * img.shape[0]))
            xmax = int(round(y[2] * img.shape[1]))
            ymax = int(round(y[3] * img.shape[0]))

            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = 'yellow'
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        plt.show()


def train():
    model_name = 'ssd_720_2'
    checkpoints_dir = '../output/checkpoints/detect_ssd/' + model_name
    tensorboard_dir = '../output/logs/detect_ssd/' + model_name
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    model = build_model(input_shape)
    model.load_weights('../output/checkpoints/detect_ssd/ssd_720_2/checkpoint-019-0.1299.hdf5')

    model.compile(loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0, pos_cost_multiplier=1.0).compute_loss,
                  optimizer=Adam(lr=4e-5))
    model.summary()

    priors = priors_from_model(model)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    dataset = SSDDataset(bbox_util=bbox_util, preprocess_input=preprocess_input)

    batch_size = 8
    val_batch_size = 8

    nb_epoch = 100

    checkpoint_best = ModelCheckpoint(checkpoints_dir + "/checkpoint-best-{epoch:03d}-{val_loss:.4f}.hdf5",
                                      verbose=1,
                                      save_weights_only=False,
                                      save_best_only=True)
    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5",
                                            verbose=1,
                                            save_weights_only=False,
                                            period=4)

    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=4, write_graph=True, write_images=True)

    model.fit_generator(dataset.generate_ssd(batch_size=batch_size, is_training=True),
                        steps_per_epoch=dataset.nb_train_samples // batch_size,
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=[checkpoint_best, checkpoint_periodical, tensorboard],
                        validation_data=dataset.generate_ssd(batch_size=val_batch_size, is_training=False),
                        validation_steps=dataset.nb_test_samples // val_batch_size,
                        initial_epoch=20)


def train_resnet():
    model_name = 'ssd_resnet_720'
    checkpoints_dir = '../output/checkpoints/detect_ssd/' + model_name
    tensorboard_dir = '../output/logs/detect_ssd/' + model_name
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    model = build_resnet(input_shape=input_shape)
    model.compile(loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0, pos_cost_multiplier=1.0).compute_loss,
                  optimizer=Adam(lr=3e-5))
    model.summary()
    # model.load_weights('../output/checkpoints/detect_ssd/ssd_resnet_720/checkpoint-best-018-0.2318.hdf5')
    # model.load_weights('../output/checkpoints/detect_ssd/ssd_resnet_720/checkpoint-best-053-0.1058.hdf5')

    priors = priors_from_model(model)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    dataset = SSDDataset(bbox_util=bbox_util, preprocess_input=preprocess_input)

    batch_size = 8
    val_batch_size = 8

    nb_epoch = 100

    checkpoint_best = ModelCheckpoint(checkpoints_dir + "/checkpoint-best-{epoch:03d}-{val_loss:.4f}.hdf5",
                                      verbose=1,
                                      save_weights_only=False,
                                      save_best_only=True)
    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5",
                                            verbose=1,
                                            save_weights_only=False,
                                            period=2)

    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=16, write_graph=False, write_images=False)

    model.fit_generator(dataset.generate_ssd(batch_size=batch_size, is_training=True),
                        steps_per_epoch=dataset.nb_train_samples // batch_size,
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=[checkpoint_best, checkpoint_periodical, tensorboard],
                        validation_data=dataset.generate_ssd(batch_size=val_batch_size, is_training=False),
                        validation_steps=dataset.nb_test_samples // val_batch_size,
                        initial_epoch=0)


def check(weights):
    model = build_model(input_shape)

    model.compile(loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0, pos_cost_multiplier=1.0).compute_loss,
                  optimizer=Adam(lr=1e-3))
    model.load_weights(weights)
    model.summary()

    priors = priors_from_model(model)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)
    dataset = SSDDataset(bbox_util=bbox_util)

    for x_batch, y_batch in dataset.generate_ssd(batch_size=4, is_training=False, verbose=True, always_shuffle=True):
        predictions = model.predict(x_batch)
        results = bbox_util.detection_out(predictions)
        for batch_id in range(4):
            display_img_with_rects(img=utils.preprocessed_input_to_img_resnet(x_batch[batch_id])*255,
                                   results=results,
                                   res_idx=batch_id)
            plt.show()
    #
    # train_id = 0
    # img = dataset.load_img(0)
    # y = dataset.generate_gt(train_id, img)
    #
    # img2, y2 = dataset.get_crop(img, y, 2500, 2500, crop_size, crop_size)
    # img2 = img2.astype(np.float32)
    # print(img2.dtype)
    # print(img2.shape)
    #
    # preds = model.predict(preprocess_input(np.array([img2])))
    # results = bbox_util.detection_out(preds)
    # display_img_with_rects(img2, results)
    # plt.show()


def check_on_train_clip(video_id, weights, suffix, is_test=False):
    if 'resnet' in weights:
        model = build_resnet(input_shape)
    else:
        model = build_model(input_shape)

    pool = ThreadPool(processes=8)

    model.compile(loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0, pos_cost_multiplier=1.0).compute_loss,
                  optimizer=Adam(lr=1e-3))
    model.load_weights(weights)
    model.summary()

    priors = priors_from_model(model)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)
    dataset = SSDDataset(bbox_util=bbox_util, is_test=is_test)

    outdir = '../output/predictions_ssd/' + video_id + suffix
    os.makedirs(outdir, exist_ok=True)
    batch_size = 4

    frame_id = 0
    for x_batch, frames in dataset.generate_x_for_train_video_id(video_id=video_id, batch_size=batch_size, pool=pool):
        predictions = model.predict(x_batch)
        results = bbox_util.detection_out(predictions)
        for batch_id in range(predictions.shape[0]):
            print(results[batch_id])
            display_img_with_rects(img=utils.preprocessed_input_to_img_resnet(x_batch[batch_id]) * 255,
                                   results=results,
                                   res_idx=batch_id)
            plt.savefig('{}/{:04}.jpg'.format(outdir, frame_id+1))
            plt.clf()
            frame_id += 1
            print(frame_id)


def generate_predictions_on_train_clips(weights, suffix, from_idx, count, use_requested_frames=False, is_test=False):
    if 'resnet' in weights:
        model = build_resnet(input_shape)
    else:
        model = build_model(input_shape)

    model.compile(loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0, pos_cost_multiplier=1.0).compute_loss,
                  optimizer=Adam(lr=1e-3))
    model.load_weights(weights)
    model.summary()

    priors = priors_from_model(model)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)
    dataset = SSDDataset(bbox_util=bbox_util, is_test=is_test)

    items = list(sorted(dataset.video_clips.keys()))

    pool = ThreadPool(processes=4)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    for video_id in items[from_idx: from_idx+count]:
        print(video_id)
        if is_test:
            outdir = '../output/predictions_ssd_roi2_test/{}/{}'.format(suffix, video_id)
        else:
            outdir = '../output/predictions_ssd_roi2/{}/{}'.format(suffix, video_id)
        os.makedirs(outdir, exist_ok=True)
        batch_size = 4

        if use_requested_frames:
            requested_frames = pickle.load(open('../output/used_frames.pkl', 'rb'))
            frames = requested_frames[video_id]
        else:
            frames = list(range(len(dataset.video_clips[video_id])))

        new_frames = []
        for frame in frames:
            if not os.path.exists('{}/{:04}.npy'.format(outdir, frame+1)):
                new_frames.append(frame)

        if len(new_frames) == 0:
            continue

        for x_batch, used_frames in utils.parallel_generator(dataset.generate_x_for_train_video_id(video_id=video_id,
                                                                          batch_size=batch_size,
                                                                          frames=new_frames,
                                                                          pool=pool), executor=executor):
            predictions = model.predict(x_batch)
            results = bbox_util.detection_out(predictions)
            for batch_id in range(predictions.shape[0]):
                np.save('{}/{:04}.npy'.format(outdir, used_frames[batch_id]+1), results[batch_id])
                print(used_frames[batch_id])


if __name__ == '__main__':
    action = sys.argv[1]

    if action == 'train':
        train()
    if action == 'train_resnet':
        train_resnet()
    elif action == 'check':
        check(weights=sys.argv[2])
    elif action == 'check_on_train_clip':
        check_on_train_clip(video_id=sys.argv[2], weights=sys.argv[3], suffix=sys.argv[4])
    elif action == 'check_on_test_clip':
        check_on_train_clip(video_id=sys.argv[2], weights=sys.argv[3], suffix=sys.argv[4], is_test=True)
    elif action == 'generate_predictions_on_train_clips':
        generate_predictions_on_train_clips(weights=sys.argv[2], suffix=sys.argv[3], from_idx=int(sys.argv[4]), count=int(sys.argv[5]))
    elif action == 'generate_predictions_on_test_clips':
        generate_predictions_on_train_clips(weights=sys.argv[2], suffix=sys.argv[3], from_idx=int(sys.argv[4]), count=int(sys.argv[5]), is_test=True)
    elif action == 'check_dataset':
        check_dataset()


