import os
import sys
import shutil

DATA_DIR = '../output/ruler_crops_batch_labeled/500'


for video_id in os.listdir(DATA_DIR):
    video_dir = os.path.join(DATA_DIR, video_id)
    for frame in os.listdir(video_dir):
        if frame.endswith('.jpg'):
            cat = open(os.path.join(video_dir, frame.replace('jpg', 'txt')), 'r').read()
            print(frame, cat)
            dst_dir = os.path.join(DATA_DIR, cat)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(os.path.join(video_dir, frame), os.path.join(dst_dir, video_id+"_"+frame))
