Submission reproducing exactly the prediction result.

Requrements:
Linux with GTX1080ti cards, CUDA and related libraries installed of version 8
Python 3.5 used with requirements listed in src/requirements.txt
ffmpeg application
for building labeling helper application, Qt5-dev packages necessary with dependencies (g++, etc)


Directory structure:
input
    input dataset
src
    sources directory
output
    outputs, checkpoints, resulting submission

datasets for extra labels:
input/train_videos/img/00WK7DR6FyPZ5u3A/masks
    ruler masks
output/ruler_crops_batch_labeled
    extra labels for "no fish", "fish clear", "fish covered" categories

fish_select2: application for labeling masks


Preparing input data:

it's necessary to extract train and test videos to train_videos and test_videos directories
and run convert_all.sh from each directory to extract jpeg frames

cd input/train_videos
bash ../convert_all.sh
cd ../test_videos
bash ../convert_all.sh


Reproducibg the submission:

cd src
bash predict.sh

