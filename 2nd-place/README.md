# DrivenData: Identify Fish Challenge (2nd Place Solution)
- Solution for N+1 fish, N+2 fish DrivenData competition (2nd place)
- Problem description: https://www.drivendata.org/competitions/48/identify-fish-challenge/page/93/
- Final leaderboard: https://www.drivendata.org/competitions/48/identify-fish-challenge/leaderboard/

## Visualisation [Youtube]

[![DrivenData: Identify Fish Challenge](https://github.com/ZFTurbo/DrivenData-Identify-Fish-Challenge-2nd-Place-Solution/blob/master/images/youtube-thumb.jpg)](http://www.youtube.com/watch?v=OlDPPF_0lWY "DrivenData: Competition N+1 fish, N+2 fish")

## Software Requirements

- **Main requirements**: Python 3.4+, keras 1.2.1, theano 0.8.2
- **Other requirements**: numpy 1.13.1+, pandas 0.20.3+, opencv-python 3.1.0+, scipy 0.19.1+, sklearn 0.18.1+, 

### Notes:
- Usage of Keras 1.2 is important, since there were many syntax changes in Keras 2.0. It probably will work in 2.0 with some warnings, but I didn't check it.
- Code is written for Theano backend. Usage with Tensorflow will require some changes considering shape of matrices.
- Code was developed in Microsoft Windows 10, but should work fine in Linux as well.

## Hardware requirements

All batch sizes for Neural nets are tuned to be used on NVIDIA GTX 1080 Ti 11 GB card. To use code with other GPUs with less memory - decrease batch size accordingly (function get_batch_size).

## How to run:

All r*.py files must be run one by one. All intermediate folders will be created automatically.
```
python r10_prepare_train_test.py
python r20_train_unet_for_roi_detection.py
python r30_create_keras_models_for_fish_classification_densenet121.py
python r30_create_keras_models_for_fish_classification_inception_v3.py
python r30_create_keras_models_for_fish_classification_resnet50.py
python r31_get_roi_for_video.py
python r32_clasterize_videos.py
python r33_process_data_with_densenet.py
python r33_process_data_with_inception.py
python r33_process_data_with_resnet.py
python r34_get_length_from_roi.py
python r35_fish_exist_predict_with_gbm.py
python r36_fish_type_predict_with_gbm.py
python r37_fish_length_predict_with_gbm.py
python r40_create_csv_from_predictions.py
python r50_merge_csvs_in_final_submission.py
```

**Optional**:
```
python r45_validation_on_csvs_v1.py
python r46_validation_on_csvs_v2.py
python r60_create_debug_videos.py
```

### Notes about a code

1) Training of neural networks can be done in parallel. There are 4 networks in total: 1 for segmentation/localization and 3 for classification. And I used 5KFold split, so 5 models for each net. In extreme you can use 20 GPUs to train in parallel with x20 speedup. This code can be run in parallel: r20_train_unet_for_roi_detection.py, r30_create_keras_models_for_fish_classification_densenet121.py, r30_create_keras_models_for_fish_classification_inception_v3.py, r30_create_keras_models_for_fish_classification_resnet50.py.
2) The same can be done with inference process: r33_process_data_with_densenet.py, r33_process_data_with_inception.py, r33_process_data_with_resnet.py
3) The accuracy of method is good enough with only one classification network DenseNet121. Two others: Inception v3 and ResNet50 add around ~0.02-~0.03 to LB score. So for speed up you can skip training of them.
4) In the header of each code file you can find some notes about its functionality

## Description of files and folders

```
-- DrivenData-Identify-Fish-Challenge-2nd-Place-Solution - folder with solution code
---- r10_prepare_train_test.py - Fix column name in training from 'species_grey sole' to 'species_grey_sole'. Extract frames from video files according to 'training.csv' and create masks
---- r20_train_unet_for_roi_detection.py - This code train 5KFold modified UNET neural network for segmentation of fishes (e.g. find exact fish location). It requires 2-3 days to complete. Can be run in parallel for 5 Folds on 5 GPUs. You can skip this part if you already have models with name 'ZF_UNET_1280_720_V2_SINGLE_OUTPUT_SMALL_fold_*.h5' in ../models/ directory.
---- r30_create_keras_models_for_fish_classification_densenet121.py - This code train 5KFold models for classification of fishes (e.g. find type of fish by crop from frame). Neural net based on pretraned DenseNet121 is used. Pretrained weights located in ../weights/ folder. It train faster than UNET. ~1 day to complete. Can be run in parallel for 5 Folds on 5 GPUs (use FOLD_TO_CALC constant). You can skip this part if you already have models with name 'DENSENET_121_fold_*.h5' in ../models/ directory.
---- r30_create_keras_models_for_fish_classification_inception_v3.py - same as previous, but for neural net based on pretraned Inception v3
---- r30_create_keras_models_for_fish_classification_resnet50.py - same as previous, but for neural net based on pretraned ResNet50
---- r31_get_roi_for_video.py - Extract region of interest for all videos based on UNET predictions. Find bounding boxes (which contains fish) for all videos.
---- r32_clasterize_videos.py - Split boats on different types (assign Boat ID for each video)
---- r33_process_data_with_densenet.py - Create prediction with probabilities about type of fish or "no fish" for each frame of each video. This file uses DenseNet121 model for inference. Predictions are cached in separate folder.
---- r33_process_data_with_inception.py - same as previous, but for neural net based on Inception v3
---- r33_process_data_with_resnet.py - same as previous, but for neural net based on ResNet50
---- r34_get_length_from_roi.py - extract length of fish for each frame, based on ROI matrices, produced by UNET neural network
---- r35_fish_exist_predict_with_gbm.py - Tries to predict using XGBoost if given frame have fish or not. Works better than heuristic algorithm. Code uses data about current frame and 7 previous and 7 next frames. Features for XGboost created from predictions of neural nets. It probably would be better to rewrite code with LightGBM instead of XGBoost to increase speed.
---- r36_fish_type_predict_with_gbm - Tries to predict using XGBoost type of fish for given frame. Works better than just average of predictions from CNNs. It uses data about current frame and 3 previous and 3 next frames.
---- r37_fish_length_predict_with_gbm.py - The same as above, but tries to predict length of fish. Works worse than naive algorithm, so doesn't used in current pipeline.
---- r40_create_csv_from_predictions.py - Join all the obtained data and create independent CSV files for each video from train and test.
---- r50_merge_csvs_in_final_submission.py - Merge all CSVs in single submission file
---- r45_validation_on_csvs_v1.py - Validation v1. Try to predict score. Doesn't work as expected. Always predicts more optimistic values. Probably because of non-random train/test split.
---- r46_validation_on_csvs_v2.py - Validation v1. Based on code provided on forum. Same problem as above.
---- r60_create_debug_videos.py - Provide functions to create visualisation of predictions directly on train/test videos. It was used to generate video: https://www.youtube.com/watch?v=OlDPPF_0lWY
---- a00_augmentation_functions.py - set of functions for image augmentation
---- a00_common_functions.py - set of helper functions which are common across other code files
---- a00_custom_layers.py - additional layers for DenseNet121 neural net
---- a01_densenet_121.py - DenseNet121 neural net
---- a02_zf_unet_model.py - modified UNET neural net model
---- a02_zoo.py - all classification neural net models and helper functions for training and inference.
---- README.md  - this file
-- models - weights for Neural nets which I obtained on my local computer. You can use them to skip training of neural nets from scratch.
-- weights - open source pretrained weights for DenseNet121: https://github.com/flyyufelix/DenseNet-Keras. Weights were obtained on large dataset: ImageNet.
```

## Initial weights and pretrained models
- Weights for DenseNet121 and pretrained models available by link (~2.7 GB): https://mega.nz/#!mQ5QSToR!NlgI6BIjdKD9DOmMnDqSqD8T_yTk33H8LMyaqdXnQCY

## Dataflow
![Dataflow](https://github.com/ZFTurbo/DrivenData-Identify-Fish-Challenge-2nd-Place-Solution/blob/master/images/Dataflow.png)