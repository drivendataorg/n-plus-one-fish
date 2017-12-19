import os
import sys

folds = [int(f) for f in sys.argv[1].split(',')]

print('folds:', folds)

weights = {
    1: '../output/checkpoints/classification/model_xception_fold_1/checkpoint-014-0.1049.hdf5',
    2: '../output/checkpoints/classification/model_xception_fold_2/checkpoint-013-0.1334.hdf5',
    3: '../output/checkpoints/classification/model_xception_fold_3/checkpoint-007-0.1556.hdf5',
    4: '../output/checkpoints/classification/model_xception_fold_4/checkpoint-010-0.1467.hdf5'
}

flips = ['', '--hflip=1', '--vflip=1', '--hflip=1 --vflip=1']

detection_models = ['resnet_62', 'resnet_53']


for flip in flips:
    for detection_model in detection_models:
        for fold in folds:
            print('\n' * 4)
            print(flip, detection_model, fold)
            print('\n' * 4)

            cmd = ('python3 fish_classification.py generate_test_results_from_detection_crops_on_fold --fold {fold}'
                   ' {flip} --detection_model {detection_model} --weights {weights} '
                   ' --classification_model={classification_model}').format(
                        fold=fold,
                        flip=flip,
                        detection_model=detection_model,
                        weights=weights[fold],
                        classification_model='xception'
                    )
            print(cmd)
            os.system(cmd)

