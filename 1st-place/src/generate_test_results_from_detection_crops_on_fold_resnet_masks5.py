import os
import sys

folds = [int(f) for f in sys.argv[1].split(',')]

print('folds:', folds)

weights = {
    1: '../output/checkpoints/classification/model_resnet50_mask_fold_1/checkpoint-005-0.1539.hdf5',
    2: '../output/checkpoints/classification/model_resnet50_mask_fold_2/checkpoint-005-0.1587.hdf5',
    3: '../output/checkpoints/classification/model_resnet50_mask_fold_3/checkpoint-005-0.1687.hdf5',
    4: '../output/checkpoints/classification/model_resnet50_mask_fold_4/checkpoint-005-0.1621.hdf5'
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
                        classification_model='resnet50_mask5'
                    )
            print(cmd)
            os.system(cmd)

