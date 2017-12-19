# find rulers

# predict ruler masks, IndexError at the end is expected
time python3 ruler_masks.py predict_test
time python3 ruler_masks.py find_ruler_angles_test
time python3 ruler_masks.py find_ruler_vectors_test

# generate crops based on ruler vectors, saved to output/ruler_crops_test
time python3 ruler_masks.py generate_crops_test

# resnet_53 - many covered crops, useful for next stage training
# resnet_62 - less covered crops, may be useful for sequence predicting

# detect fish using SSD detector

python3 fish_detection_ssd.py generate_predictions_on_test_clips ../output/checkpoints/detect_ssd/ssd_resnet_720/checkpoint-053-0.1058.hdf5 resnet_53 0 700
python3 fish_detection_ssd.py generate_predictions_on_test_clips ../output/checkpoints/detect_ssd/ssd_resnet_720/checkpoint-best-062-0.0635.hdf5 resnet_62 0 700

# generate grops with fish used by classification networks

python3 fish_classification.py generate_test_classification_crops --detection_model resnet_53
python3 fish_classification.py generate_test_classification_crops --detection_model resnet_62


# may be slow worth to run on multiple GPUs
python3 generate_test_results_from_detection_crops_on_fold_densenet.py 1,2,3,4
python3 generate_test_results_from_detection_crops_on_fold_inception.py 1,2,3,4


python3 fish_classification.py combine_results_test --detection_model resnet_53 --classification_model densenet
python3 fish_classification.py combine_results_test --detection_model resnet_62 --classification_model densenet

python3 fish_classification.py combine_results_test --detection_model resnet_53 --classification_model inception
python3 fish_classification.py combine_results_test --detection_model resnet_62 --classification_model inception

python3 fish_classification.py save_detection_results_test --detection_model resnet_53
python3 fish_classification.py save_detection_results_test --detection_model resnet_62

python3 fish_sequence_rnn.py predict_test

python3 prepare_submission.py

