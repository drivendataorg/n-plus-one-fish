Solution summary
----------------
To find sequences of frames which potentially contain a fish, being placed on a ruler, a recurrent network were designed. On such sequences for each frame the 8 class (7 species + background) classifier is appied to the whole frame (global) as well as another 8 class classifer is applied to the fish region only (local). Fish region is obtained as the most confident detection region provided by the Single Shot Multi Box detector (SSD). Sequneces with potential fish, given by recurrent network, can be discarded if fused species score from global and local classifiers is too low. The sequences which passed this check treated as sequences with unique fish, thus recived the unique fish id. Species are obtained from local classifier scores, and length is calculated from SSD region.

Prerequisites:
--------------
### What need to be installed:
* Ubuntu 16.04 (other versions may work also, but not tested)
* CMake 3.3 (other versions should work also, but not tested), https://cmake.org/download/
* Cuda 8, install instructions https://developer.nvidia.com/cuda-80-download-archive
* OpenCV 3.2 (other versions should work also, but not tested), source: https://sourceforge.net/projects/opencvlibrary/files/opencv-unix/3.2.0/, build instructions: https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html
* Caffe with SSD, source: https://github.com/weiliu89/caffe/tree/ssd, build instructions: http://caffe.berkeleyvision.org/install_apt.html (can be built with CMake)
  * OpenBLAS is used
  * Will use `lmdb` containers

### External data:
* VGG-16: https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
* VGG reduced: https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6
* Densenet 201: https://drive.google.com/file/d/0B7ubpZO7HnlCV3pud2oyR3lNMWs/view
* Mobilenet: https://github.com/shicai/MobileNet-Caffe
* PVA: https://drive.google.com/file/d/0Bw_6VpHzQoMVTjctVVhjMXo1X3c/view
* ResNet 10: https://github.com/cvjena/cnn-models/tree/master/ResNet_preact/ResNet10_cvgj

Folders structure
----------------
There are **4** major stages of pipeline, each splits in 2 subparts: dataset creation and training model based on this data:
1. Obtaining sequence of frames with interesting action: placing individual fish on a ruler:
  1 Dataset creation - folder `0_make_sequence_dataset`
  2 Training model for separation frames with interesting action from the rest frames - folder `1_train_sequence_model`
2. Full frame species classification:
  1 Dataset creation - folder `2_make_cls_global_frame_dataset`
  2 Training model for 8 class (7 species + background) full frame classification - folder `3_train_cls_global_frame_model`
3. Localization of precise fish bounding box in a frame:
  1 Dataset creation - folder `4_make_loc_dataset`
  2 Training model for fish bounding box localization in a given frame - folder `5_train_loc_model`
4. Species classification for potential bounding box with fish:
  1 Dataset creation - folder `6_make_cls_roi_dataset`
  2 Training model for fish boundign box species classification - folder `7_train_cls_roi_model`

When all models are trained, one can run a full pipeline with `full_pipeline_inference.cpp`. It performs inference for the all models above and produces a file (let's call it `baseline.txt`) in a submission-like format, but with frames which contain fish only, so no frames with zero species probabilities included in it. The final submission consists of aggregation (done by `cls_roi_aggregate.cpp`) of multiple such files, which differs in a species classifier inside fish bounding box (described in stage 4). For speed reasons `inference_cls_roi.cpp` is made, which takes `baseline.txt` and performs relocalization and reclassification (stage 3 and 4 correspondingly) for the frames in this file. After aggregation results cleaned by `cls_roi_cleanup.cpp` and species scores are binarized (for MSE gives slightly higher metric value, than probabilities like 0.9) with `cls_roi_binarize.cpp`. Final output prcessed with `make_submission.cpp` to satisfy submission format (adds frames with zeros species probabilities).

To build all necessary executables just create new directory, say `build`, open terminal, run `cd <path_to>/build`, then run `cmake <path_to_submission_folder>` (if Caffe and/or OpenCV wasn't found, rerun cmake with paths to their build folders, like `cmake -DCaffe_DIR=<path_to_caffe_build_dir> -DOpenCV_DIR=<path_to_opencv_build_dir> <path_to_submission_folder>`), then run `make`.

Result reproduction
-------------------
#### Stage 1. Training classifier for finding sequence of frames with interesting action: placing individual fish on a ruler.
##### Description:
We want to train classifier which able to discriminate between sequences of frames with interesting action: placing individual fish on a ruler, and other possible frame sequences. So, train dataset consists of positives - sequences with interesting action, each frame in these sequences has label 1, and negatives sequences without interesting action, which frames have label 0. Single positive - sequence of N (N == sequence length, we end up with N == 5, but started from N == 11) consecutive frames, which has annotated frame with fish in its center. Single negative - sequence like positive, but has annotated frame without any fish present in its center. Natural way to work with sequences - use recurrent network. We built one for that: vgg 16 is used to embed the information from visual frame into feature vector, which passed to lstm layer, then 2 class classification is performed.
##### Prerequisites:
We wrote special layer to read the sequences, so please copy `1_train_sequence_model/resd_seq_layer.hpp` to `<caffe_root>/include/caffe/layers/`, `1_train_sequence_model/resd_seq_layer.cpp` to `<caffe_root>/src/caffe/layers/` and rebuild caffe **with the new layers** (if caffe built with CMake, new layers won't be included automatically until cmake will regenerate project structure. One way to do this, for example, is to remove then add ')' on line 72 of root CMakeLists.txt of caffe and save changes, cmake will notice, that it's file changed and will regenerate project structure, so one should see `resd_seq_layer` in the `make`'s output during rebuilding caffe. If this didn't help, just rebuild caffe in new clean directory.).
##### Commands:
1. Create the list of sequences for train dataset (with balanced positive and negative number):
`./parse_sequence_dataset training.csv <path_to_train_videos_folder>/`. Outputs `sequences_11.txt` with list sequences (one training sample per line, our result is `0_make_sequence_dataset/artifacts/sequences_11.txt`).
2. Write sequence' frames and corresponding labels file:
`./make_sequence_dataset sequences_11.txt <path_to_train_videos_folder>`. Outputs sequences images and `labels.txt` with 0/1 labels for corresponding frames. We include example of two positive sequences in `0_make_sequence_dataset\artifacts\`.
3. Training the model. Please set proper paths to folder with sequence images and their labels (obtained from `make_sequence_dataset`) in `1_train_sequence_model/trainval5.prototxt` (our train/test images and labels files is in `1_train_sequence_model`). Run caffe training: `./caffe train -gpu 0 -solver <path_to>/1_train_sequence_model/solver.prototxt -weights <paht_to_vgg_reduced>.caffemodel`. We left for training 6000 samples and 2016 for testing, trained till acuracy more than 90%, 1200 iterations snapshot in our case with batch size 16 (4 * (iter_size == 4)). Let's call result model `sequences.caffemodel`.

#### Stage 2. Training global full frame species classifier.
##### Description:
Once we obtain potential sequences with fish, next step is to classify it's species. We train densenet based classifier, which outputs 8 class probability vector: 7 species + background for the full frame. We want to use augmentation of SSD's AnnotatedData layer, which takes lmdb container with images and corresponding xml file in VOC format with object regions, so input data prepared in a special way - we add a mock xml for each image.
##### Prerequisites:
For classification task we add aggressive augmentation - rotations (90, 180, 270 degrees), infrared-like distortion, random shadow stripes placing on input data. So, please, copy with replace modified `3_train_cls_global_frame_model/annotated_data_layer.cpp` to `<caffe_root>/src/caffe/layers/` and `3_train_cls_global_frame_model/data_transformer.hpp` to `<caffe_root>/include/caffe/` and rebuild (run `make` in caffe build folder). **NOTE!** This changes are valid for stage 2 and 4, for stage 3 please leave original caffe's files as is.
##### Commands:
1. Make two directories: run in terminal `mkdir images`, `mkdir annotations` - in these folders images and corresponding annotations will be placed.
2. Run `./make_global_cls_dataset training.csv <path_to_train_videos_folder>/`. We've taken ~2000 random samples for each class to balance dataset, end up with ~15000 samples for train and 1800 samples for test (corresponding train.txt, test.txt in `2_make_cls_global_frame_dataset/artifacts`, we also include two images and corresponding mock annotations examples).
3. To create lmdb file with train/test base the script `https://github.com/weiliu89/caffe/blob/ssd/data/VOC0712/create_data.sh` need to be ran, please adjust path to proper folder with `images`, `annotations` folders (replace `$data_root_dir` in line 24) and path to train.txt, test.txt images (replace `$root_dir/data/$dataset_name/$subset.txt` in line 24) correspondingly. Train and test txt files have to be formatted like ones in `2_make_cls_global_frame_dataset/artifacts` (image then corresponding annotation file per line with folder prefix), better to have shuffled lines in these files to have balanced by species batches in training.
4. Please adjust paths to created lmdb files in `3_train_cls_global_frame_model/trainval_adl.prototxt`, labelmap file and run training `./caffe train -gpu 0 -solver <path_to>/3_train_cls_global_frame_model/solver.prototxt -weights <path_to_densenete_201>.caffemodel`. We trained till accuracy more than 90%, 2500 iterations with batch size 32 in our case. Let's call result model `global_cls.caffemodel`.

#### Stage 3. Train fish detector to localize boxes.
##### Description:
Finding length of the fish is done by detection the fish, then calculation the length from fish bounding box. Detection is performed by Single Shot Multibox Detector. It outputs box with class confidence (we discard this confidence), based on box dimensions length is calculated: box like square - take the length of diagonal as fish length, box height significantly more than width - take height as fish length, or take width overwise. To train SSD bounding boxes is needed, they obtained through grabcut.
##### Prerequisites:
Please stash changes from stage 2 in `data_transformer.hpp` and `annotated_data_layer.cpp` and rebuild.
##### Commands:
1. Make two directories: run in terminal `mkdir images`, `mkdir annotations` - in these folders images and corresponding annotations will be placed.
2. Run `./make_loc_dataset training.csv <path_to_train_videos_folder>/ 4_make_loc_dataset/artifacts/video_list.txt`. We selected manually some videos where grabcut is expected to work (excluded overlighted videos) and wrote this list to video_list.txt, one train video name without extension per line (our result is in `4_make_loc_dataset/artifacts`).
3. Next the train/test lmdb need to be created, so please follow the step 3 of stage 2 with substitution paths to proper ones. Our train.txt and test.txt in `5_train_loc_dataset/artifacts/`. Don't forget to have shuffled lines in these files to have balanced by species batches in training.
4. Please adjust paths to created lmdb files in `5_train_loc_model/train.prototxt` and `5_train_loc_model/test.prototxt`, labelmap file (same with stage 3) and test_name_size file from `5_train_loc_model/test_name_size.txt`. Run training `./caffe train -gpu 0 -solver <path_to>/5_train_loc_model/solver.prototxt -weights <path_to_vgg_reduced>.caffemodel`. We trained till accuracy more than 90%, 1500 iterations with batch size 32 in our case. Let's call result model `localization.caffemodel`.

#### Stage 4. Train box (local) species classifier.
##### Description:
Once precise fish location is obtained from SSD, we can refine species confidence with local classifer, which outputs 8 class probability vector: 7 species + background for fish bounding box only. This stage is mostly the same with stage 2, but operates on local image patches with fish, uses vgg 16 as a backbone.
##### Prerequisites:
Same as for the 2nd stage.
##### Commands:
1. Make two directories: run in terminal `mkdir images`, `mkdir annotations` - in these folders images and corresponding annotations will be placed.
2. Open terminal and change directory to stage 2 images folder (we will extract fish bounding box from the same images we trained global classifier on) and run `ls *.jpg > imgs.txt` in it to create list of files to extract fish boxes from.
3. Run `./make_cls_roi_dataset -m <path_to>/localization.caffemodel -d <path_to>/5_train_loc_model/deploy.prototxt -i <path_to_stage_2_images_folder>/ -o tmp.txt`.
4. Next the train/test lmdb need to be created, so please follow the step 3 of stage 2 with substitution paths to proper ones. Our train.txt and test.txt in `2_make_cls_global_frame_dataset/artifacts/`. Don't forget to have shuffled lines in these files to have balanced by species batches in training.
5. Please adjust paths to created lmdb files in `7_train_cls_roi_model/trainval_vgg_adl_roi.prototxt`, label_map file and run training `./caffe train -gpu 0 -solver <path_to>/7_train_cls_roi_model/solver.prototxt -weights <path_to_vgg_16>.caffemodel`. We trained till accuracy more than 90%, 2000 iterations with batch size 32 in our case. Let's call result model `cls_roi.caffemodel`.

#### Stage 5. Putting it all togeter, running the pipeline.
##### Commands:
The whole pipeline is in `full_pipeline_inference.cpp`. Please adjust paths to corresponding models, deploys and `test_videos` folder (substitute `<path_to>` to real paths on machine, 9 places to modify at all) and rebuild it. Run `./full_pipeline_inference -m <path_to>/localization.caffemodel -d <path_to>/5_train_loc_model/deploy.prototxt -i <path_to>/test_videos_list_file.txt -o tmp.txt > baseline.txt`. It will process test videos one by one, output results in console, so we forward output to baseline.txt. Then run `./cls_roi_cleanup baseline.txt > baseline_cleaned.txt` to remove unconfident sequences. Then run `./cls_roi_binarize baseline_cleaned.txt > baseline_binarized.txt` to binarize species scores, since it gives higher MSE value. The result `baseline_binarized.txt` is passed to make submission to satisfy submission format - add frames without any species present: `./make_submission submission_format_zeros.csv baseline_binarized.txt`. It produces `result3.csv` which can be used for submission and expected to give 2-3% less score than final one. Final score is obtained by aggregating results from multiple box (local) species classifiers.
##### Obtaining final score:
1. To train more local (box) classifiers, please repeat step 5 of stage 4 with different backbones: mobilenet, pva, resnet 10. Corresponding trainvals in `7_train_cls_roi_model`, all trained till more than 90% accuracy, usually 3000-5000 iterations with batch size 32. To train mobilenet fast `ConvolutionDepthwise` layer was used, please download it from https://github.com/farmingyard/caffe-mobilenet, copy to proper caffe locations (as did for `read_seq_layer`) and rebuild (don't forget to fake-modify CMakeLists to make it regenerate project structure and include new layer to build, as did for `read_seq_layer`).
2. To obtain results with new local species classifiers the full pipeline can be ran again with corresponding paths substituted. For speed reasons there is `inference_cls_roi.cpp`, which run localization and box classification parts of pipeline only, so just modify paths to caffemodel, deploy and test_videos folder, rebuild and run: `./inference_cls_roi -m <path_to>/localization.caffemodel -d <path_to>/5_train_loc_model/deploy.prototxt -i <path_to>/baseline.txt -o tmp.txt > baseline_<backbone_name>.txt`. **Note** mobilenet subtracts mean pixel and scales input, pva just subtracts mean, and resnet10 takes input as is, so modify lines 257-258 corresponding to each backbone and rebuild before switching backbone.
3. After all one end up with 4 files: baseline.txt, baseline_mobilenet.txt, baseline_pva.txt, baseline_resnet10.txt. Their scores aggregated by `cls_roi_aggregate.cpp`, run: `./cls_roi_aggregate baseline.txt baseline_mobilenet.txt baseline_pva.txt baseline_resnet10.txt > baseline_aggregated.txt`, then clean sequences, binarize scores and make final submission file as did before with `baseline.txt`.

That's it! The core is stage 1 and 2. Their result (which was submitted as a report for Judges' Choice Award) gives ~60% on leaderboard, last 10% is sophisticated classifiers.
