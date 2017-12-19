[<img src='https://community.drivendata.org/uploads/default/optimized/1X/e055d38472b1ae95f54110375180ceb4449c026b_1_690x111.png'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://s3.amazonaws.com:443/drivendata/comp_images/fish-tile.png)

# N+1 Fish, N+2 Fish
## Goal of the Competition
Cod, haddock, flounder - these iconic fish have supported New England’s fishing fleets for generations. When they’re not swimming around the depths of the North Atlantic Ocean, snacking on crabs and lobster, these fish offer a source of sustainable fish and chips. Making these fisheries sustainable means accurately counting all fish caught, including those thrown back at sea because they’re the wrong size or species. Managers require fishermen to monitor that discarded catch and some fishermen recently started carrying video cameras that record fish as they’re returned to the water. But, humans still have to watch hours and hours of video footage to extract the number, size, and species of discarded catch. Can you help automate the video review and make it cheaper and easier to keep track of the fish in (and out) of the sea?

## What's in this Repository
This repository contains code from the winning competitors in the [N+1 Fish, N+2 Fish](https://www.drivendata.org/competitions/48/identify-fish-challenge/) DrivenData challenge.

## Winning Submissions

| Place | Team or User | Public Score | Private Score | Summary of Model |
| --- | --- | --- | --- | --- |
| 1 | [dmytro](https://www.drivendata.org/users/dmytro/) | 0.7661 | 0.7754 | UNET to find the fish ruler, crop frames so ruler is centered, detect fish with SSD network, crop fish and classifiy if one is found. |
| 2 | [ZFTurbo](https://www.drivendata.org/users/ZFTurbo/) | 0.7294 | 0.7365 | UNET to find fish; use bounding boxes to crop image down to fish size; DenseNet_121, ResNet50 and InceptionV3 trained on augmented data, test-time augmentation (multiple predictions per frame), GBM to determine if there is a fish, type of fish, and fish length.  |
| 3 | [Daniel_FG](https://www.drivendata.org/users/Daniel_FG/) | 0.7224 | 0.7316 | Region of interest for fish ruler found with UNET (16 frames at a time), CNNs to detect if there is fish in ROI and best frame to do classification per frame and then Extra Trees model using the frame information, fine tuned VGG models for classification and Extra Trees model on CNN outputs for fish length.  |
| 4 | [harshml](https://www.drivendata.org/users/harshml/) | 0.7036 | 0.7156 | To find sequences of frames which potentially contain a fish, being placed on a ruler, a recurrent network were designed. On such sequences for each frame the 8 class (7 species + background) classifier is applied to the whole frame (global) as well as another 8 class classifier is applied to the fish region only (local). Fish region is obtained as the most confident detection region provided by the Single Shot Multi Box detector (SSD). Sequences with potential fish, given by recurrent network, can be discarded if fused species score from global and local classifiers is too low. The sequences which passed this check treated as sequences with unique fish, thus received the unique fish id. Species are obtained from local classifier scores, and length is calculated from SSD region. |
| Judges Choice | [Kingseso](https://www.drivendata.org/users/Kingseso/) | NA | NA | The judges appreciated the clear outline of the approach, use of the YOLO network, understanding of the problem domain, and compelling hand-drawn process map! |
