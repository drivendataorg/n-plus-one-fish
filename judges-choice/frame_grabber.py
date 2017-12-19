import cv2
import os
import glob
import pandas as pd
import numpy as np

#Read column from training.csv to know which frames have fish in them
fish_folder = "D:\\Users\\208018409\\Documents\\DrivenData\\fish\\"
training_annotations_file = "training.csv"
data = pd.read_csv(fish_folder + training_annotations_file)
data = data[np.isfinite(data['length'])]
frames = data['frame'].values
video_filenames = data['video_id'].values
frame_array_len = len(frames)

training_video_folder = "D:\\Users\\208018409\\Documents\\DrivenData\\fish\\training_videos\\"
training_image_folder =  "D:\\Users\\208018409\\Documents\\DrivenData\\fish\\training_images\\"

#Read mp4 files in directory
os.chdir(training_video_folder)
file = glob.glob("*.mp4")

number_of_files = len(file)     #number of files in folder

frame_count = 0     #tracks the total number of frames being reviewed
#loop through files and generate image for each frame
for i in range (0,number_of_files):
    keeper_frames = []
    filename = file[i]
    #grab frames from cvs file that match up with video_id
    for n in range (0,frame_array_len):
        if (video_filenames[n] == filename[:-4]):
            keeper_frames.append(frames[n])


    print('File #: {0} out of {1}'.format(i + 1, number_of_files))
    vidcap = cv2.VideoCapture(file[i])
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if any(x == count for x in keeper_frames):
            image_filename = training_image_folder + filename[:-4] + "_" + str(count) + ".jpg"
            cv2.imwrite(image_filename, image)  # save frame as JPEG file
            #print(count)
        print(count)
        count += 1
