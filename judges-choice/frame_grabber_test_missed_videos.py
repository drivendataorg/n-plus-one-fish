import cv2
import os
import glob
import time
import pandas as pd
import numpy as np
import subprocess
from shutil import copyfile
import shutil
import winshell

#Read column from training.csv to know which frames have fish in them
test_video_folder = "D:\\Users\\208018409\\Documents\\DrivenData\\fish\\junk_videos\\"
test_image_folder = "D:\\Users\\208018409\\Documents\\DrivenData\\fish\\junk_images\\"
darknet_folder = "D:/Users/208018409/Documents/darknet-master/build/darknet/x64/data/"
result_folder = "D:/Users/208018409/Documents/darknet-master/build/darknet/x64/results/"
test_images_temp_folder = "D:/Users/208018409/Documents/DrivenData/fish/junk_images_temp/"
fish_folder = "D:/Users/208018409/Documents/DrivenData/fish/junk_images/"
temp_test_file = "temp_test_file.txt"

#Read mp4 files in directory
os.chdir(test_video_folder)
file = glob.glob("*.mp4")

number_of_files = len(file)     #number of files in folder
frame_count = 0     #tracks the total number of frames being reviewed
#loop through files and generate image for each frame
for i in range (0,number_of_files):
    os.chdir(test_video_folder)
    filename = file[i]
    file2 = open(darknet_folder + temp_test_file, "w")
    file3 = open(result_folder + "all_files_junk.csv", "a")
    print('File #: {0} out of {1}'.format(i + 1, number_of_files))
    vidcap = cv2.VideoCapture(file[i])
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        image_filename = test_image_folder + filename[:-4] + "_" + str(count) + ".jpg"
        file2.write(image_filename + "\n")
        cv2.imwrite(image_filename, image)  # save frame as JPEG file
        file3.write(image_filename + "\n")

        # check if all files are valid (not equal to 0k)
        if (os.stat(image_filename).st_size == 0):
            copyfile(test_image_folder + filename[:-4] + "_" + str(count - 1) + ".jpg",image_filename)

        print(count)
        count += 1
    file2.flush()
    file3.flush()
    file2.close
    file3.close


    os.chdir("D:/Users/208018409/Documents/darknet-master/build/darknet/x64/")
    command = "start cmd /c darknet.exe detector valid data/fish_final.data yolo-voc.2.0_fish.cfg backup/yolo-voc_17700.weights"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.communicate()

    #filter comp4_det_test_fish.txt
    if (os.stat(result_folder + 'comp4_det_test_fish.txt').st_size != 0):
        df = pd.read_csv(result_folder + "comp4_det_test_fish.txt",sep=" ")

        df.columns = ['filename', 'score', 'x1', 'y1', 'x2', 'y2']  #give columns names
        a = df.groupby(['filename'])['score'].transform(max) == df['score']
        b = df[a]

        keepers = b.loc[b['score'] > 0.1]

        #append to keepers.csv
        with open(result_folder + 'keepers_17700.csv', 'a') as f:
            if (os.stat(result_folder + 'keepers_junk.csv').st_size == 0):
                keepers.to_csv(f, header=True,index=False)
            else:
                keepers.to_csv(f, header=False, index=False)

        with open(result_folder + 'keepers_temp.csv', 'w') as f:
                keepers.to_csv(f, header=True, index=False)

        keepers_temp = pd.read_csv(result_folder + 'keepers_temp.csv')

        #copy over temp files only
        keeper_filenames = keepers_temp['filename']
        for p in range (0,len(keeper_filenames)):
            temp = keeper_filenames[p]
            copyfile(temp + ".jpg",test_images_temp_folder + temp[len(test_image_folder):] + ".jpg")

    #delete files in test_image folder
    os.chdir(test_image_folder)
    filelist = glob.glob("*.jpg")
    for f in filelist:
        os.remove(f)

    #empty recycling bin
    #time.sleep(10)
    #winshell.recycle_bin().empty(confirm=False, show_progress=True,sound=False)


