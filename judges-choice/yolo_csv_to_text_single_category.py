import cv2
import os
import glob
import pandas as pd
import numpy as np

#Read column from training.csv to know which frames have fish in them
fish_folder = "D:\\Users\\208018409\\Documents\\DrivenData\\fish\\"
text_folder = "D:\\Users\\208018409\\Documents\\DrivenData\\fish\\training_labels\\"
training_annotations_file = "train_labels.csv"
data = pd.read_csv(fish_folder + training_annotations_file)

filename = data['filename'].values
xmax = data['xmax'].values
ymax = data['ymax'].values
xmin = data['xmin'].values
ymin = data['ymin'].values
width = data['width'].values
height = data['height'].values
Class = data['class'].values
number_of_items = len(filename)

count = 0
for i in range (0,number_of_items):
    #create text file
    temp = filename[i]
    text_file_name = temp[:-3] + "txt"
    file = open (text_folder + text_file_name,"w")

    x = ((xmax[i]+xmin[i])/2)/width[i]
    y = ((ymax[i]+ymin[i])/2)/height[i]
    w = abs(xmax[i] - xmin[i])/width[i]
    h = abs(ymax[i] - ymin[i])/height[i]

    file.write("0" + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
    file.close
    count += 1
    print(count)

