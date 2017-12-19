import pandas as pd
import cv2

#read output.csv file for locations to crop
folder = "D:/Users/208018409/Documents/darknet-master/build/darknet/x64/results/"
inputfile = "keepers_17700.csv"
df = pd.read_csv(folder + inputfile)

#extract filenames only, not path
filename = df['filename']
x1 = df['x1']
y1 = df['y1']
x2 = df['x2']
y2 = df['y2']

num_images = len(x1)

#output folder
image_folder = "D:/Users/208018409/Documents/drivendata/fish/test_images_temp/"
output_folder = "D:/Users/208018409/Documents/drivendata/fish/cropped_test_images/"

for i in range (0,num_images):
    img = cv2.imread(image_folder + filename[i] + ".jpg")
    crop_img = img[int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
    cv2.imwrite(output_folder + filename[i] + ".jpg",crop_img)
    print(i)