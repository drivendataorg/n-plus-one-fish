import pandas as pd

#read output.csv file
folder = "D:/Users/208018409/Documents/darknet-master/build/darknet/x64/results/"
inputfile1 = "train_output.csv"
df = pd.read_csv(folder + inputfile1)
filename_output = df['filename']

#read training_modified.csv file
folder = "D:/Users/208018409/Documents/drivendata/fish/"
inputfile2 = "training_modified.csv"
labels = pd.read_csv(folder + inputfile2)
result = pd.merge(df, labels, on='filename', how='inner')
result.to_csv(folder + "crop_training.csv", index=False)
print(len(result))


