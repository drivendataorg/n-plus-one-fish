import pandas as pd

folder = "D:/Users/208018409/Documents/darknet-master/build/darknet/x64/results/"
inputfile = "comp4_det_test_fish.txt"
outputfile = "train_output.csv"
df = pd.read_csv(folder + inputfile,sep=" ")
df.columns = ['filename','score','x1','y1','x2','y2']
a = df.groupby(['filename'])['score'].transform(max) == df['score']
b = df[a]
result = b.loc[b['score'] > 0.1]
result.to_csv(folder + outputfile, index=False)
print("Done")