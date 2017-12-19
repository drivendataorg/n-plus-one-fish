#####Split training dataset into two training and test
import pandas as pd
import numpy as np

folder = "D:\\Users\\208018409\\Documents\\DrivenData\\fish\\"
filename = "fish_dataset2.csv"

df = pd.read_csv(folder + filename)
df['split'] = np.random.randn(df.shape[0],1)

msk = np.random.rand(len(df)) <= 0.9

train = df[msk]
test = df[~msk]

#remove split column
train.drop('split',axis = 1,inplace=True)
test.drop('split',axis = 1,inplace=True)

###Write training and test to files
train.to_csv(folder + "train_labels.csv",sep=',',index=False)
test.to_csv(folder + "test_labels.csv",sep=',',index=False)

print("Done")