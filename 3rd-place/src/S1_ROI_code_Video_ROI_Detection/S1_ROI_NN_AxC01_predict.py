
# For number crunching
import numpy as np
import pandas as pd

# Misc
import os
import sys
import imp
import time
from tqdm import tqdm

# Project Library
from src.lib import FCC_lib_data_v1 as ld

# READ & PREPARE DATA 
print('-'*40)
print('READING DATA')
Data = ld.FishDATA()
dsetID = ld.read_dsetID()

imgs_df = dsetID[((~ dsetID.exclude))]  
imgs_df = imgs_df.reset_index(drop=True)  
imgs_df = imgs_df.assign(idf = imgs_df.index)

start_time = time.time()

# MODEL
print('-'*40)
S1_STAGE, S1_MODEL_ID = 'S1_ROI', 'NN_AxC01'
S1_src_file = 'src/{}_models/{}_{}_model.py'.format(S1_STAGE, S1_STAGE, S1_MODEL_ID)
print('LOADING MODEL: {}'.format(S1_src_file))
S1_Model = imp.load_source('', S1_src_file).Model()


# MAKE PREDICTIONS
print('-'*40)
# Start loop to train ALL folds
df = []
folds = ['A','B','ALL',]
for fold in folds:
    valid_df = imgs_df[imgs_df.F2 == fold]
    
    print('PREDICTING FOLD: {}'.format(fold))
    
    itera = tqdm(enumerate(valid_df.itertuples()), total=len(valid_df), unit='img', file=sys.stdout)
    for i, row in itera:
        itype = row.itype
        video_id = row.video_id
        
        # Get predictions
        labels, pred, img, msk, vidD = S1_Model.get_labels(itype, video_id, use_cache=True, verbose=False)
        df.append(labels)


# SAVE TABLE
print('-'*40)
print('SAVING TABLE')
df = pd.concat(df, axis=0)
filename = os.path.join(S1_Model.path_predictions, '{}_{}_pred.csv.gz'.format(S1_STAGE, S1_MODEL_ID))
df.to_csv(filename, index=False, compression='gzip')

print('FINISHED')
print("Total time: {:.1f} min".format((time.time() - start_time)/60))
print('-'*40)