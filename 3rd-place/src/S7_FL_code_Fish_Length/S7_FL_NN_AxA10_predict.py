
# For number crunching
import numpy as np
import pandas as pd

# Misc
import sys, getopt
import imp
from tqdm import tqdm
import time

# Project Library
from src.lib import FCC_lib_data_v1 as ld

# Arguments
file_args = sys.argv[1:]  #
opts, args = getopt.getopt(file_args,"",["force_all",])

FORCE_ALL = False  # Overwrite files that already exist
for opt, arg in opts:
    if opt == '--force_all':
        FORCE_ALL = True


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
STAGE, MODEL_ID = 'S7_FL', 'NN_AxA10'
src_file = 'src/{}_models/{}_{}_model.py'.format(STAGE, STAGE, MODEL_ID)
print('LOADING MODEL: {}'.format(src_file))
Model = imp.load_source('', src_file).Model()
NNmodel = Model.get_NNmodel()
   
     
# MAKE PREDICTIONS
print('-'*40)

# Start loop to predict ALL folds
folds = ['A','B','C','ALL']
folds = ['ALL',]
for fold in folds:

    # Filter fold
    f_imgs_df = imgs_df[imgs_df[[Model.fold_column]].values == fold]

    itera = tqdm(enumerate(f_imgs_df.itertuples()), total=len(f_imgs_df), unit='img', file=sys.stdout)
    for i, row in itera:
        itype = row.itype
        video_id = row.video_id
        
        # Get predictions
        _ = Model.get_predictions(itype, video_id, return_imgs=False, use_cache=True, 
                                  force_save=FORCE_ALL, verbose=False)

        
