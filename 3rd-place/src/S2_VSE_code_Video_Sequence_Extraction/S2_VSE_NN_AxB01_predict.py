
# For number crunching
import numpy as np
import pandas as pd

# Misc
import os
import sys, getopt
import imp
from tqdm import tqdm
import time
from joblib import Parallel, delayed

# Project Library
from src.lib import FCC_lib_data_v1 as ld

# Arguments
file_args = sys.argv[1:]
opts, args = getopt.getopt(file_args,"",["force_all","max_cores=",])

FORCE_ALL = False  # Overwrite files that already exist
THREADS = 6  # Max number of threads
for opt, arg in opts:
    if opt == '--force_all':
        FORCE_ALL = True
    if opt == '--max_cores':
        THREADS = int(arg) 
        
# PARAMETERS
MULTI = 4


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
STAGE, MODEL_ID = 'S2_VSE', 'NN_AxB01'
src_file = 'src/{}_models/{}_{}_model.py'.format(STAGE, STAGE, MODEL_ID)
print('LOADING MODEL: {}'.format(src_file))
Model = imp.load_source('', src_file).Model()


# MAKE PREDICTIONS
print('-'*40)

# Functions
def read_video(item):
    
    itype = item[0]
    video_id = item[1]
    
    imgs, labels, info = Model.read_image(itype, video_id, frame = 'all', split_wrap_imgs = True,
                                         read_labels=False, verbose=False)
    return imgs
    
# Start loop to predict ALL folds
folds = ['A','B','C','ALL']
for fold in folds:
    
    # Filter fold
    f_imgs_df = imgs_df[imgs_df[[Model.fold_column]].values == fold]

    # create groups
    group = []
    groups = []
    for i, i_row in enumerate(f_imgs_df.itertuples()):
        file_to_load = os.path.join(Model.path_predictions, i_row.itype,
                                    '{}_{}_pred.npy.gz'.format(i_row.itype, i_row.video_id))
        if os.path.isfile(file_to_load) and not FORCE_ALL:
            continue
        
        group.append((i_row.itype, i_row.video_id))
        if len(group) == THREADS*MULTI:
            groups.append(group)
            group = []
    if len(group) >0:
        groups.append(group)
    
    # iterate groups
    for group in tqdm(groups):

        imgs_list = Parallel(n_jobs=THREADS)(delayed(read_video)(i) for i in group)  
        
        itype_list = [s[0] for s in group]
        image_id_list = [s[1] for s in group]
        
        Model.get_predictions_BATCH(itype_list, image_id_list, imgs_list, verbose=False)
        
        del(imgs_list, itype_list, image_id_list)
        
