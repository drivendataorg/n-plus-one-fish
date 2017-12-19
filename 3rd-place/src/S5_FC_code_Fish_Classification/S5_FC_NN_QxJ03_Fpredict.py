# Set working directory
import os
directory = "/home/daniel/MEGA/KAGGLE/_33_COMP_Fish_Counting_Competition"
os.chdir(directory) 
 
import os
#os.environ["THEANO_FLAGS"] = "device=cuda1"

# For number crunching
import numpy as np
import pandas as pd

# Misc
import os
import json 
import time
import warnings
import sys, getopt
import imp
from tqdm import tqdm
import subprocess
import multiprocessing
from joblib import Parallel, delayed
import copy
import threading
import queue
import gzip
import shutil

# Project Library
from src.lib import FCC_lib_exec_v1 as exe
from src.lib import FCC_lib_data_v1 as ld
from src.lib import FCC_lib_models_NN_torch_v1 as lm
from src.lib import FCC_lib_train_NN_v1 as lt

# Arguments
file_args = sys.argv[1:]  # -b
opts, args = getopt.getopt(file_args,"b:d:",["force_all","max_cores=","not_use_cuda"])

BATCH_SIZE = None
FORCE_ALL = False
THREADS = 8
DEVICE = '0' ################################# None
USE_CUDA = True
for opt, arg in opts:
    if opt == '-b':
        BATCH_SIZE = int(arg) 
    if opt == '-d':
        DEVICE = arg
    if opt == '--force_all':
        FORCE_ALL = True
    if opt == '--max_cores':
        THREADS = int(arg) 
    if opt == '--not_use_cuda':
        USE_CUDA = False
        
if DEVICE:
    os.environ["CUDA_VISIBLE_DEVICE"] = DEVICE
    import torch
    torch.cuda.set_device(int(DEVICE)) 
# PARAMETERS
MULTI = 1


# READ & PREPARE DATA 
print('-'*40)
print('READING DATA')
Data = ld.FishDATA()
dsetID = ld.read_dsetID()

# MODEL
STAGE, MODEL_ID = 'S5_FC', 'NN_QxJ03'
src_file = 'src/{}_models/{}_{}_model.py'.format(STAGE, STAGE, MODEL_ID)
Model = imp.load_source('', src_file).Model()
Model.get_F_models()
for NNmodel in Model.F_models:
    NNmodel.set_cuda_option(USE_CUDA)

        
# Images
imgs_df = dsetID
fold = 'ALL'

# Filter fold
f_imgs_df = imgs_df[imgs_df[[Model.fold_column]].values == fold]
#f_imgs_df = f_imgs_df[600:]

# iterate groups
for i, i_row in tqdm(enumerate(f_imgs_df.itertuples())):
    _ = Model.get_F_predictions(i_row.itype, i_row.video_id, return_imgs=False, use_cache=True)



