
# Import main packages
import numpy as np
import pandas as pd
import math


# Import Misc
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

# Project Library
from src.lib import FCC_lib_data_v1 as ld

# Arguments
file_args = sys.argv[1:]  # -f -i -c
opts, args = getopt.getopt(file_args,"f:",["max_cores=","debug_mode"])

FOLD_ID = 'COMPLETE'  # '' means train each fold, 'ALL' means train fold ALL (whole training set), 'COMPLETE' means folds+ALL
MAX_CORES = 99
FORCE_DEBUG_MODE = False  # Set to True for a light execution of the code, for debugging purpose
for opt, arg in opts:
    if opt == '-f':
        FOLD_ID = arg  
    if opt == '--max_cores':
        MAX_CORES = int(arg)  
    if opt == '--debug_mode':
        FORCE_DEBUG_MODE = True

# Parameters
EXEC_ID = ''
MODEL_ID = 'SKxET_DxA01'  # Model to train
STAGE = 'S3_VSEL2'  # Stage id
SRC_FILE = 'src/{}_models/{}_{}_model.py'.format(STAGE, STAGE, MODEL_ID)
LOG_ID = '{}_{}{}_train'.format(STAGE, MODEL_ID, EXEC_ID)

if __name__ == '__main__':
    
    
    # LOAD SETTINGS
    path_settings_file = "SETTINGS_path.json"
    with open(path_settings_file) as data_file:
        PATH_SETTINGS = json.load(data_file)
        
    exec_settings_file = "SETTINGS_exec.json"
    with open(exec_settings_file) as data_file:
        EXEC_SETTINGS = json.load(data_file)
        
    cache = EXEC_SETTINGS['cache'] == "True"
    do_warnings = EXEC_SETTINGS['warnings'] == "True"
    if not do_warnings:
        warnings.filterwarnings("ignore")
    log = EXEC_SETTINGS['log'] == "True" 
    verbose = EXEC_SETTINGS['verbose'] == "True" 
    debug_mode = EXEC_SETTINGS['debug_mode'] == "True" if FORCE_DEBUG_MODE is False else True
    try:
        max_cores = MAX_CORES
    except:
        try:
            max_cores = int(EXEC_SETTINGS['max_cores'])
        except:
            max_cores = 99
    num_cores = min(multiprocessing.cpu_count(), max_cores)


    # SET LOG
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = ld.Logger('{}{}{}_exec.log'.format(str(PATH_SETTINGS['path_log']), LOG_ID, FOLD_ID), orig_stdout)
    sys.stdout.log = log
    sys.stdout.verbose = True
    sys.stderr = sys.stdout
    
    
    # INITIATE TASK
    task = 'TRAIN'
    print('')
    print('-'*80)
    txt_warning = '          DEBUG MODE ACTIVATED!!!!' if debug_mode else ''
    print('{} MODEL: "{}{}" FOLD:{}{}'.format(task, MODEL_ID, EXEC_ID, FOLD_ID, txt_warning))
    OUTPUT_DIR = str(PATH_SETTINGS['path_outputs_{}'.format(STAGE)])
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print('Output directory: {}'.format(OUTPUT_DIR))    
    print('-'*80)
    sys.stdout.verbose = verbose
    print('Execution started @ {}'.format(time.strftime("%a, %d %b %Y %H:%M:%S +0000. ", time.localtime())))
    start_time = time.time()
                        
    # READ MODEL
    print('-'*40)
    print('READING MODEL DATA: {}'.format(MODEL_ID))
    try:
        Model = imp.load_source('', SRC_FILE).Model()
    except:
        print('... DATA NOT FOUND')
        sys.exit("Error importing model data")
    if hasattr(Model, 'isz2D'):
        print('  ISZ: {}'.format(Model.isz2D))
    if hasattr(Model, 'model_desc'):
        print('  MODEL: {}'.format(Model.model_desc))
    if hasattr(Model, 'model_size'):
        print('    SIZE : {} (channels, isz, classes)'.format(Model.model_size))
    if hasattr(Model, 'model_args'):
        for s in Model.model_args:
            print ('    {} : {}'.format(s, Model.model_args[s]))
            

    # READ & PREPARE DATASET
    print('-'*40)
    print('READING DATASET')
    Data = ld.FishDATA()
    dsetID = ld.read_dsetID()
    imgs_df = dsetID[((dsetID.itype=='train') & (~ dsetID.exclude))]
    imgs_df = imgs_df.reset_index(drop=True)
    imgs_df = imgs_df.assign(idf = imgs_df.index)
    imgs_df.rename(columns={'video_id':'image_id'}, inplace=True)
    if debug_mode:
        print ('  DEBUG MODE ACTIVATED!!!!')
        imgs_df = imgs_df[0:100]

    
    # READ DATASET IN MEMORY
    print('LOADING DATASET INTO RAM')
    def parallel_function(i_row):   
        return Model.read_image(i_row.itype, i_row.image_id,  frame = 'all', read_targets=True) 
    mc_df, mc_reading, mc_verbose, nbc, multipl = imgs_df, True, True, num_cores, 10
    result = []
    if mc_reading:
        itera = range(0,len(mc_df),nbc*multipl)
        if mc_verbose:
            itera = tqdm(itera, total=int(np.ceil(len(mc_df)/float(nbc*multipl))),
                      unit='batch', file=sys.stdout)      
        for i in itera:
            idx_list = range(i, min((i+nbc*multipl), len(mc_df)))
            tmp_result_list = Parallel(n_jobs=nbc)(delayed(parallel_function)(mc_df.iloc[idx] ) for idx in idx_list)  
            result.extend(tmp_result_list)     
    else:
        itera = range(0,len(mc_df))
        if mc_verbose:
            itera =tqdm(itera, total=len(mc_df), unit='img', smoothing=0.35, file=sys.stdout)
        for idx in itera:
            result.append(parallel_function(mc_df.iloc[idx])) 
    train_dset = result  
    train_dset = pd.concat(train_dset, axis=0)
    train_dset = train_dset.reset_index(drop=True)
    

    ##### ITERATE FOLDS #####
    if FOLD_ID == 'COMPLETE':
        folds = np.unique(imgs_df[[Model.fold_column]].values).tolist() + ['ALL',]
    elif FOLD_ID == '':
        folds = np.unique(imgs_df[[Model.fold_column]].values).tolist()
    else:
        folds = [FOLD_ID]
    scores = []
    trained_folds = []
    
    for fold in folds:
        
        # SELECT IMAGES TO TRAIN-VALIDATE
        print('-'*80)
        print('TRAINING FOLD: {}'.format(fold))
        valid_iid = (imgs_df.image_id[imgs_df[Model.fold_column] == fold]).values.tolist()
        train_iid = [s for s in imgs_df.image_id.values.tolist() if s not in valid_iid]
        train_df = train_dset.iloc[[s in train_iid for s in train_dset.image_id], :]
        print('  Training {} samples'.format(len(train_df)))
        valid_df = train_dset.iloc[[s in valid_iid for s in train_dset.image_id], :]
        if fold != 'ALL':
            print('  Validating {} samples'.format(len(valid_df)))
        
        
        x_trn = train_df[[s for s in train_df.columns if s not in ['image_id', 'itype','target']]]\
                .values.astype(np.float32)
        y_trn = train_df['target'].values.astype(np.float32)
        if fold != 'ALL':
            x_val = valid_df[[s for s in valid_df.columns if s not in ['image_id', 'itype','target']]]\
                    .values.astype(np.float32)
            y_val = valid_df['target'].values.astype(np.float32)
    
    
        # Prepare training
        SKmodel = Model.get_SKmodel()
        print('  Training Parameters:')
        for key, value in Model.train_args.iteritems():
            setattr(SKmodel, key, value)
            print('    {} = {}'.format(key, value))
        more_args = {'n_jobs':num_cores, 'verbose': False, 'warm_start': True}
        for key, value in more_args.iteritems():
            setattr(SKmodel, key, value)
            print('    {} = {}'.format(key, value))
        
        n_estimators = Model.n_estimators
        n_estimators_per_round = Model.n_estimators_per_round
        if debug_mode:
            print ('  DEBUG MODE ACTIVATED!!!!')
            n_estimators = 10
            n_estimators_per_round = 10
            
        n_round_max = n_estimators / n_estimators_per_round
        print('  Number of rounds: {}'.format(n_round_max))
        print('  Estimators per round: {}'.format(n_estimators_per_round))
        print('  Total estimators: {}'.format(n_round_max*n_estimators_per_round))
        
        
        # TRAIN
        start_time_L2 = time.time()
        for i_round in range(n_round_max):
            start_time_L3 = time.time()
            SKmodel.n_estimators = n_estimators_per_round *(i_round+1)
            SKmodel.fit(x_trn, y_trn)
            
            if fold == 'ALL':
                trn_pred = SKmodel.predict_proba(x_trn)
                trn_scores = [metric(y_trn, trn_pred) for metric in Model.metrics]
                print ('  Metrics: {}'.format(Model.metrics_desc))
                print ('    Train: {}'.format(['{:.5f}'.format(s) for s in trn_scores]))
                
                trn_score = trn_scores[0]
                time_round = (time.time() - start_time_L3)/1.0
                time_train = (time.time() - start_time_L2)/60.0
                print ('  {}: train {:.5f} - time round/total: {:.0f} s / {:.0f} min'.format(\
                       SKmodel.n_estimators, trn_score, time_round, time_train))
            else:
                trn_pred = SKmodel.predict_proba(x_trn)
                trn_scores = [metric(y_trn, trn_pred) for metric in Model.metrics]
                val_pred = SKmodel.predict_proba(x_val)
                val_scores = [metric(y_val, val_pred) for metric in Model.metrics]
                print ('  Metrics: {}'.format(Model.metrics_desc))
                print ('    Valid: {}'.format(['{:.5f}'.format(s) for s in val_scores]))
                print ('    Train: {}'.format(['{:.5f}'.format(s) for s in trn_scores]))
                
                trn_score = trn_scores[0]
                val_score = val_scores[0]
                time_round = (time.time() - start_time_L3)/1.0
                time_train = (time.time() - start_time_L2)/60.0
                print ('  {}: val {:.5f} - train {:.5f} - time round/total: {:.0f} s / {:.0f} min'.format(\
                       SKmodel.n_estimators, val_score, trn_score, time_round, time_train))
        
        # SAVE MODEL
        filepath = OUTPUT_DIR
        filename = Model.model_filename_format.format(fold_id=fold)
        print('  Saving model: {}{}.pkl.gz'.format(filepath, filename)) 
        start_time_L2 = time.time()
        Model.save_SKmodel(SKmodel, filepath, filename)
        print ('  Model saved in {:.1f} s'.format((time.time() - start_time_L2)/1.0))
        
            
    #END TASK
    print('')
    print('-'*40)
    sys.stdout.vnext = True
    print('Finish EXECUTION @ {}'.format(time.strftime("%a, %d %b %Y %H:%M:%S +0000. ", time.localtime())))
    sys.stdout.vnext = True
    print("Total time: {:.1f} min".format((time.time() - start_time)/60))
    print('-'*80)
    sys.stdout.closelog()
    sys.stdout = orig_stdout
        


