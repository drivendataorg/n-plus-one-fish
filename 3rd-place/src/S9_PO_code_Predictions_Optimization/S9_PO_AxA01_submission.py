
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
file_args = sys.argv[1:]  #
opts, args = getopt.getopt(file_args,"",["max_cores=",])

FORCE_DEBUG_MODE = False
FOLD_ID = 'TEST'
MAX_CORES = 16
for opt, arg in opts:
    if opt == '--max_cores':
        MAX_CORES = int(arg)  

# Parameters
EXEC_ID = ''
MODEL_ID = 'AxA01'
STAGE = 'S9_PO'
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
    task = 'GEENRATE SUBMISSION'
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
    print('READING TEST DATASET')
    Data = ld.FishDATA()
    dsetID = ld.read_dsetID()
    test_df = dsetID[((dsetID.itype=='test'))]
    test_df = test_df.reset_index(drop=True)
    test_df.rename(columns={'video_id':'image_id'}, inplace=True)
    if debug_mode:
        print ('  DEBUG MODE ACTIVATED!!!!')
        test_df = test_df[0:100]
        

    # READ DATASET IN MEMORY
    print('LOADING DATASET INTO RAM')
    def parallel_function(i_row):   
        return Model.read_video(i_row.itype, i_row.image_id,  read_targets=True) 
    mc_df, mc_reading, mc_verbose, nbc, multipl = test_df, True, True, num_cores, 10
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
    test_dset = result  
    test_dset = pd.concat(test_dset, axis=0)
    test_dset = test_dset.reset_index(drop=True)

    ##### PARAMETERS #####
    # Sequence extractor
    THR_SEQUENCE = 6.5  # [0,10] define the end of a sequence involving the same fish (aprox 4 frames after best frame)
    # Length prediction
    THR_LENGTH_FRAMES = 80  # [0,100] percentile for select best frames to classify fishes.
    # Species prediction
    THR_SPECIES_FRAMES = 95  # [0,100] percentile for select best frames to classify fishes.
 
    # FUNCTIONS
    def model_function(t_dset, video_id):
        # video_id = '3CQpNbbfjyfu6Imt' 
        
        # Filter dataset for video_id
        mini_df = t_dset[t_dset.video_id==video_id]
        mini_df = mini_df.set_index('frame', drop=False)
        mini_df = mini_df.sort_index(axis=0, ascending=True)
        
        # Set 'best frame probability'
        prob_bst_frm = mini_df.Dx_Pred.values
        
        ## END OF SEQUENCE DETECTION
        
        # End of sequence occurs when R10_Pred >= THR_SEQUENCE
        xc = np.clip(mini_df.R10_Pred.values, 0, THR_SEQUENCE)
        mini_df = mini_df.assign(f01 = xc)
        
        # Initiate sequence flags
        frame = []
        fish_number = []
        fr_ini = [0,]  # List of frames numbers starting a sequence
        fr_end = []  # List of frames numbers finishing a sequence
        fr_fn = [1]  # List of fish number
        fr_max = np.max(mini_df.frame.values)  # max frame in video
        ctr=0  # Control flag
        
        # Run frames over time to find sequences
        for i_row in mini_df.itertuples():
            
            # decision
            if i_row.f01 == THR_SEQUENCE:   # Detected end of sequence
                ctr = 1
                frame.append(i_row.frame)
                fish_number.append(fr_fn[-1])
                
            elif ctr == 1 and i_row.frame <= (fr_max-6):  # Start new fish number
                # Set sequence
                fr_end.append(i_row.frame)
                frame.append(i_row.frame)
                fish_number.append(fr_fn[-1])
                
                fr_ini.append(i_row.frame+1)
                fr_fn.append(fr_fn[-1]+1)
                ctr = 0
            else:
                ctr = 0
                frame.append(i_row.frame)
                fish_number.append(fr_fn[-1])
            
        # result
        rst_df = pd.DataFrame({'video_id':video_id, 
                               'frame': frame,
                               'PRED_fish_number': fish_number,
                               'PRED_prob_bst_frame': prob_bst_frm,
                               })   
        
        # Best frames for LENGTH predictions        
        rst_group = rst_df.groupby('PRED_fish_number')
        bst_frm_thr = rst_group['PRED_prob_bst_frame'].agg([lambda x: np.nanpercentile(x,THR_LENGTH_FRAMES)])
        bst_frm_thr.columns = ['PRED_bst_frm_thr_length']
        bst_frm_thr = bst_frm_thr.assign(PRED_fish_number=bst_frm_thr.index)
        rst_df = pd.merge(rst_df, bst_frm_thr, how='left', on='PRED_fish_number')
        rst_df = rst_df.assign(PRED_bst_frame_length = rst_df.PRED_prob_bst_frame >= rst_df.PRED_bst_frm_thr_length)
        rst_df = rst_df.assign(frame_merge_length = np.nan)
        rst_df.loc[rst_df.PRED_bst_frame_length.values, 'frame_merge_length'] \
                   = rst_df.frame[rst_df.PRED_bst_frame_length.values].values
        
        # Best frames for SPECIES predictions
        rst_group = rst_df.groupby('PRED_fish_number')
        bst_frm_thr = rst_group['PRED_prob_bst_frame'].agg([lambda x: np.nanpercentile(x,THR_SPECIES_FRAMES)])
        bst_frm_thr.columns = ['PRED_bst_frm_thr_species']
        bst_frm_thr = bst_frm_thr.assign(PRED_fish_number=bst_frm_thr.index)
        rst_df = pd.merge(rst_df, bst_frm_thr, how='left', on='PRED_fish_number')
        rst_df = rst_df.assign(PRED_bst_frame_species = rst_df.PRED_prob_bst_frame >= rst_df.PRED_bst_frm_thr_species)
        rst_df = rst_df.assign(frame_merge_species = np.nan)
        rst_df.loc[rst_df.PRED_bst_frame_species.values, 'frame_merge_species'] \
                   = rst_df.frame[rst_df.PRED_bst_frame_species.values].values
        
        # fill NA with nearest values
        rr = rst_df.frame_merge_length.astype(np.float64).interpolate(method='nearest')
        rr = rr.ffill()
        rr = rr.bfill()
        rr = rr.astype(int)
        rst_df = rst_df.assign(frame_merge_length = rr)
        
        rr = rst_df.frame_merge_species.astype(np.float64).interpolate(method='nearest')
        rr = rr.ffill()
        rr = rr.bfill()
        rr = rr.astype(int)
        rst_df = rst_df.assign(frame_merge_species = rr)
        
        return rst_df
    
    def prepare_test(rst_df, test_dset, video_id):

        mini_rst_df = rst_df[rst_df.video_id==video_id]
        mini_rst_df = mini_rst_df[['video_id','frame','frame_merge_length','frame_merge_species',
                                   'PRED_fish_number','PRED_bst_frame_length','PRED_bst_frame_species']]
        
        # LENGTH
        mini_test_df = test_dset[test_dset.video_id==video_id]
        mini_test_df = mini_test_df[['frame','PRED_length',]]
        mini_test_df = mini_test_df[[s in mini_rst_df.frame.values[mini_rst_df.PRED_bst_frame_length]\
                                      for s in mini_test_df.frame.values]]
        mini_test_df.rename(columns={'frame': 'frame_merge_length'}, inplace=True)
        mini_rst_df = pd.merge(mini_rst_df, mini_test_df, how='left', on='frame_merge_length')
        
        # SPECIES
        mini_test_df = test_dset[test_dset.video_id==video_id]
        mini_test_df = mini_test_df[['frame',]+['PRED_'+s for s in Data.clss_names]]
        mini_test_df = mini_test_df[[s in mini_rst_df.frame.values[mini_rst_df.PRED_bst_frame_species]\
                                      for s in mini_test_df.frame.values]]
        mini_test_df.rename(columns={'frame': 'frame_merge_species'}, inplace=True)
        mini_rst_df = pd.merge(mini_rst_df, mini_test_df, how='left', on='frame_merge_species')
        
        mini_test_df = mini_rst_df
        
        return mini_test_df
    
    
    # MAKE PREDICTIONS
    print('MAKE PREDICTIONS')
    def parallel_function(i_row): 
        rst_df = model_function(test_dset, i_row.image_id)
        rst_test_df = prepare_test(rst_df, test_dset, i_row.image_id)
        
        nPRED_COLUMNS = ['frame',
           'video_id',
           'PRED_fish_number',
           'PRED_length',
           'PRED_species_fourspot',
           'PRED_species_grey sole',
           'PRED_species_other',
           'PRED_species_plaice',
           'PRED_species_summer',
           'PRED_species_windowpane',
           'PRED_species_winter']
        COLUMNS = ['frame',
           'video_id',
           'fish_number',
           'length',
           'species_fourspot',
           'species_grey sole',
           'species_other',
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']
        rst_test_df = rst_test_df[nPRED_COLUMNS]
        rst_test_df.columns = COLUMNS
        return rst_test_df
    mc_df, mc_reading, mc_verbose, nbc, multipl = test_df, True, True, num_cores, 10
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
    test_pred = result  
    test_pred = pd.concat(test_pred, axis=0)
    test_pred = test_pred.reset_index(drop=True)
    
    
    # generate submission
    submission = pd.read_csv(Data.paths['sample'])
    subm = submission[['row_id','frame','video_id']]
    subm = pd.merge(subm, test_pred, how='left', on=['frame','video_id'])
    subm[['length',
           'species_fourspot',
           'species_grey sole',
           'species_other',
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']] = subm[['length',
           'species_fourspot',
           'species_grey sole',
           'species_other',
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']].astype(np.float64)
    subm = subm.bfill()
    
    print("Generating submission file...")
    file_to_save = os.path.join(str(PATH_SETTINGS['path_submissions']), 'submission_{}_{}.csv'.format(STAGE, MODEL_ID))
    subm.to_csv(file_to_save, index=False)
    

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
