#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Set working directory
import os
directory = "/home/daniel/MEGA/KAGGLE/_33_COMP_Fish_Counting_Competition"
os.chdir(directory) 


import os
#os.environ["THEANO_FLAGS"] = "device=cuda1"


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
from src.lib import FCC_lib_exec_v1 as exe
from src.lib import FCC_lib_data_v1 as ld
from src.lib import FCC_lib_models_NN_torch_v1 as lm
from src.lib import FCC_lib_train_NN_v1 as lt

# Arguments
file_args = sys.argv[1:]  # -f -i -c
opts, args = getopt.getopt(file_args,"f:i:d:",["max_cores=","not_use_cuda"])

FOLD_ID = 'COMPLETE'  # '' means train each fold, 'ALL' means train fold ALL (whole training set), 'COMPLETE' means folds+ALL
INIT_EPOCHS = 0
MAX_CORES = 8  ################################# 99
DEVICE = '1' ################################# None
USE_CUDA = True
for opt, arg in opts:
    if opt == '-f':
        FOLD_ID = arg
    if opt == '-d':
        DEVICE = arg
    if opt == '-i':
        INIT_EPOCHS = int(arg)   
    if opt == '--max_cores':
        MAX_CORES = int(arg)  
    if opt == '--not_use_cuda':
        USE_CUDA = False

if DEVICE:
    os.environ["CUDA_VISIBLE_DEVICE"] = DEVICE
    import torch
    torch.cuda.set_device(int(DEVICE))
    
# Parameters
EXEC_ID = ''
MODEL_ID = 'NN_QxG18'
STAGE = 'S5_FC'
SRC_FILE = 'src/{}_models/{}_{}_model.py'.format(STAGE, STAGE, MODEL_ID)
LOG_ID = '{}_{}{}_train'.format(STAGE, MODEL_ID, EXEC_ID)

INIT_EPOCHS = 0
TRAIN_ROUNDS = 5 #20
EPOCHS_PER_ROUND = 1
TOTAL_EPOCHS = TRAIN_ROUNDS * EPOCHS_PER_ROUND
TRAIN_ALL = False  # Train all epochs or search and save the best ones
MIN_EPOCHS = 1

BATCH_SIZE = 8
SAMPLE_PER_EPOCH_MULTIPLIER = 1 #10
SH_EPOCHS = [    1,     3,      5, ]  # Always set value for epoch==1
SH_LR     = [0.001, 0.0003,  0.0001, ]

# BATCH EVALUATION PARAMETERS
HOW_VALIDATE = 'generator' #'in_memory', 'generator', 'batch'
EVAL_BATCH_SIZE = 4
EVAL_THREADS = 7
EVAL_MULTI = 4

FORCE_DEBUG_MODE = False

if __name__ == '__main__':
    
    ##### STARTING TRAINING #####
    
    # LOAD SETTINGS
    cache, log, verbose, debug_mode, num_cores = eval(exe.load_settings_rA1.func_code)

    # SET LOG
    orig_stdout, orig_stderr, sys.stdout, sys.stderr = eval(exe.set_log_rA1.func_code)
    
    # INITIATE TASK
    task = 'TRAIN'
    start_time = eval(exe.initiate_task_rA1.func_code) 
    
    # READ MODEL
    Model = eval(exe.read_model_rA1.func_code) 
    
    # READ PYTORCH NNMODEL & PRINT INFO
    import torch
    eval(exe.read_pytorch_NNmodel_rA1.func_code) 
    
    
    ##### COMMON SETTINGS #####

    # READ & PREPARE DATASET
    print('-'*40)
    print('READING DATASET')
    Data = ld.FishDATA()
    dsetID = ld.read_dsetID()
    imgs_df = dsetID[((dsetID.itype=='train') & (~ dsetID.exclude))]
    #imgs_df = pd.merge(imgs_df, Data.annotations, how='inner', on='video_id')
    imgs_df = imgs_df.reset_index(drop=True)  # Important to keep consistancy between list index
    imgs_df = imgs_df.assign(idf = imgs_df.index)
    imgs_df.rename(columns={'video_id':'image_id'}, inplace=True)
    if debug_mode:
        print ('  DEBUG MODE ACTIVATED!!!!')
        imgs_df = imgs_df[0:100]
        
    # READ IMAGES IN MEMORY
    print('LOADING IMAGES INTO RAM')
    def parallel_function(i_row):   
        return Model.read_image(i_row.itype, i_row.image_id,  frame = 'all_train', read_labels=True) 
    mc_df, mc_reading, mc_verbose, nbc, multipl = imgs_df, True, True, num_cores, 10
    wrap_imgs = eval(exe.mc_row_iterator_rA1.func_code)   
    

    class DataFeed(object):
    
        def __init__(self, x_lst, y_lst=None, transform=None):
            
            self.x_lst = x_lst
            self.y_lst = y_lst
            self.length = len(x_lst)
            self.transform = transform
            
        def __getitem__(self, index):

            if hasattr(index, '__iter__'):
                x_rsl = [self.x_lst[s] for s in index]
                y_rsl = [self.y_lst[s] for s in index]
                
                if self.transform is not None:
                    x_rsl = [self.transform(x) for x in x_rsl]
                x_rsl = np.array(x_rsl)
                
                if self.y_lst is not None:
                    y_rsl = np.vstack(y_rsl)
                    
            else:
                x_rsl = self.x_lst[index]
                if self.transform is not None:
                    x_rsl = self.transform(x_rsl)
                
                if self.y_lst is not None:
                    y_rsl = self.y_lst[index]
            
            x_rsl = torch.from_numpy(x_rsl.astype(np.float32))
            if self.y_lst is not None:
                y_rsl = torch.from_numpy(y_rsl.astype(np.float32))
                return x_rsl, y_rsl
            return x_rsl
        
        def __len__(self):
            return self.length


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
        valid_idx = (imgs_df.index[imgs_df[Model.fold_column] == fold]).values.tolist()
        train_idx = [s for s in imgs_df.index.values.tolist() if s not in valid_idx]
        train_imgs_df = imgs_df.loc[train_idx] 
        print('  Training {} images'.format(len(train_imgs_df)))
        valid_imgs_df = imgs_df.loc[valid_idx] 
        if fold != 'ALL':
            trAll = TRAIN_ALL
            MAX_EPOCHS = TOTAL_EPOCHS
            print('  Validating {} images'.format(len(valid_imgs_df)))
        else:
            trAll = True
            try:  # Get mean from previously calculated folds
                MAX_EPOCHS = int(np.ceil(np.mean([s2 for s1,s2 in scores])))
            except:
                try:  # Read previously calculated folds
                    filename = '{}{}_{}_{}_scores.csv.gz'.format(OUTPUT_DIR, STAGE, MODEL_ID, FOLD_ID)
                    scores_pd = pd.read_csv(filename)
                    MAX_EPOCHS = int(np.ceil(scores_pd.epochs[scores_pd.fold_id == 'ALL'].values[0]))
                except:  # Train all the epochs
                    MAX_EPOCHS = TOTAL_EPOCHS
            if not (MAX_EPOCHS > 0):  # to be sure the calcualted mean is > zero
                MAX_EPOCHS = TOTAL_EPOCHS
            print('  Training {} epochs'.format(MAX_EPOCHS))


        # EXTRACT PATCHES
        print('-'*40)
        print('EXTRACTING PATCHES')
        
        print('Extracting train patches')  
        train_videos = [ s for si, s in enumerate(wrap_imgs) if si in train_idx]
        train_videos = [item for sublist in train_videos for item in sublist]
        train_images = [s[0] for s in train_videos]
        train_targets = [s[1] for s in train_videos]
        
        print('Generating validation patches') 
        valid_videos = [ s for si, s in enumerate(wrap_imgs) if si in valid_idx]
        valid_videos = [item for sublist in valid_videos for item in sublist]
        valid_images = [s[0] for s in valid_videos]
        valid_targets = [s[1] for s in valid_videos]
        
        print('  Patches summary:')
        print('    Train patches: {}'.format(len(train_images)))
        if fold != 'ALL':
            print('    Validation patches: {}'.format(len(valid_images)))
        
        
        # GENERATORS
        from torch.utils.data import DataLoader
        use_cuda = torch.cuda.is_available()
        
        train_batch_loader = DataLoader(DataFeed(train_images, train_targets, Model.data_transforms['train']),
                                           BATCH_SIZE, shuffle = True, pin_memory=use_cuda, num_workers = num_cores)
        valid_batch_loader = DataLoader(DataFeed(valid_images, valid_targets, Model.data_transforms['valid']),
                                           BATCH_SIZE, shuffle = False, pin_memory=use_cuda, num_workers = num_cores)
        
        
        # PREPARE TO TRAIN  
        print('-'*40)
        print('TRAINING PARAMETERS')   
        print('  Trained / Total epochs: {}/{}'.format(INIT_EPOCHS, TOTAL_EPOCHS))
        print('  Rounds / Epochs per round: {}/{}'.format(TRAIN_ROUNDS, EPOCHS_PER_ROUND))
        print('  Batch size: {}'.format(BATCH_SIZE))

        model_name = '{}{}_{}_{}_model'.format(OUTPUT_DIR, STAGE, MODEL_ID, fold)
        model_tmp = '{}{}_{}_{}_model'.format(str(PATH_SETTINGS['tmp_cache']), STAGE, MODEL_ID, fold)
        lst_model_file = '{}_lst.torch'.format(model_tmp)
        #model_checkpoint = keras.callbacks.ModelCheckpoint(lst_weights_file, monitor='loss', save_best_only=False) 

        steps_per_epoch=np.int(np.ceil(len(train_images)/float(BATCH_SIZE)) * SAMPLE_PER_EPOCH_MULTIPLIER)
        validation_steps=np.int(np.ceil(len(valid_images)/float(BATCH_SIZE)) * 1)
        spe = steps_per_epoch if not debug_mode else 2             
        validation_steps=validation_steps if not debug_mode else 2 
        print('  Calculated steps_per_epoch: {}'.format(steps_per_epoch)) 
        print('  Calculated samples_per_epoch: {}'.format(steps_per_epoch*BATCH_SIZE))
        
        
        # INITIATE MODEL
        print('-'*40)
        print('TRAINING')
        NNmodel = Model.get_NNmodel()
        NNmodel.set_cuda_option(USE_CUDA)
        if cache:
            try:
                model_file = '{}_e{}.torch'.format(model_tmp, INIT_EPOCHS)
                NNmodel.load_model(model_file)
                print('  Reading pre-trained model: {}'.format(model_file))
            except:
                None
        
        # scheduler
        def exp_lr_scheduler_v1(optimizer, epoch, decay=0.1, lr_decay_epoch=10):
            """Decay learning rate by a factor of decay every lr_decay_epoch epochs."""
        
            if epoch > 0 and epoch % lr_decay_epoch == 0:
                print('LR is set to LR*{}'.format(decay))
                print('')
        
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * decay
            
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param_group['lr'] > 0:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                       
            return optimizer
        
        lr_schedulers = [lambda x,y:exp_lr_scheduler_v1(x,y, decay=0.3162277, lr_decay_epoch=5),]
        
        
        # Execution control
        bst_score = 9999
        bst_epoch = None
        bst_eval  = None
        stop_after = 5
        bst_count = 0
        
        
        # Loop training epochs
        iround = int(math.floor(INIT_EPOCHS / float(EPOCHS_PER_ROUND)))
        finished_epochs = 0
        start_time_L0 = time.time()
        while iround < TRAIN_ROUNDS and finished_epochs < MAX_EPOCHS:

            round_init_epoch = iround*EPOCHS_PER_ROUND
            round_nb_epochs = round_init_epoch + max(0, min(MAX_EPOCHS - round_init_epoch, EPOCHS_PER_ROUND))

            if finished_epochs == 0:
                print('  Training round: {}'.format(iround+1))
            else:
                #lrs = np.unique([param_group['lr'] for param_group in s.param_groups \
                #                 for s in NNmodel.optimizers]).tolist()
                lrs = np.unique([param_group['lr'] for s in NNmodel.optimizers \
                                for param_group in s.param_groups]).tolist()
                print('  Training round: {}, trained {} epochs, lr: {}'.format(iround+1, round_init_epoch, lrs))

            # Fit model
            #NNmodel.load_model(model_file) 
            NNmodel.train_loader(train_batch_loader, round_nb_epochs, round_init_epoch, spe, lr_schedulers)
            finished_epochs += round_nb_epochs - round_init_epoch
            iround += 1
            
            # Save model
            model_file = '{}_e{}.torch'.format(model_tmp, finished_epochs)
            NNmodel.save_model(model_file)

            # validation
            if fold != 'ALL':
                print("Evaluating...")
                start_time_L1 = time.time()
                if HOW_VALIDATE == 'in_memory':
                    print ("TO IMPLEMENT")
                    
                elif HOW_VALIDATE == 'batch':
                    print ("TO IMPLEMENT")
                
                else:  #generator by default
                    evaluation = NNmodel.evaluate_loader(valid_batch_loader)
                    
                print NNmodel.metrics_names
                print evaluation #[s.astype(np.float16) for s in evaluation]
                print("Evaluated validation set in {:.2f} min".format((time.time() - start_time_L1)/60.0))
                print 
                                      
                bst_count += 1
                
                if evaluation[0] < bst_score:
                    bst_count = 0
                    bst_score = evaluation[0] 
                    bst_epoch = finished_epochs
                    bst_eval  = evaluation
                    # save weights
                    if not trAll:
                        model_file = '{}.torch'.format(model_name)
                        NNmodel.save_model(model_file)  
                    
            if bst_count >= stop_after and finished_epochs >= MIN_EPOCHS:
                break
            
        # Summary
        print("Trained fold {} in {:.2f} min".format(fold, (time.time() - start_time_L0)/60.0))  
        if fold != 'ALL':
            print("Validation scores at epoch: {}".format(bst_epoch if bst_epoch is not None else finished_epochs))
            print NNmodel.metrics_names
            print bst_eval 
        print 
        
        # save weights
        if trAll:
            model_file = '{}.torch'.format(model_name)
            NNmodel.save_model(model_file)  
        
        if fold != 'ALL':
            scores.append([bst_score, bst_epoch])
            trained_folds.append(fold)
    
    # SAVE SUMMARY
    if FOLD_ID == '' or FOLD_ID == 'COMPLETE':
        scores_pd = pd.DataFrame(scores, columns=['score','epochs'])
        scores_pd = scores_pd.assign(fold_id = trained_folds)
        scores_pd.loc[len(scores_pd)] = [np.mean(scores_pd.score.values), np.mean(scores_pd.epochs.values), 'ALL']
        filename = '{}{}_{}_{}_scores.csv.gz'.format(OUTPUT_DIR, STAGE, MODEL_ID, FOLD_ID)
        scores_pd.to_csv(filename, index=False, compression='gzip')
    
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