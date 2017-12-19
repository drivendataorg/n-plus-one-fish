
# For number crunching
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
file_args = sys.argv[1:]  # -f -i
opts, args = getopt.getopt(file_args,"f:i:",["max_cores=","debug_mode"])

FOLD_ID = ''  # '' means train each fold, 'ALL' means train fold ALL (whole training set), 'COMPLETE' means folds+ALL
INIT_EPOCHS = 0
MAX_CORES = 99
FORCE_DEBUG_MODE = False  # Set to True for a light execution of the code, for debugging purpose
for opt, arg in opts:
    if opt == '-f':
        FOLD_ID = arg
    if opt == '-i':
        INIT_EPOCHS = int(arg)   
    if opt == '--max_cores':
        MAX_CORES = int(arg)  
    if opt == '--debug_mode':
        FORCE_DEBUG_MODE = True

# Parameters
EXEC_ID = ''
MODEL_ID = 'NN_BxB01'  # Model to train
STAGE = 'S2_VSE'  # Stage id
SRC_FILE = 'src/{}_models/{}_{}_model.py'.format(STAGE, STAGE, MODEL_ID)
LOG_ID = '{}_{}{}_train'.format(STAGE, MODEL_ID, EXEC_ID)

INIT_EPOCHS = 0  # Start training from this epoch. Use in case of training interruption.
TRAIN_ROUNDS = 20  # Total rounds to train
EPOCHS_PER_ROUND = 1  # Epochs for each training round
TOTAL_EPOCHS = TRAIN_ROUNDS * EPOCHS_PER_ROUND
TRAIN_ALL = False  # Train all epochs or search and save the best ones
MIN_EPOCHS = 10  # Min epochs to train

BATCH0_SIZE = 16  # Batch size for positive samples
BATCH1_SIZE = 112  # Batch size for negative samples
BATCH_SIZE = BATCH1_SIZE + BATCH0_SIZE
SAMPLE_PER_EPOCH_MULTIPLIER = 10  # Number of times the training samples are passed per epoch 
SH_EPOCHS = [   1,     2,      10, ]  # Learning Rate Shedule: Epochs (Always set value for epoch==1)
SH_LR     = [0.01, 0.001,  0.0003, ]  # Learning Rate Shedule: lr value

# BATCH EVALUATION PARAMETERS
HOW_VALIDATE = 'generator' #'in_memory', 'generator', 'batch'

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
    
    
    # READ KERAS NNMODEL & PRINT INFO
    print('-'*40)
    print('KERAS & GPU INFO:')
    import keras
    try:
        print('  Environment: {}'.format(os.environ.get('CONDA_DEFAULT_ENV')))
    except:
        None
    print('  Version: {} {}'.format(keras.__name__, keras.__version__))
    print('  Backend: {}'.format(keras.backend._BACKEND))
    print('  Device: {}'.format(keras.backend.T.config.device))
    print('  Image Format: {}'.format(keras.backend.image_data_format()))
    NNmodel = Model.get_NNmodel()
    if log:
        from keras.utils import plot_model
        plot_model(NNmodel, to_file='{}{}_{}_model.png'.format(str(PATH_SETTINGS['path_log']),STAGE, MODEL_ID), 
             show_shapes = True)
    print('  Input shape: {}'.format(NNmodel.layers[0].input_shape))
    print('  Layers: {}'.format(len(NNmodel.layers)))
    print('  Output shape: {}'.format(NNmodel.layers[-1].output_shape))
    print('  Optimizer: {}'.format(NNmodel.optimizer))
    print('  Loss: {}'.format(NNmodel.loss))
    print('  Metrics: {}'.format(NNmodel.metrics_names))
    print('GPU INFO:')
    print(subprocess.check_output("nvidia-smi", shell=True))


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
        
    
    # READ IMAGES IN MEMORY
    print('LOADING IMAGES INTO RAM')
    def parallel_function(i_row):   
        return Model.read_image(i_row.itype, i_row.image_id,  frame = 'all_labeled', read_labels=True) 
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
    wrap_imgs = result
    
    
    # DATA FEEDER
    class DataFeed(object):
        """ Class to get samples"""
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
                
                if self.y_lst is not None:
                    y_rsl = self.y_lst[index]
            
            if self.y_lst is not None:
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
            if not (MAX_EPOCHS > 0):  # calculated mean has to be > zero
                MAX_EPOCHS = TOTAL_EPOCHS
            print('  Training {} epochs'.format(MAX_EPOCHS))


        # EXTRACT PATCHES
        print('-'*40)
        print('EXTRACTING PATCHES')
        
        print('Extracting train patches')  
        train_videos = [ s for si, s in enumerate(wrap_imgs) if si in train_idx]
        train_videos = [item for sublist in train_videos for item in sublist]
        train1_images = [s[0] for s in train_videos if s[1]!=0]
        train1_targets = [s[1] for s in train_videos if s[1]!=0]
        train0_images = [s[0] for s in train_videos if s[1]==0]
        train0_targets = [s[1] for s in train_videos if s[1]==0]
        
        print('Generating validation patches') 
        valid_videos = [ s for si, s in enumerate(wrap_imgs) if si in valid_idx]
        valid_videos = [item for sublist in valid_videos for item in sublist]
        valid_images = [s[0] for s in valid_videos]
        valid_targets = [s[1] for s in valid_videos]
        
        print('  Patches summary:')
        print('    Train positive patches: {}'.format(len(train1_images)))
        print('    Train negative patches: {}'.format(len(train0_images)))
        if fold != 'ALL':
            print('    Validation patches: {}'.format(len(valid_images)))
        
        
        # GENERATORS
        train1_batch_generator = Model.batch_generator(
                DataFeed(train1_images, train1_targets, Model.data_transforms['train']),
                BATCH1_SIZE, Model.train_gen_params) 
        train0_batch_generator = Model.batch_generator(
                DataFeed(train0_images, train0_targets, Model.data_transforms['train']),
                BATCH0_SIZE, Model.train_gen_params) 
        valid_batch_generator  = Model.batch_generator(
                DataFeed(valid_images, valid_targets, Model.data_transforms['valid']),
                BATCH_SIZE, Model.valid_gen_params) 
        
        def train_batch_generator(train1_batch_generator, train0_batch_generator):
            """ Merge generators to balance positive&negative samples in each epoch """
            while True:
                x1_trn, y1_trn = next(train1_batch_generator)
                x2_trn, y2_trn = next(train0_batch_generator)
                x_trn, y_trn = np.vstack([x1_trn, x2_trn]), np.vstack([y1_trn, y2_trn])
                btch = np.arange(x_trn.shape[0])
                np.random.shuffle(btch)
                x_trn, y_trn = x_trn[btch, ...], y_trn[btch, ...]
                yield x_trn, y_trn
                

        # PREPARE TO TRAIN  
        print('-'*40)
        print('TRAINING PARAMETERS')   
        print('  Trained / Total epochs: {}/{}'.format(INIT_EPOCHS, TOTAL_EPOCHS))
        print('  Rounds / Epochs per round: {}/{}'.format(TRAIN_ROUNDS, EPOCHS_PER_ROUND))
        print('  Batch size: {}'.format(BATCH_SIZE))
        weights_name = '{}{}_{}_{}_weights'.format(OUTPUT_DIR, STAGE, MODEL_ID, fold)
        weights_tmp = '{}{}_{}_{}_weights'.format(str(PATH_SETTINGS['tmp_cache']), STAGE, MODEL_ID, fold)
        lst_weights_file = '{}_lst.hdf5'.format(weights_tmp)
        model_checkpoint = keras.callbacks.ModelCheckpoint(lst_weights_file, monitor='loss', save_best_only=False)   
        steps_per_epoch=np.int(np.ceil(len(train1_images)/float(BATCH1_SIZE)) * SAMPLE_PER_EPOCH_MULTIPLIER)
        validation_steps=np.int(np.ceil(len(valid_images)/float(BATCH_SIZE)) * 1)
        spe = steps_per_epoch if not debug_mode else 2             
        validation_steps=validation_steps if not debug_mode else 2 
        print('  Calculated steps_per_epoch: {}'.format(steps_per_epoch)) 
        print('  Calculated samples_per_epoch: {}'.format(steps_per_epoch*BATCH_SIZE))
        

        # INITIATE MODEL
        print('-'*40)
        print('TRAINING')
        NNmodel = Model.get_NNmodel()
        if cache:
            try:
                # Try to load init weights
                weights_file = '{}_e{}.hdf5'.format(weights_tmp, INIT_EPOCHS)
                NNmodel.load_weights(weights_file)
                print('  Reading pre-trained weights: {}'.format(weights_file))
            except:
                None

        # LEARNING RATE SCHEDULER
        def scheduler(epoch):
            current_epoch = epoch+1
            sh_epochs = SH_EPOCHS
            sh_lr     = SH_LR
            olr = float(NNmodel.optimizer.lr.get_value())
            nlr = sh_lr[0]
            for ep, lr in zip(sh_epochs, sh_lr):
                if current_epoch >= ep:
                    nlr = lr
            if current_epoch == 1:
                print('  Setting learning rate: {:.6f}'.format(nlr)) 
                return nlr
            if (abs(olr-nlr)/olr) > 0.01:
                print('  Changing learning rate: {:.6f} --> {:.6f} at epoch {}'.format(olr, nlr, current_epoch)) 
            return nlr
                
        # Training Time Execution control
        bst_score = 9999
        bst_epoch = None
        bst_eval  = None
        stop_after = 5
        bst_count = 0
        
        
        # Loop traiing epochs
        iround = int(math.floor(INIT_EPOCHS / float(EPOCHS_PER_ROUND)))
        finished_epochs = 0
        start_time_L0 = time.time()
        while iround < TRAIN_ROUNDS and finished_epochs < MAX_EPOCHS:

            round_init_epoch = iround*EPOCHS_PER_ROUND
            round_nb_epochs = round_init_epoch + max(0, min(MAX_EPOCHS - round_init_epoch, EPOCHS_PER_ROUND))
            if finished_epochs == 0:
                print('  Training round: {}'.format(iround+1))
            else:
                print('  Training round: {}, trained {} epochs, lr: {:.6f}'.format(iround+1, 
                      round_init_epoch, float(NNmodel.optimizer.lr.get_value())))
            
            # Fit model
            NNmodel.fit_generator(generator=train_batch_generator(train1_batch_generator, train0_batch_generator), 
                                  epochs=round_nb_epochs, initial_epoch=round_init_epoch,
                                  steps_per_epoch=spe,
                                  callbacks=[model_checkpoint, keras.callbacks.LearningRateScheduler(scheduler)], 
                                  verbose=1, pickle_safe=True, workers=num_cores, max_q_size=100)
                                  
            finished_epochs += round_nb_epochs - round_init_epoch
            iround += 1
            
            weights_file = '{}_e{}.hdf5'.format(weights_tmp, finished_epochs)
            NNmodel.save(weights_file, overwrite=True)  # save whole model, not just weights
            
            # validation
            if fold != 'ALL':
                print("Evaluating...")
                start_time_L1 = time.time()
                if HOW_VALIDATE == 'in_memory':
                    print ("TO IMPLEMENT")
                    
                elif HOW_VALIDATE == 'batch':
                    print ("TO IMPLEMENT")
                
                else:  #generator by default
                    evaluation = NNmodel.evaluate_generator(generator=valid_batch_generator, 
                                       steps=validation_steps, 
                                       max_q_size=10, workers=1, pickle_safe=False)
                    
                print NNmodel.metrics_names
                print evaluation #[s.astype(np.float16) for s in evaluation]validation set in {:.2f} min".format((time.time() - start_time_L1)/60.0))
                print 
                                      
                bst_count += 1
                
                if evaluation[0] < bst_score:
                    bst_count = 0
                    bst_score = evaluation[0] 
                    bst_epoch = finished_epochs
                    bst_eval  = evaluation
                    # save weights
                    if not trAll:
                        weights_file = '{}.hdf5'.format(weights_name)
                        NNmodel.save_weights(weights_file, overwrite=True)
                    
            if bst_count >= stop_after and finished_epochs >= MIN_EPOCHS:
                break  # Stop training when 'stop_after' epochs after best epoch
                
        # Summary
        print("Trained fold {} in {:.2f} min".format(fold, (time.time() - start_time_L0)/60.0))  
        if fold != 'ALL':
            print("Validation scores at epoch: {}".format(bst_epoch if bst_epoch is not None else finished_epochs))
            print NNmodel.metrics_names
            print bst_eval 
        print 
        
        # save weights
        if trAll:
            weights_file = '{}.hdf5'.format(weights_name)
            NNmodel.save_weights(weights_file, overwrite=True)  
        
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