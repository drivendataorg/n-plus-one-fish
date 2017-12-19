
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

# Project Library
from src.lib import FCC_lib_data_v1 as ld

# Arguments
file_args = sys.argv[1:]  # -f
opts, args = getopt.getopt(file_args,"f:",["debug_mode"])
FOLD_ID = ''
FORCE_DEBUG_MODE = False  # Set to True for a light execution of the code, for debugging purpose
for opt, arg in opts:
    if opt == '-f':  
        # Argument to select fold
        # ['A', 'B'] set this fold as validation fold.
        # 'ALL' train all dataset (ther is not valivation set)
        # '' for all options ['A', 'B'. 'ALL']
        FOLD_ID = arg
    if opt == '--debug_mode':
        FORCE_DEBUG_MODE = True
        
# Parameters
EXEC_ID = ''
MODEL_ID = 'NN_AxC01'  # Model to train
STAGE = 'S1_ROI'  # Stage id
SRC_FILE = 'src/{}_models/{}_{}_model.py'.format(STAGE, STAGE, MODEL_ID)
LOG_ID = '{}_{}{}_train'.format(STAGE, MODEL_ID, EXEC_ID)

INIT_EPOCHS = 0  # Start training from this epoch. Use in case of training interruption.
STAGE_EPOCHS = (20,0,0)  # Total epochs for each training stage
STEP_SIZE = 1  # Epochs for each training step

BATCH_SIZE = 8  # Number of samples in training mini-batches
SAMPLE_PER_EPOCH_MULTIPLIER = 8  # Number of times the training samples are passed per epoch 

TRAIN_ALL = True  # If False, train all epochs, else train until not improvement
    
    
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
        max_cores = int(EXEC_SETTINGS['max_cores'])
    except:
        max_cores = 99
    num_cores = min(multiprocessing.cpu_count(), max_cores)


    # SET LOG & STDOUT PARAMETERS
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = ld.Logger('{}{}{}_exec.log'.format(str(PATH_SETTINGS['path_log']), LOG_ID, FOLD_ID), orig_stdout)
    sys.stdout.log = log
    sys.stdout.verbose = True
    sys.stderr = sys.stdout

    
    # INITIATE TASK
    print('')
    print('-'*80)
    txt_warning = '          DEBUG MODE ACTIVATED!!!!' if debug_mode else ''
    print('TRAIN MODEL: "{}{}" FOLD:{}{}'.format(MODEL_ID, EXEC_ID, FOLD_ID, txt_warning))
    output_dir = str(PATH_SETTINGS['path_outputs_{}'.format(STAGE)])
    weights_name = '{}{}_{}_{}_weights'.format(output_dir, STAGE, MODEL_ID, FOLD_ID)
    weights_tmp = '{}{}_{}_{}_weights'.format(str(PATH_SETTINGS['tmp_cache']), STAGE, MODEL_ID, FOLD_ID)
    print('Output file (to save weights): {}_*.hdf5'.format(weights_name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
        
    print('  SCALE: {}'.format(Model.scale))
    print('  ISZ: {}'.format(Model.isz2D))
    print('  MODEL: {}'.format(Model.model_desc))
    print('    SIZE : {} (channels, isz, classes)'.format(Model.model_size))
    for s in Model.model_args:
        print ('    {} : {}'.format(s, Model.model_args[s]))
        
        
    # IMPORT KERAS AND PRINT MODEL INFORMATION
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
    
    
    # READ & PREPARE DATA 
    print('-'*40)
    print('READING DATA')
    Data = ld.FishDATA()
    dsetID = ld.read_dsetID()
    
    imgs_df = dsetID[((dsetID.itype=='train') & (~ dsetID.exclude))] 
    imgs_df = imgs_df.reset_index(drop=True)  
    imgs_df = imgs_df.assign(idf = imgs_df.index)
    imgs_df.rename(columns={'video_id':'image_id'}, inplace=True)
    if debug_mode:
        print ('  DEBUG MODE ACTIVATED!!!!')
        imgs_df = imgs_df[0:100]
    all_folds = np.unique(imgs_df[[Model.fold_column]].values).tolist()
    all_folds = all_folds + ['ALL']

    
    # READ IMAGE FUNCTION
    def read_wrap_img(image_id):
        ''' custom function to read image, so we can read from memory or disk'''
        result = Model.read_image('train', image_id, read_mask=True, seed=Model.seed)
        return result
    
    def read_wrap_img_TRAIN(image_id):      
        # Specific parameters (if any) for train samples
        return Model.read_image('train', image_id, read_mask=True)     
    
    # Start loop to train folds
    folds = [FOLD_ID] if FOLD_ID != '' else all_folds
    scores = []
    
    for fold in folds:
        
        print('-'*80)
        print('TRAINING FOLD: {}'.format(fold))
        if fold == 'ALL':
            valid_idx = (imgs_df.index[imgs_df[Model.fold_column] == 'A']).values.tolist()
            train_idx = imgs_df.index.values.tolist()
            trAll = True
            
        else:
            valid_idx = (imgs_df.index[imgs_df[Model.fold_column] == fold]).values.tolist()
            train_idx = [s for s in imgs_df.index.values.tolist() if s not in valid_idx]
            trAll = TRAIN_ALL
            
        train_imgs_df = imgs_df.loc[train_idx] 
        print('  Training {} images'.format(len(train_imgs_df)))
        valid_imgs_df = imgs_df.loc[valid_idx] 
        print('  Validating {} images'.format(len(valid_imgs_df)))


        # patches generation. Useful when training patches instead of whole images
        print('-'*40)
        print('GENERATING PATCHES')
        print('Generating train POSITIVE patches')  
        train1_patches_df = train_imgs_df[['image_id']].copy()
        train1_patches_df =train1_patches_df.assign(x0 = 0)
        train1_patches_df =train1_patches_df.assign(y0 = 0)
        train1_patches_df =train1_patches_df.assign(x1 = Model.isz2D[0])
        train1_patches_df =train1_patches_df.assign(y1 = Model.isz2D[1])
        print('Generating validation patches')  
        valid_patches_df = valid_imgs_df[['image_id']].copy()
        valid_patches_df =valid_patches_df.assign(x0 = 0)
        valid_patches_df =valid_patches_df.assign(y0 = 0)
        valid_patches_df =valid_patches_df.assign(x1 = Model.isz2D[0])
        valid_patches_df =valid_patches_df.assign(y1 = Model.isz2D[1])
        print('  Patches summary:')
        print('    Train POSITIVE patches: {}'.format(len(train1_patches_df)))
        print('    Validation patches: {}'.format(len(valid_patches_df)))
        
        
        # PREPARE TO TRAIN  
        print('-'*40)
        print('TRAINING PARAMETERS')   
        print('  Trained / Total epochs: {}/{}'.format(INIT_EPOCHS, sum(STAGE_EPOCHS)))
        nb_epoch = STEP_SIZE
        print('  Batch size: {}'.format(BATCH_SIZE))
        valid_batch_generator = Model.batch_generator(valid_patches_df, read_wrap_img, Model.patch_generator, 
                                                   BATCH_SIZE, Model.valid_gen_params) 
        weights_name = '{}{}_{}_{}_weights'.format(output_dir, STAGE, MODEL_ID, fold)
        weights_tmp = '{}{}_{}_{}_weights'.format(str(PATH_SETTINGS['tmp_cache']), STAGE, MODEL_ID, fold)
        suff = 'lst' if trAll else 'bst'
        lst_weights_file = '{}_{}.hdf5'.format(weights_tmp, suff)
        model_checkpoint = keras.callbacks.ModelCheckpoint(lst_weights_file, monitor='loss', 
                                                           save_best_only=(not trAll))   


        # STAGE 1
        print('-'*40)
        print('TRAIN STAGE 1')
        NNmodel = Model.get_NNmodel()
        if cache:
            try:
                # Try to load init weights
                weights_file = '{}_e{}.hdf5'.format(weights_tmp, INIT_EPOCHS)
                NNmodel.load_weights(weights_file)
                print('  Reading pre-trained weights: {}'.format(weights_file))
            except:
                None
        if fold == 'ALL':
            try:
                # Calculate epochs as mean fold's best epochs
                round_epochs = int(np.ceil(np.mean([s2 for s1,s2 in scores])))
            except:
                round_epochs = STAGE_EPOCHS[0]
            if not (round_epochs > 0) or TRAIN_ALL:
                round_epochs = STAGE_EPOCHS[0]
            epochs_lst = range(round_epochs)
        else:
            epochs_lst = range(max(0,INIT_EPOCHS), sum(STAGE_EPOCHS[0:1]))
            
        # Set image pathes to train
        patches_df = train1_patches_df
        train_batch_generator = Model.batch_generator(patches_df, read_wrap_img_TRAIN, 
                                                      Model.patch_generator, 
                                                      BATCH_SIZE, Model.train_gen_params) 
        
        steps_per_epoch=np.int(np.ceil(len(patches_df)/float(BATCH_SIZE)) *
                                SAMPLE_PER_EPOCH_MULTIPLIER)
        validation_steps=np.int(np.ceil(len(valid_patches_df)/float(BATCH_SIZE)) * 1)
        spe = steps_per_epoch if not debug_mode else 2                
        validation_steps=validation_steps if not debug_mode else 2 
        print('  Calculated steps_per_epoch: {}'.format(steps_per_epoch)) 
        print('  Calculated samples_per_epoch: {}'.format(steps_per_epoch*BATCH_SIZE))
        
        # Function: Load validation set in memory
        Xval, Yval = None, None
        def load_validation():
            print('-'*40)
            print('LOADING VALIDATION SET IN MEMORY')
            Xval = []
            Yval = []
            valid_batch_generator = Model.batch_generator(valid_patches_df, read_wrap_img, Model.patch_generator, 
                                                       1, Model.valid_gen_params)
            for i in tqdm(range(len(valid_patches_df)), total=len(valid_patches_df), unit='img', file=sys.stdout):
                tmp = next(valid_batch_generator)
                Xval.append(tmp[0])
                Yval.append(tmp[1])
            tmp = np.vstack(Xval)
            Xval = tmp
            del tmp
            tmp = np.vstack(Yval)
            Yval = tmp
            del tmp
            return Xval, Yval
        
        
        # Training Time Execution control
        bst_score = 9999
        bst_epoch = 0
        stop_after = 99
        bst_count = 0
        check2 = False
        check3 = False
        
        for fit_step in epochs_lst:
            total_epochs = fit_step*nb_epoch           
            print('  Training step: {}, trained {} epochs'.format(fit_step+1, total_epochs))
            
            if (fit_step+1) >= 12 and not check2:
                """ Update training parameters """
                args_dict = (Model.model_args).copy()
                from keras.optimizers import Adam
                args_dict['optimizer'] =  Adam(lr=0.0005)

                NNmodel = Model.get_NNmodel(model_args=args_dict)
                NNmodel.load_weights(lst_weights_file)
                
                check2 = True 
                
            # STEP 3
            if (fit_step+1) >= 18 and not check3:
                """ Update training parameters """
                args_dict = (Model.model_args).copy()
                from keras.optimizers import Adam
                args_dict['optimizer'] =  Adam(lr=0.0001)
                NNmodel = Model.get_NNmodel(model_args=args_dict)
                NNmodel.load_weights(lst_weights_file)
                
                check3 = True 
            
            # Train NeuralNet
            NNmodel.fit_generator(generator=train_batch_generator, epochs=nb_epoch, 
                                  steps_per_epoch=spe,
                                  callbacks=[model_checkpoint], verbose=1,
                                  pickle_safe=True, workers=num_cores, max_q_size=100)
            total_epochs += nb_epoch
            bst_count += 1
            
            # Evaluate validation set
            if fold != 'ALL':
                if Xval is None:
                    Xval, Yval = load_validation()
                print
                evaluation = NNmodel.evaluate(Xval, Yval, batch_size=BATCH_SIZE, verbose=1)
                                           
                print NNmodel.metrics_names
                print evaluation 
                print 
                
                if evaluation[0] < bst_score:
                    bst_count = 0
                    bst_score = evaluation[0] 
                    bst_epoch = total_epochs
                    # save weights
                    if not trAll:
                        weights_file = '{}.hdf5'.format(weights_name)
                        NNmodel.save_weights(weights_file, overwrite=True)
            
            # Save weights
            weights_file = '{}_e{}.hdf5'.format(weights_tmp, total_epochs)
            NNmodel.save_weights(weights_file, overwrite=True)
                
            if bst_count == stop_after:
                break  # Stop training when 'stop_after' epochs after best epoch
        
        # save weights
        if trAll:
            weights_file = '{}.hdf5'.format(weights_name)
            NNmodel.save_weights(weights_file, overwrite=True)    
        
        scores.append([bst_score, bst_epoch])
    
    # SAVE SUMMARY
    scores_pd = pd.DataFrame(scores, columns=['score','epochs'])
    scores_pd = scores_pd.assign(fold_id = folds)
    scores_pd.loc[len(scores_pd)] = [np.mean(scores_pd.score.values), np.mean(scores_pd.epochs.values), 'ALL']
    filename = '{}{}_{}_{}_scores.csv.gz'.format(output_dir, STAGE, MODEL_ID, FOLD_ID)
    scores_pd.to_csv(filename, index=False, compression='gzip')
    print scores_pd
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
