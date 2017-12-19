
# Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import time
import gzip
import shutil
import os
import math
import copy
import imp

# Project Library
from src.lib import FCC_lib_data_v1 as ld

MODEL_ID = "SKxET_AxA02"
STAGE    = "S8_FLL2"

class Model(object):
    
    def __init__(self):
        
        self.reset_variables()
        self.reset_parameters_PREV_MODEL()
        self.reset_parameters_MODEL()
        self.reset_parameters_TRAIN()
        
    def reset_parameters_MODEL(self):
        # Parameters: MODEL
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        self.SKmodel_CLASS = ExtraTreesRegressor
        self.model_desc = "ExtraTreesRegressor"
        self.metrics = [r2_score, mean_squared_error,mean_absolute_error]
        self.metrics_desc = ["R2_Score", "    mse", "    mae", ]
        self.model_args = {
            'criterion':'mse',
            'max_depth':10, 
            'min_samples_split':2, 
            'min_samples_leaf':3,
            'min_weight_fraction_leaf':0.0, 
            'max_features':0.90, 
            'max_leaf_nodes':None, 
            }
        
    def reset_parameters_PREV_MODEL(self):
        self.PREV_MODEL_SID = [('S7_FL' , 'NN_BxA01'),  # Bx, fish length
                               ('S7_FL' , 'NN_BxA02'),  # Bx, fish length
                               
                               ('S7_FL' , 'NN_AxA10'),  # Bx, fish bbox
                              ]  # PREV_STAGE, PREV_MODEL_ID
        PREV_src_file = ['src/{}_models/{}_{}_model.py'.format(s1, s1, s2) for s1,s2 in self.PREV_MODEL_SID]
        self.PREV_Model = [imp.load_source('', s1).Model() for s1 in PREV_src_file]

    def reset_parameters_TRAIN(self):
        # Parameters: TRAINING
        self.fold_column = 'Fs3'
        self.seed = 0
        self.train_args = {
            'bootstrap':False,
            'random_state':self.seed
            }
        self.n_estimators = 250
        self.n_estimators_per_round = 250
        
    def reset_variables(self):
        # Initializations
        self.dsetID = None
        self.Data = ld.FishDATA()
        self.SKmodel = None
        
        self.output_dir = str(self.Data.path_settings['path_outputs_{}'.format(STAGE)])
        self.stage = STAGE
        self.model_id = MODEL_ID
        self.model_filename_format = '{}_{}_{}_SKmodel'.format(self.stage, self.model_id, '{fold_id}')
        self.path_predictions = os.path.join(self.output_dir, self.model_id)
        self.model_file = None
        self.prev_foldID = None

    def read_image(self, itype, image_id, 
                   frame = 'example',  # int, 'all', 'example'(0)
                               #'all_labeled' --> only if training
                               #'all_train' --> only if training
                   read_targets=False, 
                   use_cache=None, verbose=False):
        '''Custom read_image function for this model.
        '''
        
        start_time_L1 = time.time()

        targets=None
        
        # Read image.
        vidD = self.Data.load_vidDATA(itype, image_id)

        # Read annotations
        df = self.Data.annotations
        mini_df = df[df.video_id == image_id]
        mini_df = mini_df[np.logical_not(np.isnan(mini_df.fish_number))]
        nb_frames = len(mini_df)
        
        
        # Create frames list
        if frame == 'all':
            frames = range(len(vidD.vi))
        elif frame == 'example':
            frames = [0,]
        elif frame == 'all_labeled' and nb_frames > 0:
            frames = mini_df.frame.values.tolist()
        elif frame == 'all_train' and nb_frames > 0:
            i_frames = mini_df.frame.values.tolist()
            frames = [[s,] for s in i_frames]
            frames = [s for ss in frames for s in ss]
            frames = np.unique(np.clip(frames, 0, len(vidD.vi)-1))
        else:
            frames = [int(frame),]
        
        # Start df
        cols = {}
        cols['itype']  = itype
        cols['image_id']  = image_id
        
        # Bx, fish_length
        i_model = -1
        bx_models = 2
        for i in range(bx_models):
            i_model += 1
            Model = self.PREV_Model[i_model] 
            pred= Model.get_predictions(itype, image_id, return_imgs=False, use_cache=use_cache, verbose=verbose)
            pred = pred[:,0]
            
            cols['Bx{}_{}Pred'.format(i_model, i)]   = pred
            cols['Bx{}_{}M3'.format(i_model, i)]     = ld.mov_avg(pred, 1, 1)
            cols['Bx{}_{}M5'.format(i_model, i)]     = ld.mov_avg(pred, 2, 2)
            cols['Bx{}_{}M11'.format(i_model, i)]    = ld.mov_avg(pred, 5, 5)
            cols['Bx{}_{}M0i3max'.format(i_model, i)]= ld.mov_func(pred, 0, 3, np.nanmax)
            cols['Bx{}_{}M3i0max'.format(i_model, i)]= ld.mov_func(pred, 3, 0, np.nanmax)
            cols['Bx{}_{}M0i5max'.format(i_model, i)]= ld.mov_func(pred, 0, 5, np.nanmax)
            cols['Bx{}_{}M5i0max'.format(i_model, i)]= ld.mov_func(pred, 5, 0, np.nanmax)
            cols['Bx{}_{}M0i11max'.format(i_model, i)]= ld.mov_func(pred, 0, 11, np.nanmax)
            cols['Bx{}_{}M11i0max'.format(i_model, i)]= ld.mov_func(pred, 11, 0, np.nanmax)
        
        # Bx, fish_bbox
        i_model += 1
        Model = self.PREV_Model[i_model] 
        pred= Model.get_predictions(itype, image_id, return_imgs=False, use_cache=use_cache, verbose=verbose)
        pred = pred.iloc[:,4:].values
        
        for i in range(pred.shape[1]):
            cols['Bx{}_{}Pred'.format(i_model, i)]   = pred[:,i]
            cols['Bx{}_{}M3'.format(i_model, i)]     = ld.mov_avg(pred[:,i], 1, 1)
            cols['Bx{}_{}M5'.format(i_model, i)]     = ld.mov_avg(pred[:,i], 2, 2)
            cols['Bx{}_{}M11'.format(i_model, i)]    = ld.mov_avg(pred[:,i], 5, 5)

        # Data Frame
        dset_df = pd.DataFrame(cols)
        
        # Convert infinites and nans
        dset_df = dset_df.replace([np.inf, -np.inf, np.nan], 0)

        # Add fetaures
        dset_df = dset_df.assign(frame= np.arange(len(dset_df)))
        dset_df = dset_df.assign(sample_weight= 1/float(len(dset_df)))
        dset_df = dset_df.assign(frames_left = len(dset_df)-dset_df.frame.values)
        

        # Add target
        if read_targets:
            targets = mini_df[['frame',]+['length',]]
        
        if targets is not None:
            dset_df = pd.merge(dset_df, targets, how='left', on='frame')     

        # Filter frames
        dset_df = dset_df.iloc[[s in frames for s in dset_df.frame], :]
        
        if verbose:
            print("Read image {} dataset in {:.2f} s".format(image_id, (time.time() - start_time_L1)/1))
        
        return dset_df

        
    def get_SKmodel(self, model_args=None, SKmodel_CLASS=None):
        
        model_args = self.model_args if model_args is None else model_args
        SKmodel_CLASS = self.SKmodel_CLASS if SKmodel_CLASS is None else SKmodel_CLASS
        
        SKmodel = SKmodel_CLASS()
        for key, value in self.model_args.iteritems():
            setattr(SKmodel, key, value)
        self.SKmodel = SKmodel
        
        return SKmodel
    
    def save_SKmodel(self, SKmodel, filepath, filename):
        from sklearn.externals import joblib
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        joblib.dump(SKmodel, filepath + filename + '.pkl' + '.gz', compress=('gzip', 3)) 
    
    def load_SKmodel(self, model_file):
        from sklearn.externals import joblib
        SKmodel = joblib.load(model_file)
        self.SKmodel = SKmodel
        return SKmodel

    def load_model(self, model_filename, models_path=None, verbose=False):
        models_path = self.output_dir if models_path is None else models_path
        self.model_file = models_path + model_filename + '.pkl' + '.gz'
        if self.SKmodel is None:
            self.get_SKmodel()
        self.SKmodel = self.load_SKmodel(self.model_file)
        if verbose:
            print('  Read model: {}'.format(self.model_file))

    def predict(self, dset):
        '''
        dset: pandas dataframe
        '''
        
        if self.SKmodel is None:
            self.get_SKmodel()
        if self.model_file is None:
            sys.exit("Model not loaded")
        
        #pre-process dataframe
        x_test = dset[[s for s in dset.columns if s not in ['image_id', 'itype', 'length']]]\
                .values.astype(np.float32)
        
        # make batch size = 1
        pred = self.SKmodel.predict(x_test)
        
        # Change predictions dtype
        pred = pred.astype(np.float32)

        return pred
        
    def get_predictions(self, itype, image_id,
                        return_dset = False, avoid_read_model=False, return_score = False, 
                        use_cache=None, force_save=False, verbose=True):
        
        start_time_L1 = time.time()
        use_cache = self.Data.exec_settings['cache'] == "True" if use_cache is None else use_cache
        pred = None
        score = None
        score_txt = 'log_loss'
        
        if use_cache & (not force_save):
            try:
                file_to_load = os.path.join(self.path_predictions, itype, '{}_{}_pred.npy.gz'.format(itype, image_id))
                with gzip.open(file_to_load, 'rb') as f:
                    pred = np.load(f)
                if not return_dset:
                    if verbose:
                        print("Read prediction {}_{} in {:.2f} s".format(itype, image_id, 
                              (time.time() - start_time_L1)/1))
                    return pred
            except:
                if verbose:
                    print("File not in cache")
                    

        dset = self.read_image(itype, image_id, frame = 'all', read_targets=(itype=='train'), verbose=verbose)
        
        if pred is None:
            
            #get model
            if (self.model_file is None) or not avoid_read_model:
                self.dsetID = ld.read_dsetID() if self.dsetID is None else self.dsetID
                fold_id = self.dsetID.loc[(self.dsetID.video_id == image_id) & (self.dsetID.itype == itype), 
                                          self.fold_column]
                fold_id = fold_id.values[0]
                if self.prev_foldID != fold_id:
                    model_filename = self.model_filename_format.format(fold_id=fold_id)
                    self.load_model(model_filename, verbose=verbose)
                    self.prev_foldID = fold_id            
            
            # predict
            pred = self.predict(dset)
            
            # Save cache
            if use_cache|force_save:
                if not os.path.exists(os.path.join(self.path_predictions, itype)):
                    os.makedirs(os.path.join(self.path_predictions, itype))
                file_to_save = os.path.join(self.path_predictions, itype, '{}_{}_pred.npy'.format(itype, image_id))    
                np.save(file_to_save, pred)
                with open(file_to_save, 'rb') as f_in, gzip.open(file_to_save + '.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(file_to_save)                
        
        # evaluate
        try:
            labels = dset.length.values
        except:
            labels = None
            
        if labels is not None:
            from sklearn.metrics import r2_score
            np_labels = labels
            np_preds = pred
            y_true = (np_labels[np.logical_not(np.isnan(np_labels))]).astype(np.float32)
            y_pred = (np_preds[np.logical_not(np.isnan(np_labels))]).astype(np.float32)
            score = r2_score(y_true, y_pred)
        
        if verbose:
            if score is not None:
                print("Read prediction {}_{} ({}: {:.5f}) in {:.2f} s".format(itype, image_id, score_txt, score, 
                      (time.time() - start_time_L1)/1))        
            else:
                print("Read prediction {}_{} in {:.2f} s".format(itype, image_id, (time.time() - start_time_L1)/1))        
        
        if return_dset:
            if return_score:
                return pred, dset, labels, score
            else:
                return pred, dset, labels
            
        if return_score:
            return pred, score
        else:
            return pred
