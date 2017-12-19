
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
from src.lib import FCC_lib_preprocess_v1 as pp
from src.lib import FCC_lib_models_NN_keras_v1 as lm
from src.lib import FCC_lib_train_NN_v1 as lt
from src.lib import FCC_lib_2Dimg_v1 as ltt

MODEL_ID = "SKxET_BxA20"
STAGE    = "S6_FCL2"

class Model(object):
    
    def __init__(self):
        
        self.reset_variables()
        self.reset_parameters_PREV_MODEL()
        self.reset_parameters_MODEL()
        self.reset_parameters_TRAIN()
        #self.reset_parameters_PREDICT()
        
    def reset_parameters_MODEL(self):
        # Parameters: MODEL
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import log_loss
        
        self.SKmodel_CLASS = ExtraTreesClassifier
        self.model_desc = "ExtraTreesClassifier"
        self.metrics = [log_loss, ]
        self.metrics_desc = ["log_loss",]
        self.model_args = {
            'criterion':'entropy',
            'max_depth':None, 
            'min_samples_split':2, 
            'min_samples_leaf':3,
            'min_weight_fraction_leaf':0.0, 
            'max_features':30, 
            'max_leaf_nodes':None, 
            'class_weight': None,
            }
        
    def reset_parameters_PREV_MODEL(self):
        self.PREV_MODEL_SID = [('S5_FC' , 'NN_QxG02'),  # Cx, fish class
                               ('S5_FC' , 'NN_QxJ04'),  # Cx, fish class
                               
                               ('S2_VSE', 'NN_DxB01'),  # Dx, Best Frame Prediction
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
            frames = [[s-1, s, s+1] for s in i_frames]
            frames = [s for ss in frames for s in ss]
            frames = np.unique(np.clip(frames, 0, len(vidD.vi)-1))
        else:
            frames = [int(frame),]
        
        # Start df
        cols = {}
        cols['itype']  = itype
        cols['image_id']  = image_id
        cols['frame']       = range(len(vidD.vi))
        
        # Init variables 
        i_model = -1

        # Cx, fish class
        cx_models=2
        for i in range(cx_models):
            i_model += 1
            Model = self.PREV_Model[i_model] 
            pred= Model.get_predictions(itype, image_id, return_imgs=False, use_cache=use_cache, verbose=verbose)
            for i in range(pred.shape[1]):
                cols['Cx{}_{}Pred'.format(i_model, i)]  = pred[:,i]
                cols['Cx{}_{}M3'.format(i_model, i)]    = ld.mov_avg(pred[:,i], 1, 1)
                cols['Cx{}_{}M5'.format(i_model, i)]    = ld.mov_avg(pred[:,i], 2, 2)
                cols['Cx{}_{}M0i5'.format(i_model, i)]  = ld.mov_avg(pred[:,i], 0, 5)
                cols['Cx{}_{}M5i0'.format(i_model, i)]  = ld.mov_avg(pred[:,i], 5, 0)

        # Dx, best frame prediction
        i_model += 1
        Model = self.PREV_Model[i_model]  
        pred= Model.get_predictions(itype, image_id, use_cache=use_cache, verbose=verbose)
        pred = pred[:, 0] if len(pred.shape) == 2 else pred
        
        cols['DxPred']      = pred
        cols['DxP1']        = np.concatenate([np.repeat(pred[0], 1), pred[:-1]])
        cols['DxP2']        = np.concatenate([np.repeat(pred[0], 2), pred[:-2]])
        cols['DxP3']        = np.concatenate([np.repeat(pred[0], 3), pred[:-3]])
        cols['DxN1']        = np.concatenate([pred[1:], np.repeat(pred[-1], 1)])
        cols['DxN2']        = np.concatenate([pred[2:], np.repeat(pred[-1], 2)])
        cols['DxN3']        = np.concatenate([pred[3:], np.repeat(pred[-1], 3)])
        cols['DxM3']        = ld.mov_avg(pred, 1, 1)
        cols['DxM5']        = ld.mov_avg(pred, 2, 2)
        cols['DxM3i1']      = ld.mov_avg(pred, 3, 1)
        cols['DxM1i3']      = ld.mov_avg(pred, 1, 3)
        

        # Data Frame
        dset_df = pd.DataFrame(cols)
        init_dset = dset_df.loc[:,[s for s in dset_df.columns if s not in \
                              ['itype', 'image_id', 'frames_left', 'target']]].copy()
        
        # Add past data
        tmp_dset = init_dset.copy()
        tmp_dset['frame']  = tmp_dset['frame'].values  - 5
        tmp_dset.columns = [s+'_p5' if s!='frame' else s for s in tmp_dset.columns]
        dset_df = pd.merge(dset_df, tmp_dset, how='left', on='frame')
        
        # Add future data
        tmp_dset = init_dset.copy()
        tmp_dset['frame']  = tmp_dset['frame'].values  + 5
        tmp_dset.columns = [s+'_n5' if s!='frame' else s for s in tmp_dset.columns]
        dset_df = pd.merge(dset_df, tmp_dset, how='left', on='frame')
        
        # Convert infinites and nans
        dset_df = dset_df.replace([np.inf, -np.inf, np.nan], 0)

        # Add fetaures
        dset_df = dset_df.assign(sample_weight= 1/float(len(dset_df)))
        dset_df = dset_df.assign(frames_left = len(dset_df)-dset_df.frame.values)
        

        # Add target
        if read_targets:
            targets = mini_df[['frame',]+self.Data.clss_names]
            
            tmp_dset = targets.copy()
            tmp_dset['frame']  = tmp_dset['frame'].values  +1
            targets = pd.concat([targets, tmp_dset], axis=0)
            tmp_dset['frame']  = tmp_dset['frame'].values  -2
            targets = pd.concat([targets, tmp_dset], axis=0)
        
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
        x_test = dset[[s for s in dset.columns if s not in ['image_id', 'itype',]+self.Data.clss_names]]\
                .values.astype(np.float32)
                
        # make batch size = 1
        pred = self.SKmodel.predict_proba(x_test)
        pred = np.transpose(np.array([s[:,1] for s in pred]), (1,0))                
        
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
        labels = None
        if (itype=='train'):
            labels = dset[self.Data.clss_names].values.astype(np.float32)
            from sklearn.metrics import log_loss
            y_true = labels[~np.isnan(labels)]
            y_pred = pred[~np.isnan(labels)]
            score = log_loss(y_true, y_pred)
        
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
