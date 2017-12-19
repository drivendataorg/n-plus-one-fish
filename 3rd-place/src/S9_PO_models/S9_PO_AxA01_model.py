
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

MODEL_ID = "AxA01"
STAGE    = "S9_PO"


class Model(object):
    
    def __init__(self):
        
        self.reset_variables()
        self.reset_parameters_PREV_MODEL()
        
    def reset_parameters_PREV_MODEL(self):
        self.PREV_MODEL_SID = [('S3_VSEL2', 'SKxET_CxA01'),  # R10_POST 4 FRAMES
                               ('S7_FL' , 'NN_AxA10'),  # Bx, fish bbox
                               ('S3_VSEL2', 'SKxET_DxA01'),       # BEST FRAMES
                              ]  # PREV_STAGE, PREV_MODEL_ID
        PREV_src_file = ['src/{}_models/{}_{}_model.py'.format(s1, s1, s2) for s1,s2 in self.PREV_MODEL_SID]
        self.PREV_Model = [imp.load_source('', s1).Model() for s1 in PREV_src_file]
        
        self.PREV_MODEL_SID_length = [('S2_VSE', 'NN_BxB01'),
                                      ('S8_FLL2', 'SKxET_AxA02'),]
        PREV_src_file = ['src/{}_models/{}_{}_model.py'.format(s1, s1, s2) for s1,s2 in self.PREV_MODEL_SID_length]
        self.PREV_Model_length = [imp.load_source('', s1).Model() for s1 in PREV_src_file]
        
        self.PREV_MODEL_SID_species = [('S6_FCL2', 'SKxET_BxA20'),]
        PREV_src_file = ['src/{}_models/{}_{}_model.py'.format(s1, s1, s2) for s1,s2 in self.PREV_MODEL_SID_species]
        self.PREV_Model_species = [imp.load_source('', s1).Model() for s1 in PREV_src_file]
        
    def reset_variables(self):
        # Initializations
        self.dsetID = None
        self.Data = ld.FishDATA()
    
    def read_video(self, itype, video_id, 
                   read_targets=False, 
                   use_cache=None, verbose=False):
        '''Custom read_image function for this model.
        '''
        
        start_time_L1 = time.time()
        
        # Start data class & varaiables
        Data = self.Data
        targets=None
        
        # Start df
        cols = {}
        cols['itype']  = itype
        cols['video_id']  = video_id
        
        # R10: [+10, -10] frame-score
        i_model = 0  # --------------------
        Model = self.PREV_Model[i_model]
        pred= Model.get_predictions(itype, video_id, return_dset=False, use_cache=use_cache, verbose=verbose)
        pred = pred[:, ...]
        cols['frame']       = range(pred.shape[0])
        cols['R10_Pred']    = pred
        
        x = pred
        Nprev = 2
        Nnxt = 2
        N = Nprev + 1 + Nnxt
        xp = np.concatenate([np.repeat(x[0], Nprev), x, np.repeat(x[-1], Nnxt)])
        kernel = np.array([1,1,1,-1,-1]) # max = -31
        xc = np.convolve(xp, kernel, mode='valid')
        xc = np.clip(xc, -30, 0)
        xc = xc/2.0-10
        cols['R10_Mod']    = xc

        # Bx, fish_bbox
        i_model += 1
        Model = self.PREV_Model[i_model] 
        pred= Model.get_predictions(itype, video_id, return_imgs=False, use_cache=use_cache, verbose=verbose)
        pred = pred.iloc[:,5].values
        cols['BBox_Length']    = pred
        
        # BEST FRAMES
        i_model += 1 # --------------------
        Model = self.PREV_Model[i_model]
        pred= Model.get_predictions(itype, video_id, use_cache=use_cache, verbose=verbose)
        pred = pred[:, 0] if len(pred.shape) == 2 else pred
        cols['Dx_Pred']    = pred        
        
        # LENGTH        
        preds = []
        for Model in self.PREV_Model_length:
            pred= Model.get_predictions(itype, video_id, use_cache=use_cache, verbose=verbose)
            pred = pred[:, 0] if len(pred.shape) == 2 else pred
            preds.append(pred)
        cols['PRED_length']    = np.mean(np.array(preds), axis=0)
        
        # SPECIES
        preds = []
        for Model in self.PREV_Model_species:
            pred= Model.get_predictions(itype, video_id, return_dset=False, use_cache=use_cache, verbose=verbose)
            preds.append(pred)  
        pred = np.mean(np.array(preds), axis=0)
        for i in range(pred.shape[1]):
            cols['PRED_{}'.format(Data.clss_names[i])]  = pred[:,i]
        


        
        # Data Frame
        dset_df = pd.DataFrame(cols)
        
        # Add target
        if read_targets:
            df = self.Data.annotations
            mini_df = df[df.video_id == video_id]
            mini_df = mini_df[np.logical_not(np.isnan(mini_df.fish_number))]
            targets = mini_df.loc[:, ['frame', 'fish_number', 'length']+ self.Data.clss_names]
        
        if targets is None:
            dset_df = dset_df.assign(target = np.nan)
        else:
            dset_df = pd.merge(dset_df, targets, how='left', on='frame')
            #dset_df = dset_df.assign(target = np.nan_to_num(dset_df.target.values))      

        # Convert infinites and nans
        #dset_df = dset_df.replace([np.inf, -np.inf, np.nan], 0)
        
        if verbose:
            print("Read video {} dataset in {:.2f} s".format(video_id, (time.time() - start_time_L1)/1))
        
        return dset_df
        
        return pred

