
# Libraries
import numpy as np
import pandas as pd
import sys
import time
import gzip
import shutil
import os
import imp
import matplotlib.pyplot as plt
import math

# Project Library
from src.lib import FCC_lib_data_v1 as ld
from src.lib import FCC_lib_models_NN_keras_v1 as lm
from src.lib import FCC_lib_train_NN_v1 as lt
from src.lib import FCC_lib_2Dimg_v1 as ltt

MODEL_ID = "NN_FxB01"
STAGE    = "S2_VSE"

class Model(object):
    
    def __init__(self):
        
        self.reset_variables()
        self.reset_parameters_DATA()
        self.reset_parameters_S1_MODEL()
        self.reset_parameters_MODEL()
        self.reset_parameters_TRANSFORMATIONS()
        self.reset_parameters_TRAIN()
        self.reset_parameters_PREDICT()
        
    def reset_parameters_DATA(self):
        # Parameters: DATA
        self.init_patch_size =(384, 192)
        self.final_patch_size=(72, 48)
        self.size=(64, 48)
        self.channels = 1
        self.DT_mean = np.array([0.5*255] * self.channels)
        self.DT_std = np.array([0.5*255] * self.channels)
        self.DT_zero = (np.array([0] * self.channels) - self.DT_mean) / self.DT_std
    
    def reset_parameters_S1_MODEL(self):
        self.S1_STAGE, self.S1_MODEL_ID = 'S1_ROI', 'NN_AxC01'
        S1_src_file = 'src/{}_models/{}_{}_model.py'.format(self.S1_STAGE, self.S1_STAGE, self.S1_MODEL_ID)
        self.S1_Model = imp.load_source('', S1_src_file).Model('test')
        filename = os.path.join(self.S1_Model.path_predictions, '{}_{}_pred.csv.gz'\
                                .format(self.S1_STAGE, self.S1_MODEL_ID))
        try:
            self.S1_Model_DF = pd.read_csv(filename)
        except:
            self.S1_Model_DF = None
            
    def reset_parameters_MODEL(self):
        # Parameters: MODEL   
        from keras.optimizers import Adam
        
        self.NNmodel_FUNC = lm.get_CNN_C_r0
        self.isz2D = self.size  # Size of 2D patches
        self.model_size = (self.channels, self.isz2D, 1)  #channels, ISZ, classes
        self.model_desc = "Cnn"
        self.model_args = {
            'batch_norm': True,
            'conv_Blks': [[2, 32, True,  0.5, False],  # nb_convs, nb_filters, pool, drop, separable
                          [2, 64, True,  0.5, False],
                          [2, 128, True, 0.5, False],
                          [2, 256, False, 0.5, False],
                          ],                  
            'conv_Blk_args': {'pad': 0,
                              'kernel': 3,
                              'conv_strides': 1,
                              'conv_padding': "same",
                              'pool_size': 2, 
                              'pool_strides': 2,
                              'pool_padding': "same",
                              },
            'dense_Blks': [[512, 0.5],
                           ],            
            'final_activation': 'sigmoid',
            'optimizer': Adam(lr=0.01),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']
            }
    
    def reset_parameters_TRANSFORMATIONS(self):
        # Parameters: TRANSFORMATIONS
        self.data_transforms = {
            'train': ltt.Compose([
                        ltt.RandomRotate(15),
                        ltt.RandomCrop((64, 48)),
                        ltt.RandomVerticalFlip(p=0.5),
                        ltt.RandomHorizontalFlip(p=0.5),
                        ltt.ToArray(np.float16),
                        ]),
            'valid': ltt.Compose([
                        ltt.CenterCrop((64, 48)),
                        ltt.ToArray(np.float16),
                        ]),
            'test': ltt.Compose([
                        ltt.CenterCrop((64, 48)),
                        ltt.ToArray(np.float16),
                        ]),
        }
        self.TT_scale = ltt.Scale(size=self.final_patch_size, method='PIL')
        
    def reset_parameters_TRAIN(self):
        # Parameters: TRAINING
        self.fold_column = 'Fs3'
        self.seed = 0
        self.gen_comm_params = {'seed': None} 
        self.train_gen_params = self.gen_comm_params.copy() 
        self.valid_gen_params = self.gen_comm_params.copy()    
        self.train_gen_params = self.gen_comm_params.copy() 
        self.valid_gen_params.update({'shuffle': False, })                           
    
    def reset_parameters_PREDICT(self): 
        self.predict_gen_params = self.gen_comm_params.copy() 
        self.predict_gen_params.update({'shuffle':False, 'predicting':True})
        self.predict_batch_size = 128
        
    def reset_variables(self):
        # Initializations
        self.dsetID = None
        self.Data = ld.FishDATA()
        self.img_raw = None
        self.img = None
        self.info = None
        
        self.output_dir = str(self.Data.path_settings['path_outputs_{}'.format(STAGE)])
        self.NNmodel = None
        self.stage = STAGE
        self.model_id = MODEL_ID
        self.weights_format = '{}_{}_{}_weights'.format(self.stage, self.model_id, '{fold_id}')
        self.path_predictions = os.path.join(self.output_dir, self.model_id)
        self.weights_file = None
        self.prev_foldID = None
        
        
    def read_image(self, itype, image_id, 
                   frame = 'example',  # int, 'all', 'example'(0 or max_size)
                               #'all_labeled' --> only with annotations
                               #'all_train' --> only if training
                   read_labels=False, split_wrap_imgs = False, seed=None, 
                   use_cache=None, verbose=False):
        '''Custom read_image function for this model.
        '''
        
        start_time_L1 = time.time()
        
        # Initiate data class & variables
        labels=[] if read_labels else None
        info={}
        
        # Read image.
        vidD = self.Data.load_vidDATA(itype, image_id)
        
        # Read bbox from S1_Model
        use_cache = self.Data.exec_settings['cache'] == "True" if use_cache is None else use_cache
        
        try:
            bbox_df = self.S1_Model_DF
            bbox_df = bbox_df[bbox_df.image_id == image_id]
            bbox_df = bbox_df[bbox_df.ich == 0]
            bbox_irow = bbox_df.iloc[0]
        except:
            bbox_df, _, _, _, _ = self.S1_Model.get_labels(itype, image_id, use_cache=use_cache, verbose=verbose)
            bbox_df = bbox_df[bbox_df.image_id == image_id]
            bbox_df = bbox_df[bbox_df.ich == 0]
            bbox_irow = bbox_df.iloc[0]
        
        xc, yc, ang = int(bbox_irow.xc), int(bbox_irow.yc), int(bbox_irow.ang)
        
        try:
            max_frame = int(bbox_irow.max_frame)
        except:
            max_frame = None
        
        # Read annotations
        df = self.Data.annotations
        mini_df = df[df.video_id == image_id]
        nb_frames = len(mini_df)
        
        # Create frames list
        if frame == 'all':
            frames = range(len(vidD.vi))
        elif frame == 'example':
            frames = [0,] if max_frame is None else [max_frame,]
        elif frame == 'all_labeled' and nb_frames > 0:
            frames = mini_df.frame.values.tolist()
        elif frame == 'all_train' and nb_frames > 0:
            frames = mini_df.frame.values.tolist()
            frames = [range(s-5*2, s+5*4, 1) for s in frames]  # from -2s to +4s
            frames = [item for sublist in frames for item in sublist]
            frames = np.unique(np.clip(np.array(frames), 0, len(vidD.vi))).tolist()
            frames = [s for s in range(len(vidD.vi)) if s in frames]  # because weird error reading some frame videos
        else:
            frames = [int(frame),]
    
        # Extract patches
        patches = []
        for i_frame in frames:
            try:
                img_frame = vidD.vi.get_data(i_frame)  # error in edbV6x9C2r5wEUqg.mp4
            except:
                txt_msg = 'Error reading frame {} in file {}'.format(i_frame, image_id)
                print txt_msg
                continue
            patch = self.Data.extract_patch(img_frame,(xc, yc), ang, 
                                            size = self.init_patch_size, convert_BnW=True)
            
            #preprocess
            patch = patch.astype(np.float16)
            patch = lt.standarize_image(patch, self.DT_mean, self.DT_std, on_site=True)
            patch = self.TT_scale(patch)
            
            patches.append(patch)
            
            if read_labels:
                label = mini_df[mini_df.frame == (i_frame-4)][self.Data.clss_none]  # NOTE i_frame-4
                if len(label) == 0:
                    lbl = np.array([0]).astype(np.uint8)
                    labels.append(lbl[0])
                else:
                    labels.append((np.abs(label.values[0][0]-1)).astype(np.uint8))

        
        # Include usefull information
        info = {'meta': vidD.vi._meta}
    
        # wrap results
        if len(patches)>1:
            if split_wrap_imgs:
                wrap_img = [patches, labels, info]
            else:
                wrap_img = [[patches[s1], labels[s1], info] for s1 in range(len(patches))]
        else:
            wrap_img = [patches[0], labels[0], info]
        
        if verbose:
            print("Read image {} in {:.2f} s".format(image_id, (time.time() - start_time_L1)/1))
    
        return wrap_img

    def batch_generator(self, datafeed, batch_size=1, params={}):
        
        # Parameters
        seed = params.get('seed', None)
        shuffle = params.get('shuffle', True)
        predicting = params.get('predicting', False)
        
        sample_index = np.arange(len(datafeed))
        number_of_batches = np.ceil(len(sample_index)/batch_size)
        
        if seed is not None:
            np.random.seed(seed)
        if shuffle:
            np.random.shuffle(sample_index)
        
        counter = 0
        while True:
            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
            x_trn, y_trn = datafeed[batch_index]
            
            # Yield
            counter += 1
            if predicting:
                yield x_trn
            else:
                yield x_trn, y_trn
                
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0  
        
    def get_NNmodel(self, model_size=None, model_args=None, NNmodel_FUNC=None):
        
        model_size = self.model_size if model_size is None else model_size
        model_args = self.model_args if model_args is None else model_args
        NNmodel_FUNC = self.NNmodel_FUNC if NNmodel_FUNC is None else NNmodel_FUNC
        
        NNmodel = NNmodel_FUNC(channels = model_size[0], isz = model_size[1], classes = model_size[2], 
                               args_dict = model_args)
        self.NNmodel = NNmodel
        
        return NNmodel
    
    def load_weights(self, weights_filename, weights_path=None, verbose=False):
        weights_path = self.output_dir if weights_path is None else weights_path
        self.weights_file = '{}{}.hdf5'.format(weights_path, weights_filename)
        if self.NNmodel is None:
            self.get_NNmodel()
        self.NNmodel.load_weights(self.weights_file)
        if verbose:
            print('  Read weights: {}'.format(self.weights_file))

    def predict(self, image, pred_type='test'):
        '''
        image: img (np.array) or image_id
        '''
        img = image if (type(image) == np.ndarray) else self.read_image(image)[0]
        
        if self.NNmodel is None:
            self.get_NNmodel()
        if self.weights_file is None:
            sys.exit("Weights not loaded")
        
        #apply transformations
        timg = self.data_transforms[pred_type](img)
        
        # make batch size = 1
        timg = timg[np.newaxis, ...]
        
        # Predict
        pred = self.NNmodel.predict(timg)
        
        # Change predictions dtype
        pred = pred.astype(np.float16)

        return pred
    
    def predict_BATCH(self, images, pred_type='test', batch_size = None):
        '''
        images: list(img (np.array)) or image_id
        '''
        
        if self.NNmodel is None:
            self.get_NNmodel()
        if self.weights_file is None:
            sys.exit("Weights not loaded")
        
        #apply transformations
        timgs = self.data_transforms[pred_type](images)
        
        # convert to array
        timgs = np.array(timgs)
        
        # predict in batches
        preds = []
        batch_size = self.predict_batch_size if batch_size is None else batch_size
        for start in range(0, timgs.shape[0], batch_size):
            end = min(start+batch_size, timgs.shape[0])
            pred = self.NNmodel.predict_on_batch(timgs[start:end])
            preds.append(pred)
        preds = np.vstack(preds)    
        
        # Change predictions dtype
        preds = preds.astype(np.float16)

        return preds    
    
    def get_predictions(self, itype, image_id,
                        return_imgs = False, avoid_read_weights=False, return_score = False, 
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
                if not return_imgs:
                    if verbose:
                        print("Read prediction {}_{} in {:.2f} s".format(itype, image_id, 
                              (time.time() - start_time_L1)/1))
                    return pred
            except:
                if verbose:
                    print("File not in cache")
                    
        imgs, labels, info = self.read_image(itype, image_id, frame = 'all', split_wrap_imgs = True,
                                         read_labels=(itype=='train'), verbose=verbose)
        
        if pred is None:
            
            #get weights
            if (self.weights_file is None) or not avoid_read_weights:
                self.dsetID = ld.read_dsetID() if self.dsetID is None else self.dsetID
                fold_id = self.dsetID.loc[(self.dsetID.video_id == image_id) & (self.dsetID.itype == itype), 
                                          self.fold_column]
                fold_id = fold_id.values[0]
                if self.prev_foldID != fold_id:
                    weight_file = self.weights_format.format(fold_id=fold_id)
                    self.load_weights(weight_file, verbose=verbose)
                    self.prev_foldID = fold_id            
            
            # predict
            pred = self.predict_BATCH(imgs)
            
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
        if labels is not None:
            from sklearn.metrics import log_loss
            np_labels = np.vstack(labels)[:,0]
            np_preds = pred[:,0]
            y_true = (np_labels[np.logical_not(np.isnan(np_labels))]).astype(np.float32)
            y_pred = (np_preds[np.logical_not(np.isnan(np_labels))]).astype(np.float32)
            score = log_loss(y_true, y_pred)
        
        if verbose:
            if score is not None:
                print("Read prediction {}_{} ({}: {:.5f}) in {:.2f} s".format(itype, image_id, score_txt, score, 
                      (time.time() - start_time_L1)/1))        
            else:
                print("Read prediction {}_{} in {:.2f} s".format(itype, image_id, (time.time() - start_time_L1)/1))        
        
        if return_imgs:
            if return_score:
                return pred, imgs, labels, score
            else:
                return pred, imgs, labels
            
        if return_score:
            return pred,  score
        else:
            return pred
        
    def get_predictions_BATCH(self, itype_list, image_id_list, imgs_list, batch_size = None, verbose=False):
        '''
        Predict from a list of imgs (outputs from self.read_image)
        '''
        
        for itype, image_id, imgs in zip(itype_list, image_id_list, imgs_list):
            
            #get weights
            if (self.weights_file is None):
                self.dsetID = ld.read_dsetID() if self.dsetID is None else self.dsetID
                fold_id = self.dsetID.loc[(self.dsetID.video_id == image_id) & (self.dsetID.itype == itype), 
                                          self.fold_column]
                fold_id = fold_id.values[0]
                if self.prev_foldID != fold_id:
                    weight_file = self.weights_format.format(fold_id=fold_id)
                    self.load_weights(weight_file, verbose=False)
                    self.prev_foldID = fold_id            
            
            # predict
            pred = self.predict_BATCH(imgs, batch_size = batch_size)
            
            # Save cache
            if not os.path.exists(os.path.join(self.path_predictions, itype)):
                os.makedirs(os.path.join(self.path_predictions, itype))
            file_to_save = os.path.join(self.path_predictions, itype, '{}_{}_pred.npy'.format(itype, image_id))    
            np.save(file_to_save, pred)
            with open(file_to_save, 'rb') as f_in, gzip.open(file_to_save + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(file_to_save)


    def show_imgs(self, imgs, labels=None, preds=None, grid = None, size=(12,6), title=""):
        nb_frames = len(imgs)
        
        imgs_show = [np.transpose((s[0]*125.7+127.5).astype(np.uint8), (1,0)) for s in imgs]
        
        # plot images
        if grid is None:
            nbx = int(math.sqrt(nb_frames))
            nby = int(np.ceil(nb_frames/float(nbx)))
        else:
            nbx, nby = grid
        fig,axes = plt.subplots(nbx,nby,figsize=size)
        fig.suptitle(title)
        ax = axes.ravel()
        
        for i, img in enumerate(imgs_show):
            ax[i].imshow(img, cmap='gray')
            try:
                i_label = labels[i]
            except:
                i_label = np.nan
            try:
                i_pred = preds[i][0]
            except:
                i_pred = np.nan
            ititle = 'True: {:.0f} - Predicted: {:.4f}'.format(i_label, i_pred)
            ax[i].set_title(ititle) 
            if i == len(ax)-1:
                break
        plt.show()
    
    def show_preds(self, preds, labels=None, size=(12,6), title=""):
        
        preds_data = preds[:,0]
        if labels is not None:
            labels_data = np.nan_to_num(np.vstack(labels)[:,0]*-.1)+1.1
            data = np.transpose(np.vstack([preds_data, labels_data]), (1,0))
        else:
            data = np.transpose(np.vstack([preds_data, ]), (1,0))
        
        fig,ax = plt.subplots(1,1,figsize=size)
        ax.plot(data)
        ax.set_title(title) 
        plt.show()