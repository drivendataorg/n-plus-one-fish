
# Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import time
import gzip
import shutil
import os
import math
import copy
import imp
import matplotlib.pyplot as plt

from PIL import Image

# Project Library
from src.lib import FCC_lib_data_v1 as ld
from src.lib import FCC_lib_preprocess_v1 as pp
from src.lib import FCC_lib_models_NN_keras_v1 as lm
from src.lib import FCC_lib_train_NN_v1 as lt
from src.lib import FCC_lib_2Dimg_v1 as ltt

MODEL_ID = "NN_AxA10"
STAGE    = "S7_FL"

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
        self.ppFUNC = pp.ppPCH01
        self.pp_patch_size = (448, 224)
        self.prev_crop = (448, 224)
        self.size = (224, 112)
        self.channels = 3
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
        
        self.NNmodel_FUNC = lm.get_UNET_C_r1
        self.isz2D = self.size  # Size of 2D patches
        self.model_size = (self.channels, self.isz2D, 1)  #channels, ISZ, classes
        self.model_desc = "UNET"
        self.model_args = {
            #CONF: 2
            'convs'  : [ [2, 2, 2, 2]     , 2  , [2, 2, 2, 2] ],  #down-mid-up
            'filters': [ [16, 32, 64, 128], 256, [128, 64, 32, 16] ],  #down-mid-up
            'drops'  : [ [0, 0, 0, 0]     , 0  , [0.0, 0.0, 0.0, 0.0] ],

            'type_filters' : '',  # '', ConvDila, ConcDila, inception
            'strides': True,
            'strides_conc' : 'conv',  # conv, pool, input
            'strides_type' : '',  # '', vgg, residual
            'learnable': False,

            'kernel': 3,  
            'batch_norm': True,
            
            'flat_output': False,
            'final_activation': 'sigmoid', 
            'optimizer': Adam(lr=0.001),
            'loss': lambda x,y: lm.similarity_coef(x, y, loss=True, coef=(1,1,1,1),  #jaccard
                                                   axis=[0,-1,-2]),  
            'metrics': [lm.jaccard_coef, lm.dice_coef, lm.binary_cross]  
            
            }
    
    def reset_parameters_TRANSFORMATIONS(self):
        # Parameters: TRANSFORMATIONS
        self.data_transforms = {
            'train': ltt.Compose([
                        ltt.RandomBright((0.8,1.2), p=0.5),
                        ltt.RandomContrast((0.5,1.5), p=0.5),
                        ltt.RandomColor((0.75,1.5), p=0.5),
                        ltt.RandomRotate(10),
                        ltt.RandomShuffleChannels(),
                        ltt.RandomScale(range_factor=0.1, method='PIL'),
                        ltt.RandomMove(radius=50, method='PIL'),
                        ltt.RandomShear((0,0.15), method='PIL', p=0.5),
                        ltt.RandomShear((0.15,0), method='PIL', p=0.5),
                        ltt.RandomVerticalFlip(p=0.5),
                        ltt.RandomHorizontalFlip(p=0.5),
                        
                        ltt.CenterCrop(self.prev_crop),
                        ltt.Scale(self.size),
                        ltt.ToArray(np.float16, np.uint8, img_DT_mean = self.DT_mean, img_DT_std = self.DT_std),
                        ]),
            'valid': ltt.Compose([
                        ltt.CenterCrop(self.prev_crop),
                        ltt.Scale(self.size),
                        ltt.ToArray(np.float16, np.uint8, img_DT_mean = self.DT_mean, img_DT_std = self.DT_std),
                        ]),
            'test': ltt.Compose([
                        ltt.CenterCrop(self.prev_crop),
                        ltt.Scale(self.size),
                        ltt.ToArray(np.float16, img_DT_mean = self.DT_mean, img_DT_std = self.DT_std),
                        ]),
        }
        
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
        self.thr = 0.80
        
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
                               #'all_labeled' --> frames with annotations
                               #'all_train' --> only if training
                   read_labels=False, split_wrap_imgs = False, seed=None, 
                   use_cache=None, verbose=False):
        '''Custom read_image function for this model.
        '''

        start_time_L1 = time.time()
        
        # Start data class & variables
        Data = self.Data
        labels=[] if read_labels else None
        info={}
        
        # Read image.
        vidD = self.Data.load_vidDATA(itype, image_id)
        
        # Read annotations
        df = self.Data.annotations
        mini_df = df[df.video_id == image_id]
        nb_frames = len(mini_df)
        
        # Create frames list
        if frame == 'all':
            frames = range(len(vidD.vi))
        elif frame == 'example':
            frames = [0,]
        elif frame == 'all_labeled' and nb_frames > 0:
            frames = mini_df.frame.values.tolist()
        elif frame == 'all_train' and nb_frames > 0:
            frames = mini_df[mini_df.fish_number >= 0].frame.values.tolist()
        else:
            frames = [int(frame),]
        
        # Read bbox from S1_Model
        use_cache = self.Data.exec_settings['cache'] == "True" if use_cache is None else use_cache
        
        # Read transformation parameters for masks
        if read_labels:
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
            size = self.pp_patch_size
            
        
        # Extract patches
        patches = []
        for i_frame in frames:
            # Only use cache images if frame in annotations
            use_cache_pp = use_cache and (i_frame in mini_df.frame.values.tolist())
            patch = self.ppFUNC(itype, image_id, i_frame,
                                Data = Data, vidD = vidD, S1_Model_DF=self.S1_Model_DF,
                                use_cache = use_cache_pp, verbose = False)
            
            patches.append(patch)
            
            # Read mask
            if read_labels:
                from PIL import Image, ImageDraw
                from PIL import ImageChops
                
                i_row = mini_df[mini_df.frame == i_frame]
                
                if len(i_row) > 0 :
                    i_row = i_row.iloc[0,]
                    im = Image.new('1', (1280, 720))
                    draw = ImageDraw.Draw(im)
                    draw.line((i_row.x1, i_row.y1, i_row.x2, i_row.y2), fill=255, width=11)
                    
                    # center image
                    im = ImageChops.offset(im, -int(xc-im.size[0]/2.0), -int(yc-im.size[1]/2.0))
                    # rotate image
                    im = im.rotate(ang, expand=0)
                    # crop image
                    bbox = (int((im.size[0]-size[0])/2.0), int((im.size[1]-size[1])/2.0), 0, 0)
                    bbox = (bbox[0], bbox[1], bbox[0]+size[0], bbox[1]+size[1])
                    im = im.crop(bbox)
                    
                    labels.append(im)
                
                else:
                    im = Image.new('1', (size[0], size[1]))
                    labels.append(im)

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
        image: img (PIL) or image_id
        '''
        img = image if isinstance(image, Image.Image) else self.read_image(image)[0]
        
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
            pred = self.NNmodel.predict(timgs[start:end])
            preds.append(pred)
        preds = np.vstack(preds)    
        
        # Change predictions dtype
        preds = preds.astype(np.float16)

        return preds  
    
    def get_predictions_raw(self, itype, image_id,
                        return_imgs = False, avoid_read_weights=False, return_score = False, thr=0.8,
                        use_cache=None, force_save=False, verbose=True):
        
        start_time_L1 = time.time()
        use_cache = self.Data.exec_settings['cache'] == "True" if use_cache is None else use_cache
        pred = None
        score = None
        score_txt = 'dice_coef'
        
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
            pp_labels = [self.data_transforms['test'](s1,s1)[1] for s1 in labels]
            select = [np.sum(s1)>0 for s1 in pp_labels]
            np_labels = [s1 for s1, s2 in zip(pp_labels, select) if s2]
            np_labels = np.vstack(np_labels)
            
            np_preds = [s1 for s1, s2 in zip(pred, select) if s2]
            np_preds = np.vstack(np_preds)
            
            score = ld.dice_coef(np_preds, np_labels, thr=thr)
        
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
        
    def get_labels(self, itype, image_id,
                        avoid_read_weights=False,
                        use_cache=None, force_save=False, verbose=True):
        
        vidD = self.Data.load_vidDATA(itype, image_id)
        
        # Predict masks
        pred, imgs, msks, score = self.get_predictions_raw(itype, image_id, 
                                          return_imgs=True, avoid_read_weights=False,
                                          return_score = True, thr = self.thr,
                                          use_cache=use_cache, force_save=force_save, verbose=verbose)
        #scale predictions
        sPred = pred[:,0,...].astype(np.float32)
        sPred = lt.scale_image(sPred, new_size=self.pp_patch_size, method='linear')
        
        # get regions
        import skimage.morphology as morph
        from skimage.measure import regionprops
        pred_thr = np.where(sPred >= self.thr,1.0,0.0)   
        pred_labels = np.array([morph.label(pred_thr[s]) for s in range(pred_thr.shape[0])]).astype(int) 
        regions_lst = [regionprops(pred_labels[s]) for s in range(pred_labels.shape[0])]
        
        # create list
        labels = []
        for ich, regions in enumerate(regions_lst):
            
            if len(regions) == 0:
                center = (np.nan, np.nan)
                ang = np.nan
                length = np.nan
            else:
                region = regions[np.argmax([region.area for region in regions])]  # take biggest region
                center = np.round(region.centroid).astype(int)
                ang = np.round(region.orientation * 180 / math.pi).astype(int)+90
                length = int(math.ceil(region.major_axis_length))
            
            labels.append([image_id, ich, center[0], center[1], ang, length])
        
        labels = pd.DataFrame(labels, columns=['image_id','ich','xc','yc','ang','length'])
        
        return labels, pred, imgs, msks, vidD

    def get_predictions(self, itype, image_id,
                        return_imgs = False, avoid_read_weights=False, return_score = False, 
                        use_cache=None, force_save=False, verbose=True):
        
        start_time_L1 = time.time()
        use_cache = self.Data.exec_settings['cache'] == "True" if use_cache is None else use_cache
        labels = None
        score = None
        pred = None
        score_txt = 'dice_coef'
        
        if use_cache & (not force_save):
            try:
                file_to_load = os.path.join(self.path_predictions, itype, '{}_{}_pred.csv.gz'.format(itype, image_id))
                labels = pd.read_csv(file_to_load)
                if not return_imgs:
                    if verbose:
                        print("Read prediction {}_{} in {:.2f} s".format(itype, image_id, 
                              (time.time() - start_time_L1)/1))
                    return labels
            except:
                if verbose:
                    print("File not in cache")
                    
        imgs, msks, info = self.read_image(itype, image_id, frame = 'all', split_wrap_imgs = True,
                                         read_labels=(itype=='train'), verbose=verbose)
        
        if labels is None:
            
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
            
            #scale predictions
            sPred = pred[:,0,...].astype(np.float32)
            sPred = lt.scale_image(sPred, new_size=self.pp_patch_size, method='linear')
            
            # get regions
            import skimage.morphology as morph
            from skimage.measure import regionprops
            pred_thr = np.where(sPred >= self.thr,1.0,0.0)   
            pred_labels = np.array([morph.label(pred_thr[s]) for s in range(pred_thr.shape[0])]).astype(int) 
            regions_lst = [regionprops(pred_labels[s]) for s in range(pred_labels.shape[0])]
            
            # create list
            labels = []
            for ich, regions in enumerate(regions_lst):
                
                if len(regions) == 0:
                    center = (np.nan, np.nan)
                    ang = np.nan
                    length = np.nan
                else:
                    region = regions[np.argmax([region.area for region in regions])]  # take biggest region
                    center = np.round(region.centroid).astype(int)
                    ang = np.round(region.orientation * 180 / math.pi).astype(int)+90
                    length = int(math.ceil(region.major_axis_length))
                
                labels.append([image_id, ich, center[0], center[1], ang, length])
            
            labels = pd.DataFrame(labels, columns=['image_id','ich','xc','yc','ang','length'])
        
            # Save cache
            if use_cache|force_save:
                if not os.path.exists(os.path.join(self.path_predictions, itype)):
                    os.makedirs(os.path.join(self.path_predictions, itype))
                file_to_save = os.path.join(self.path_predictions, itype, '{}_{}_pred.csv.gz'.format(itype, image_id))    
                labels.to_csv(file_to_save, index=False, compression='gzip')
                        
        
        # evaluate
        if (msks is not None) and (pred is not None):
            pp_labels = [self.data_transforms['test'](s1,s1)[1] for s1 in msks]
            select = [np.sum(s1)>0 for s1 in pp_labels]
            np_labels = [s1 for s1, s2 in zip(pp_labels, select) if s2]
            np_labels = np.vstack(np_labels)
            
            np_preds = [s1 for s1, s2 in zip(pred, select) if s2]
            np_preds = np.vstack(np_preds)
            
            score = ld.dice_coef(np_preds, np_labels, thr=self.thr)
        
        if verbose: 
            if score is not None:
                print("Read prediction {}_{} ({}: {:.5f}) in {:.2f} s".format(itype, image_id, score_txt, score, 
                      (time.time() - start_time_L1)/1))        
            else:
                print("Read prediction {}_{} in {:.2f} s".format(itype, image_id, (time.time() - start_time_L1)/1))        
        
        if return_imgs:
            if return_score:
                return labels, imgs, msks, score
            else:
                return labels, imgs, msks
            
        if return_score:
            return labels,  score
        else:
            return labels
        
    def show_img(self, img, msk=None, pred=None, grid = None, size=(12,6), title=""):
        
        if isinstance(img, list):
            nb_frames = len(img)
            msk = msk if msk is not None else [msk]*nb_frames
            pred = pred if pred is not None else [pred]*nb_frames
        else:
            nb_frames = 1
            img = [img,]
            msk = [msk,]
            pred = [pred,]
            
        # plot images
        if grid is None:
            nbx = int(math.sqrt(nb_frames))
            nby = int(np.ceil(nb_frames/float(nbx)))
        else:
            nbx, nby = grid
        fig,axes = plt.subplots(nbx,nby,figsize=size)
        fig.suptitle(title)
        ax = axes.ravel() if nb_frames>1 else [axes,]
        
        for i in range(nb_frames): 
            rst = None
            cmk = None
            
            if img[i] is not None:
                img_show = np.transpose((np.mean(img[i], axis=0)*127.5+127.5).astype(np.uint8), (1,0))
                img_show = np.transpose(np.stack((img_show,)*3), (1,2,0))
                rst = img_show
        
            if msk[i] is not None:
                msk_show = np.transpose((np.max(msk[i], axis=0)).astype(np.uint8), (1,0))
                msk_show = np.transpose(np.stack([msk_show*255,msk_show*0,msk_show*0]), (1,2,0))
                rst = np.zeros_like(msk_show) if rst is None else rst
                cmk = msk_show
        
            if pred[i] is not None:
                pred_show = np.transpose((np.max(pred[i], axis=0)).astype(np.uint8), (1,0))
                pred_show = np.transpose(np.stack([pred_show*0,pred_show*255,pred_show*0]), (1,2,0))
                rst = np.zeros_like(pred_show) if rst is None else rst
                cmk = pred_show if cmk is None else cmk+pred_show
        
            if rst is not None:
                ax[i].imshow(rst+rst*cmk)
            if i == len(ax)-1:
                break
        plt.show()

    
    def show_patch(self, img, bboxs, grid = None, size=(12,6), title=""):
        
        if isinstance(img, list):
            nb_frames = len(img)
        else:
            nb_frames = 1
            img = [img,]

        # plot images
        if grid is None:
            nbx = int(math.sqrt(nb_frames))
            nby = int(np.ceil(nb_frames/float(nbx)))
        else:
            nbx, nby = grid
        fig,axes = plt.subplots(nbx,nby,figsize=size)
        fig.suptitle(title)
        ax = axes.ravel() if nb_frames>1 else [axes,]
        
        for i in range(nb_frames): 
            if bboxs.iloc[i].length > 0:
                patch = self.Data.extract_patch_PIL(img[i],(bboxs.iloc[i].xc, bboxs.iloc[i].yc), bboxs.iloc[i].ang, 
                                                    size=(bboxs.iloc[i].length, int(bboxs.iloc[i].length/2.0)))
            else:
                patch = img[i]
            ax[i].imshow(patch)
        
            if i == len(ax)-1:
                break
        plt.show()