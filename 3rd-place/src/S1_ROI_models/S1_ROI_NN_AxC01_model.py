
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
import matplotlib.pyplot as plt

# Project Library
from src.lib import FCC_lib_data_v1 as ld
from src.lib import FCC_lib_preprocess_v1 as pp
from src.lib import FCC_lib_models_NN_keras_v1 as lm
from src.lib import FCC_lib_train_NN_v1 as lt

MODEL_ID = "NN_AxC01"
STAGE    = "S1_ROI"

class Model(object):
    
    def __init__(self, phase='train'):
        
        self.phase = phase
        self.reset_variables()
        self.reset_parameters_DATA()
        self.reset_parameters_MODEL()
        self.reset_parameters_TRAIN()
        self.reset_parameters_PREDICT()
        
    def reset_parameters_DATA(self):
        # Parameters: DATA
        self.scale = 1.0
        self.channels = 16
        self.DT_mean = np.array([0.5*255] * self.channels)
        self.DT_std = np.array([0.5*255] * self.channels)
        self.DT_zero = (np.array([0] * self.channels) - self.DT_mean) / self.DT_std
        if self.phase == 'train':
            _ = self.Data.get_S1_target_v2()  # Preload targets
    
    def reset_parameters_MODEL(self):
        # Parameters: MODEL
        from keras.optimizers import Adam
        
        self.NNmodel_FUNC = lm.get_UNET_C_r1
        self.isz2D = (352,192)  # Size of 2D patches
        self.model_size = (self.channels, self.isz2D, 2)  #channels, ISZ, classes
        self.model_desc = "UNET"
        self.model_args = {
            #CONF: 2
            'convs'  : [ [2, 2, 2, 2, 2]     , 2  , [2, 2, 2, 2, 2] ],  #down-mid-up
            'filters': [ [32, 64, 128, 256, 512], 1024, [512, 256, 128, 64, 32] ],  #down-mid-up
            'drops'  : [ [0, 0, 0, 0, 0]     , 0  , [0.5, 0.5, 0.5, 0.5, 0.5] ],

            'type_filters' : '',  # '', ConvDila, ConcDila, inception
            'strides': True,
            'strides_conc' : 'conv',  # conv, pool, input
            'strides_type' : 'residual',  # '', vgg, residual
            'learnable': False,

            'kernel': 3,  
            'batch_norm': True,
            
            'flat_output': False,
            'final_activation': 'sigmoid', # softmax sigmoid
            'optimizer': Adam(lr=0.001),
            'loss': lambda x,y: lm.similarity_coef(x, y, loss=True, coef=(1,1,1,1),  #jaccard
                                                   axis=[0,-1,-2]),  
            'metrics': [lm.jaccard_coef, lm.dice_coef, lm.binary_cross]  
            
            }
    
    def reset_parameters_TRAIN(self):
        # Parameters: TRAINING
        self.fold_column = 'F2'
        self.seed = 0
        self.train1_patches_params = {'slope2D':(0.00, 0.00), 'keepNP':(False, True), 'mask_present_trs':1}
        self.train0_patches_params = {'slope2D':(0.00, 0.00), 'keepNP':(True, False), 'mask_present_trs':1}
        self.valid_patches_params  = {'slope2D':(0.00, 0.00), 'keepNP':(True, True), 'mask_present_trs':1} 
        self.patch_get_FUNC = lt.get_2Dpatches_vA1
        self.patch_generator = lt.generate_2Dpatches_vA2
        self.batch_generator = lt.batch_generator_vA1
        self.gen_comm_params = {'DT_zero': self.DT_zero, 
                                'seed': None,
                                'flat_output': False} 
        self.valid_gen_params = self.gen_comm_params.copy()    
        self.train_gen_params = self.gen_comm_params.copy() 
        self.train_gen_params.update({'mirror':(True, 0.5),
                                      'shuffle_ch':True,
                                      })                           
    
    def reset_parameters_PREDICT(self):
        self.predict_patches_params  = {'slope2D':(0.0, 0.0)}  
        self.predict_gen_params = self.gen_comm_params.copy() 
        self.predict_gen_params.update({'shuffle':False, 'predicting':True})
        self.predict_batch_size = 8*4
        self.predict_crop2D = (0.0,0.0)
        
        self.thr = 0.85
        
    def reset_variables(self):
        # Initializations
        self.dsetID = None
        self.Data = ld.FishDATA()
        self.img_raw = None
        self.img = None
        self.msk = None
        self.info = None
        
        self.output_dir = str(self.Data.path_settings['path_outputs_{}'.format(STAGE)])
        self.NNmodel = None
        self.stage = STAGE
        self.model_id = MODEL_ID
        self.weights_format = '{}_{}_{}_weights'.format(self.stage, self.model_id, '{fold_id}')
        self.path_predictions = os.path.join(self.output_dir, self.model_id)
        self.weights_file = None
        self.prev_foldID = None
        
        
    def read_image(self, itype, image_id, read_mask=False, seed=None, batch=1, verbose=False):
        '''Custom read_image function for this model.
        '''
        
        #libraries

        start_time_L1 = time.time()
        
        # Start data class & varaiables
        Data = self.Data
        img=None
        imgs=[]
        msk=None
        info={}
        
        # Read image.
        img_raw = pp.ppS1B(itype, video_id=image_id, Data=Data, 
                           use_cache=True, cache_only_training=False, verbose=False)
        
        # Select frames as channels
        if seed is not None:
            np.random.seed(seed)
        imgs = []
        for i in range(batch):
            img = img_raw[np.random.choice(range(img_raw.shape[0]),self.channels, True), 0, ...]
            if i == 0 and self.channels >=8:  # always take first frame
                frames = np.concatenate([np.array([0]), 
                                         np.random.choice(range(img_raw.shape[0]),self.channels-1, True)])
                np.random.shuffle(frames)
            else:
                frames = np.random.choice(range(img_raw.shape[0]),self.channels, True)
            img = img_raw[frames, 0, ...]
            
            # Preprocess
            img = img.astype(np.float16)
            img = lt.standarize_image(img, self.DT_mean, self.DT_std, on_site=True)
            
            #Batch
            imgs.append(img)
            
        # Read mask
        if read_mask:
            
            df = Data.get_S1_target_v2()
            i_row = df[df.video_id == image_id].iloc[0,]
            
            msk = []
            
            from PIL import Image, ImageDraw
            im = Image.new('1', (1280, 720))
            draw = ImageDraw.Draw(im)
            draw.line((i_row.x1, i_row.y1, i_row.x2, i_row.y2), fill=255, width=50)
            msk.append(np.transpose(np.array(im), (1,0)))
            
            im = Image.new('1', (1280, 720))
            draw = ImageDraw.Draw(im)
            draw.ellipse((i_row.x2-20, i_row.y2-20, i_row.x2+20, i_row.y2+20), fill=255)
            msk.append(np.transpose(np.array(im), (1,0)))
            
            msk = np.array(msk)
           
            # Preprocess
            msk = lt.scale_image(msk.astype(np.float32), new_size=img.shape[-2:], method='linear')
            msk = msk.astype(np.uint8)
        
        # Include usefull information
        info = {'img_raw_size': img_raw.shape}
    
        # wrap results
        if batch>1:
            wrap_img = [imgs, msk, info]
        else:
            wrap_img = [img, msk, info]
        
        if verbose:
            print("Read image {} in {:.2f} s".format(image_id, (time.time() - start_time_L1)/1))
    
        return wrap_img
    
    def read_image_PRED(self, itype, image_id, read_mask=False, verbose=False):
        return self.read_image(itype, image_id, read_mask=read_mask, seed=self.seed, batch=16, verbose=verbose)

    def get_patches(self, imgs_df, read_wrap_imgs, args_dict={}, verbose=True):
        
        #Parameters
        slope2D = args_dict.get('slope2D', (0.0, 0.0, 0.0))
        keepNP = args_dict.get('keepNP', (True, True))
        mask_present_trs = args_dict.get('mask_present_trs', 1)
        scales = args_dict.get('scales', [1.0])
        remove_blk = args_dict.get('remove_blk', None)
        
        patches_df = None
        if verbose:
            itera = tqdm(enumerate(imgs_df.itertuples()), total=len(imgs_df), unit='img', file=sys.stdout)
        else:
            itera = enumerate(imgs_df.itertuples())
        for i, row in itera:
             
            wrap_img = read_wrap_imgs(row.image_id)
            
            for scale in scales:
                bbox = [int(s/scale) for s in self.isz2D]
                tmp_df = self.patch_get_FUNC(wrap_img[0], wrap_img[1], row.image_id, bbox, 
                                              slope2D=slope2D, keepNP=keepNP, 
                                              mask_present_trs=mask_present_trs, remove_blk=remove_blk)
                tmp_df = tmp_df.assign(scale = scale)
                patches_df =  tmp_df if   patches_df is None else  pd.concat([patches_df, tmp_df])                            
    
        return patches_df       
        
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

    def predict(self, image):
        '''
        image: img (np.array) or image_id
        '''
        img = image if (type(image) == np.ndarray) else self.read_image(image)[0]
        
        if self.weights_file is None:
            sys.exit("Weights not loaded")
        if self.NNmodel is None:
            self.get_NNmodel()
        
        predict_patches_df = self.patch_get_FUNC(img, None, 'predict', self.isz2D, 
                                                 slope2D=self.predict_patches_params['slope2D'])
        predict_batch_size = min(len(predict_patches_df), self.predict_batch_size)    

        predict_patches = []
        for i, row in enumerate(predict_patches_df.itertuples()):
            predict_patches.append(img[:,row.x0:row.x1,row.y0:row.y1])
        predict_patches = np.array(predict_patches)
        
        preds = []
        for i in range(0, len(predict_patches), predict_batch_size):
            preds.append(self.NNmodel.predict(predict_patches[i:(i+predict_batch_size)], 
                         batch_size=predict_batch_size))
        preds = np.vstack(preds)
        
        if self.predict_gen_params['flat_output']:
            preds = np.transpose(preds, (0,2,1))
            preds = np.resize(preds, (preds.shape[0], preds.shape[1], self.isz2D[0], self.isz2D[1]))
        
        pred = lt.paste_preds2Dpatches(predict_patches_df, preds, (img.shape[-2], img.shape[-1]), 
                                       self.isz2D, self.predict_crop2D)
        
        pred = pred.astype(np.float16)

        return pred
    
    def predict_BATCH(self, images):
        '''
        images: list(img (np.array)) or image_id
        '''
        
        if self.weights_file is None:
            sys.exit("Weights not loaded")
        if self.NNmodel is None:
            self.get_NNmodel()
        
        
        predict_patches = []
        pp_df = []
        for img in images:
            predict_patches_df = self.patch_get_FUNC(img, None, 'predict', self.isz2D, 
                                                     slope2D=self.predict_patches_params['slope2D'])
            pp_df.append(predict_patches_df)    
            
            for i, row in enumerate(predict_patches_df.itertuples()):
                predict_patches.append(img[:,row.x0:row.x1,row.y0:row.y1])
        predict_patches = np.array(predict_patches)

        predict_batch_size = min(predict_patches.shape[0], self.predict_batch_size)
        preds = []
        for i in range(0, len(predict_patches), predict_batch_size):
            preds.append(self.NNmodel.predict(predict_patches[i:(i+predict_batch_size)], 
                         batch_size=predict_batch_size))
        preds = np.vstack(preds)
    
        if self.predict_gen_params['flat_output']:
            preds = np.transpose(preds, (0,2,1))
            preds = np.resize(preds, (preds.shape[0], preds.shape[1], self.isz2D[0], self.isz2D[1]))
        
        final_preds = []
        init = 0
        for predict_patches_df in pp_df:
            end = init+len(predict_patches_df)
            pred = lt.paste_preds2Dpatches(predict_patches_df, preds[init:end], 
                                           (img.shape[-2], img.shape[-1]), 
                                           self.isz2D, self.predict_crop2D)
        
            pred = pred.astype(np.float16)
            final_preds.append(pred)
            init = end

        return final_preds    
    
    def get_predictions(self, itype, image_id,
                        return_img = False, avoid_read_weights=False, return_score = False, thr=0.8,
                        use_cache=None, force_save=False, verbose=True):
        
        start_time_L1 = time.time()
        use_cache = self.Data.exec_settings['cache'] == "True" if use_cache is None else use_cache
        score = None
        pred = None
        
        if use_cache & (not force_save):
            try:
                file_to_load = os.path.join(self.path_predictions, itype, '{}_{}_pred.npy.gz'.format(itype, image_id))
                with gzip.open(file_to_load, 'rb') as f:
                    pred = np.load(f)
                if not return_img:
                    if verbose:
                        print("Read prediction {}_{} in {:.2f} s".format(itype, image_id, 
                              (time.time() - start_time_L1)/1))
                    return pred, None, None
            except:
                if verbose:
                    print("File not in cache")
                    
        imgs, msk, info = self.read_image_PRED(itype, image_id, read_mask=(itype=='train'),verbose=verbose)
        
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
            preds = self.predict_BATCH(imgs)
            pred = np.max(np.array(preds), axis=0)  ##### MAX!!!
            
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
        if msk is not None:
            score = ld.dice_coef(pred[0], msk[0], thr=thr)
        
        if verbose:
            if score is not None:
                print("Read prediction {}_{} (score: {:.5f}) in {:.2f} s".format(itype, image_id, score, 
                      (time.time() - start_time_L1)/1))        
            else:
                print("Read prediction {}_{} in {:.2f} s".format(itype, image_id, (time.time() - start_time_L1)/1))        
        
        if return_img:
            if return_score:
                return pred, imgs[0], msk, score
            else:
                return pred, imgs[0], msk
            
        if return_score:
            return pred,  score
        else:
            return pred
    
    def get_labels(self, itype, image_id,
                        avoid_read_weights=False,
                        use_cache=None, force_save=False, verbose=True):
        
        
        
        pred, img, msk, score = self.get_predictions(itype, image_id, 
                                          return_img=True, avoid_read_weights=False,
                                          return_score = True, thr = self.thr,
                                          use_cache=use_cache, force_save=force_save, verbose=verbose)
        
        vidD = self.Data.load_vidDATA(itype, image_id)
        
        sPred = pred.astype(np.float32)
        sPred = lt.scale_image(sPred, new_size=(1280,720), method='linear')
        
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
            else:
                region = regions[np.argmax([region.area for region in regions])]  # take biggest region
                center = np.round(region.centroid).astype(int)
                ang = np.round(region.orientation * 180 / math.pi).astype(int)+90
            
            if (itype=='train'):
                # Annotations
                df = self.Data.get_S1_target_v2()
                i_row = df[df.video_id == image_id].iloc[0,]
                T_center = (i_row.xc, i_row.yc)
                T_ang = i_row.ang
                
                # Distance to predicted center
                df = self.Data.annotations
                df = df[df.video_id == image_id]
                df = df.dropna(how="any")
                df = df.assign(dx1 = np.abs(df.x1.values-center[0]))
                df = df.assign(dx2 = np.abs(df.x2.values-center[0]))
                df = df.assign(dy1 = np.abs(df.y1.values-center[1]))
                df = df.assign(dy2 = np.abs(df.y2.values-center[1]))
                df = df.assign(dist1 = np.sqrt(np.power(df.dx1.values, 2)+np.power(df.dy1.values, 2)))
                df = df.assign(dist2 = np.sqrt(np.power(df.dx2.values, 2)+np.power(df.dy2.values, 2)))
                df = df.assign(ang1 = np.abs((np.arctan((df.dy1)/(df.dx1+0.0001)) *180/math.pi)))
                df = df.assign(ang2 = np.abs((np.arctan((df.dy2)/(df.dx2+0.0001)) *180/math.pi)))
                max_dist = np.round(np.max(np.array([df.dist1.values, df.dist2.values])))
                max_ang = np.round(np.max(np.array([df.ang1.values, df.ang2.values])))
                
                center_error = math.ceil(math.sqrt((center[0]-T_center[0])**2 + (center[1]-T_center[1])**2))
                ang_error = abs(ang-T_ang)
                
                # evaluate
                if msk is not None:
                    score = ld.dice_coef(pred[ich], msk[ich], thr=self.thr)
                
                labels.append([image_id, ich, center[0], center[1], ang,
                               T_center[0], T_center[1], T_ang, i_row.max_frame,
                               center_error, ang_error, max_dist, max_ang, score])
            else:
                labels.append([image_id, ich, center[0], center[1], ang])
                
        if (itype=='train'):
            labels = pd.DataFrame(labels, columns=['image_id','ich','xc','yc','ang',
                                                   'Txc','Tyc','Tang','max_frame',
                                                   'c_error', 'ang_error',
                                                   'max_dist', 'max_ang', 'dice_score'])
        else:
            labels = pd.DataFrame(labels, columns=['image_id','ich','xc','yc','ang'])
        
        return labels, pred, img, msk, vidD
    
    def show_img(self, img=None, msk=None, pred=None, size=(12,6), title=""):
        
        rst = None
        cmk = None
        
        if img is not None:
            img_show = np.transpose((np.mean(img, axis=0)*127.5+127.5).astype(np.uint8), (1,0))
            img_show = np.transpose(np.stack((img_show,)*3), (1,2,0))
            rst = img_show
        
        if msk is not None:
            msk_show = np.transpose((np.max(msk, axis=0)).astype(np.uint8), (1,0))
            msk_show = np.transpose(np.stack([msk_show*255,msk_show*0,msk_show*0]), (1,2,0))
            rst = np.zeros_like(msk_show) if rst is None else rst
            cmk = msk_show
        
        if pred is not None:
            pred_show = np.transpose((np.max(pred, axis=0)).astype(np.uint8), (1,0))
            pred_show = np.transpose(np.stack([pred_show*0,pred_show*255,pred_show*0]), (1,2,0))
            rst = np.zeros_like(pred_show) if rst is None else rst
            cmk = pred_show if cmk is None else cmk+pred_show
        
        if rst is not None:
            fig,ax = plt.subplots(1,1,figsize=size)
            ax.imshow(rst+rst*cmk)
            ax.set_title(title) 
            plt.show()
    
    def show_patch(self, labels, vidD, size=(12,6), title=""):
        try:
            frame = labels.iloc[0].max_frame
        except:
            frame = 0
        patch = self.Data.extract_patch(vidD.vi.get_data(frame),
                                        (labels.iloc[0].xc, labels.iloc[0].yc), labels.iloc[0].ang)
        fig,ax = plt.subplots(1,1,figsize=size)
        ax.imshow(np.transpose(patch , (2, 1,0)).astype(np.uint8))
        ax.set_title(title) 
        plt.show()



