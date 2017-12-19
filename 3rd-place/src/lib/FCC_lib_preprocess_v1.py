
# Arrays & Dataframes
import numpy as np
import pandas as pd

# Images
from PIL import Image

# Misc
import os
import time
import gzip
import shutil
import imp

# Project Library
from src.lib import FCC_lib_data_v1 as ld


def ppS1B(itype, video_id, 
           Data=None,
           use_cache = True, force_save = False, cache_only_training = True,
           verbose = False):
    """ Read pre-process images from cache """
    
    start_time_L1 = time.time()
    
    # Parameters
    pp_file_name = 'S1B'
    final_size = (352,192)
    fps = 1
    
    # DATA class
    Data = ld.FishDATA() if Data is None else Data
    
    # Read cache
    if cache_only_training and itype != 'train':
        use_cache = False
        force_save = False
        
    if use_cache & (not force_save):
        try:
            file_to_load = os.path.join(Data.pp_data, '{}_{}'.format(pp_file_name, itype), '{}.npy.gz'.format(video_id))
            with gzip.open(file_to_load, 'rb') as f:
                imgs = np.load(f)
            if verbose:
                print("Read video {}_{}_{} in {:.2f} s".format(pp_file_name, itype, video_id, 
                      (time.time() - start_time_L1)/1))
            return imgs
        except:
            if verbose:
                print("File not in cache")

    # Load video
    vidD = Data.load_vidDATA(itype, video_id)
    
    # Get video metadata
    nb_frames = len(vidD.vi)
    vi_fps = vidD.vi._meta['fps']

    # extract frames
    imgs = []
    for i in range(0, nb_frames, int(vi_fps/float(fps))):
        
        # Extract frame
        img = vidD.vi.get_data(i)
        
        # Convert to B&W
        im = Image.fromarray(img)
        im = im.convert('L')
        
        # Resize
        im = im.resize(final_size, Image.BICUBIC)
    
        # Convert to np.array
        img = np.transpose(np.array(im), (1,0))
        img = img[np.newaxis, ...]
        
        imgs.append(img)
    
    imgs = np.array(imgs)
    
    # Save Images
    if use_cache|force_save:
        outputdir = os.path.join(Data.pp_data, '{}_{}'.format(pp_file_name, itype))
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        
        file_to_save = os.path.join(outputdir,'{}.npy'.format(video_id))
        np.save(file_to_save, imgs)
    
        with open(file_to_save, 'rb') as f_in, gzip.open(file_to_save + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(file_to_save)
    
    # Return images
    if verbose:
        print("Read video {}_{}_{} in {:.2f} s".format(pp_file_name, itype, video_id, (time.time() - start_time_L1)/1)) 
        
    return imgs   
 

def ppPCH01(itype, video_id, frame,
           Data=None,
           vidD=None,
           S1_Model_DF=None,
           use_cache = True, force_save = False, cache_only_training = True,
           verbose = False):
    """ Read pre-process images from cache """
    
    start_time_L1 = time.time()
    
    # Parameters
    S1_STAGE, S1_MODEL_ID = 'S1_ROI', 'NN_AxC01'
    pp_file_name = 'PCH01'
    init_patch_size = (448, 224)
    
    # DATA class
    Data = ld.FishDATA() if Data is None else Data
    
    # Filename
    outputdir = os.path.join(Data.pp_data, '{}_{}'.format(pp_file_name, itype))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)       
    filename = os.path.join(outputdir, '{}_{}.jpg'.format(video_id, frame))
            
    # Read cache
    if cache_only_training and itype != 'train':
        use_cache = False
        force_save = False
    
    if use_cache & (not force_save):
        try:
            patch = Image.open(filename)
            if verbose:
                print("Read patch {}_{}_{} in {:.2f} s".format(pp_file_name, itype, video_id, 
                      (time.time() - start_time_L1)/1))
            return patch
        except:
            if verbose:
                print("File not in cache")
                
    # VIDEO class
    vidD = Data.load_vidDATA(itype, video_id) if vidD is None else vidD

    # BBOX
    if S1_Model_DF is None:
        S1_src_file = 'src/{}_models/{}_{}_model.py'.format(S1_STAGE, S1_STAGE, S1_MODEL_ID)
        S1_Model = imp.load_source('', S1_src_file).Model('test')
        filename = os.path.join(S1_Model.path_predictions, '{}_{}_pred.csv.gz'.format(S1_STAGE, S1_MODEL_ID))
        try:
            S1_Model_DF = pd.read_csv(filename)
        except:
            S1_Model_DF = None
    try:
        bbox_df = S1_Model_DF
        bbox_df = bbox_df[bbox_df.image_id == video_id]
        bbox_df = bbox_df[bbox_df.ich == 0]
        bbox_irow = bbox_df.iloc[0]
    except:
        bbox_df, _, _, _, _ = S1_Model.get_labels(itype, video_id, use_cache=use_cache, verbose=verbose)
        bbox_df = bbox_df[bbox_df.image_id == video_id]
        bbox_df = bbox_df[bbox_df.ich == 0]
        bbox_irow = bbox_df.iloc[0]
    
    xc, yc, ang = int(bbox_irow.xc), int(bbox_irow.yc), int(bbox_irow.ang)
    
    # Extract patches
    patch = Data.extract_patch_PIL(Image.fromarray(vidD.vi.get_data(frame)), (xc, yc), ang, size = init_patch_size)
    
    # Save Images
    if use_cache|force_save:
        patch.save(filename, quality=95, optimize=True)
    
    # Return images
    if verbose:
        print("Read patch {}_{}_{} in {:.2f} s".format(pp_file_name, itype, video_id, (time.time() - start_time_L1)/1)) 
        
    return patch  
    