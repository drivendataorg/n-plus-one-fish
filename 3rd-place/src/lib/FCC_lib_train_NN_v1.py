# For number crunching
import numpy as np
import pandas as pd

# Images
import cv2

# Misc
import random



#################################
# IMAGE TOOLS

def scale_image(img, scale_factor=1.0, method='linear', new_size=None):
    if method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif method == 'linear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'area':
        interpolation = cv2.INTER_AREA
    elif method =='cubic':
        interpolation = cv2.INTER_CUBIC
    
    if ((scale_factor != 1.0) | (new_size is not None)):
        if new_size is None:
            new_size = int(np.round(img.shape[1] * scale_factor, 0)), int(np.round(img.shape[2] * scale_factor, 0))
        tmp = np.zeros((img.shape[0],new_size[0], new_size[1]))
        for i_band in range(img.shape[0]):
            tmp[i_band] = cv2.resize(img[i_band], dsize=(new_size[1], new_size[0]), 
                                     interpolation = interpolation)
        return tmp
    else:
        return img

def standarize_image(img, DT_mean, DT_std, on_site=False):
    tmp_img = img if on_site else np.copy(img)
    for i_band in range(tmp_img.shape[0]):
        tmp_img[i_band] = (tmp_img[i_band] - DT_mean[i_band]) / DT_std[i_band]
    return tmp_img

#####
#    UNET LIKE FUNCTIONS (OUTPUT IS A MASK)

def get_2Dpatches_vA1(image2D, mask2D, image_id, isz2D, slope2D=(0,0), keepNP=(True,True), 
                      mask_present_trs=1, remove_blk=None):

    import math

    #some calculations
    ch_nb, sX, sY = image2D.shape
    tX = int(math.ceil(sX / (isz2D[0]*(1.0-slope2D[0]))))
    tY = int(math.ceil(sY / (isz2D[1]*(1.0-slope2D[1]))))
    iX = np.rint((sX-isz2D[0])/(float(tX-1)+0.00001) * np.arange(0,tX)).astype(np.uint16)
    iY = np.rint((sY-isz2D[1])/(float(tY-1)+0.00001) * np.arange(0,tY)).astype(np.uint16)
    if isz2D[0] > sX:
        iX = [0]
    if isz2D[1] > sY:
        iY = [0]
    
    #generate patches
    img_sta = []  # idx, x,y,x1,y1
    for i_x in iX:
        for i_y in iY:
            #Initiate
            tmp_img_sta = []  # w,h,w1,h1,rotate[1,2,3], mirror[True/False] )
    
            # Check & remove blacks
            if remove_blk is not None:
                thr = remove_blk[0]
                nn2 = remove_blk[1]
                nn1 = image2D[:, i_x:(i_x+isz2D[0]), i_y:(i_y+isz2D[1])].copy()
                ratio = np.mean((np.sum(np.array([nn1[0] < nn2[0], nn1[1] < nn2[1], nn1[2] < nn2[2]]), axis=0) == 3))
                if ratio > thr:
                    continue
            
            # Raw image
            tmp_img_sta.extend([image_id, i_x, i_y, i_x+isz2D[0], i_y+isz2D[1]])
            

            # Filter acording to mask_present_trs
            if mask2D is not None:
                mask_patch = mask2D[:, i_x:(i_x+isz2D[0]), i_y:(i_y+isz2D[1])]
                tmp_img_sta.extend([np.sum(mask_patch, axis=(1,2)).tolist()])
                if (np.sum(mask_patch) >= mask_present_trs):
                    if keepNP[1]:
                        img_sta.extend([tmp_img_sta])
                        continue
                elif keepNP[0]:
                    img_sta.extend([tmp_img_sta])
                    continue
            else:
                tmp_img_sta.extend([None])
                img_sta.extend([tmp_img_sta])     

    img_sta = pd.DataFrame(img_sta, columns=['image_id','x0','y0','x1','y1','msk_sta'])
    
    return img_sta  
    
def generate_2Dpatches_vA2(patches_df, FUNC_read_wrap_img, params={}, 
                           cache=False, stored_cache=[None,None]):
    # For Unet like NNs
    # A2: add scaled patches
    '''
    Input:  img (channel, x, y)
            msk (class, x, y)
    Output: img (batch_size, channel, x, y)
            msk (class, channel, x, y
    patches_df = valid_patches_df
    FUNC_read_wrap_img = read_wrap_img
    params = valid_gen_params
    '''
    
    import math
    
    # Parameters
    DT_zero    = params.get('DT_zero', None)
    rot90      = params.get('rot90', ([],0) )  # ([0,1,2,3], prob[0..1])
    mirror     = params.get('mirror', (False,0) )  # (True/False, prob[0..1])
    move_msk   = params.get('move_msk', 0)  # pixels
    shuffle_ch = params.get('shuffle_ch', False)  # pixels
    predicting = params.get('predicting', False)  # When predicting, don't make any image transformation
    output_size= params.get('output_size', None)
    scales     = params.get('scales', [])
    
    #generate patches    
    patches_df = patches_df.reset_index(drop=True) 
    img_rst = [None]*len(patches_df)
    msk_rst = [None]*len(patches_df)
    unique_image_id = np.unique(patches_df.image_id.values)
    
    for image_id in unique_image_id:  # Iterate first for each image to save time if there are some repetition
        
        if cache & (stored_cache[0] == image_id):
            wrap_img = stored_cache[1]
        else:
            wrap_img = FUNC_read_wrap_img(image_id)    
        img, msk, info = wrap_img
        msk = img if msk is None else msk  # Avoid errors later
        
        for row in patches_df[patches_df.image_id == image_id].itertuples():
            
            # Raw image
            try:
                tmp_img = img[:, row.x0:row.x1, row.y0:row.y1]
                tmp_msk = msk[:, row.x0:row.x1, row.y0:row.y1]
                
            except:
                x0, y0 , x1, y1 = int(row.x0), int(row.y0), int(row.x1), int(row.y1)
                tmp_img = np.ones((img.shape[0], x1-x0, y1-y0)).astype(img.dtype)
                DT_zero = np.zeros(img.shape[0]) if DT_zero is None else DT_zero
                tmp_img = np.einsum('j,jkl->jkl', DT_zero, tmp_img)
                tmp_msk = np.zeros((msk.shape[0], x1-x0, y1-y0)).astype(msk.dtype)
                
                fx0, fy0 = max(0, x0), max(0, y0)
                fx1, fy1 = min(x1, img.shape[-2]), min(y1, img.shape[-1])
                ix0, iy0 = max(0, fx0-x0), max(0, fy0-y0)
                ix1, iy1 = min(x1-x0, ix0+(fx1-fx0)), min(y1-y0, iy0+(fy1-fy0))
                
                tmp_img[:, ix0:ix1, iy0:iy1] = img[:, fx0:fx1, fy0:fy1]
                tmp_msk[:, ix0:ix1, iy0:iy1] = msk[:, fx0:fx1, fy0:fy1]


            # Add scales
            patch_size = (tmp_img.shape[-2], tmp_img.shape[-1])
            for scale in scales:
                dx = int((patch_size[0] / scale - patch_size[0]) / 2.0)
                dy = int((patch_size[1] / scale - patch_size[1]) / 2.0)              
                x0, y0 , x1, y1 = int(row.x0) - dx, int(row.y0) - dy, int(row.x1) + dx, int(row.y1) + dy
                
                tmp_Simg = np.ones((img.shape[0], x1-x0, y1-y0)).astype(img.dtype)
                tmp_Simg = np.einsum('j,jkl->jkl', DT_zero, tmp_Simg)
                
                fx0, fy0 = max(0, x0), max(0, y0)
                fx1, fy1 = min(x1, img.shape[-2]), min(y1, img.shape[-1])
                ix0, iy0 = max(0, fx0-x0), max(0, fy0-y0)
                ix1, iy1 = min(x1-x0, ix0+(fx1-fx0)), min(y1-y0, iy0+(fy1-fy0))

                tmp_Simg[:, ix0:ix1, iy0:iy1] = img[:, fx0:fx1, fy0:fy1] 
                tmp_Simg = (scale_image(tmp_Simg.astype(np.float32), method='linear', 
                                          new_size=patch_size)).astype(img.dtype)
                tmp_img = np.vstack([tmp_img, tmp_Simg])
            
            # Scale to output size
            if output_size is not None:
                tmp_img = (scale_image(tmp_img.astype(np.float32), method='linear', 
                                          new_size=output_size)).astype(img.dtype)
                tmp_msk = (scale_image(tmp_msk.astype(np.float32), method='nearest', 
                                          new_size=output_size)).astype(msk.dtype)
                
            if predicting:
                # Finish
                img_rst[row.Index] = tmp_img
                msk_rst[row.Index] = tmp_msk
                continue
        
            # rot90
            if (len(rot90[0]) > 0) & (np.random.random() < rot90[1]):
                k = np.random.choice(rot90[0])
                tmp_img = np.rot90(tmp_img, k=k, axes=(1,2))
                tmp_msk = np.rot90(tmp_msk, k=k, axes=(1,2))             
                        
            # Mirror
            if (mirror[0]) & (np.random.random() < mirror[1]):
                tmp_img = np.fliplr(tmp_img)
                tmp_msk = np.fliplr(tmp_msk)
            
            # Shuffle channels
            if shuffle_ch:
                ch = range(0,tmp_img.shape[0])
                random.shuffle(ch)
                random.shuffle(ch)
                tmp_img = tmp_img[ch, ...]
            
            # Move Mask
            if move_msk > 0:
                # get desplazament coords
                ang, dist = np.random.sample(2)
                ang = math.radians(360*ang)
                dist = move_msk*dist
                dx = int(np.round(dist*math.cos(ang)))
                dy = int(np.round(dist*math.sin(ang)))
                
                # move img
                new_tmp_msk = np.zeros_like(tmp_msk) 
                new_x0 = max(0, dx)
                new_y0 = max(0, dy)
                new_x1 = min(tmp_msk.shape[1], tmp_msk.shape[1]+dx)
                new_y1 = min(tmp_msk.shape[2], tmp_msk.shape[2]+dy)
                old_x0 = max(0, -dx)
                old_y0 = max(0, -dy)
                old_x1 = min(tmp_msk.shape[1], tmp_msk.shape[1]-dx)
                old_y1 = min(tmp_msk.shape[2], tmp_msk.shape[2]-dy)
                new_tmp_msk[:, new_x0:new_x1, new_y0:new_y1] = tmp_msk[:, old_x0:old_x1, old_y0:old_y1] 
                tmp_msk = new_tmp_msk
            
            # Finish
            img_rst[row.Index] = tmp_img
            msk_rst[row.Index] = tmp_msk
    
    img_rst = np.array(img_rst)
    msk_rst = np.array(msk_rst)
    
    if cache:
        return img_rst, msk_rst, [image_id, wrap_img]
    
    return img_rst, msk_rst
    
    
def batch_generator_vA1(train_patches_df, FUNC_read_wrap_img, FUNC_generate_patches, batch_size, params={}):
    # For Unet NNs (target is a mask)
    
    # Parameters
    seed = params.get('seed', 0)
    shuffle = params.get('shuffle', True)
    flat_output = params.get('flat_output', False)
    predicting = params.get('predicting', False)
    
    sample_index = np.arange(len(train_patches_df))
    number_of_batches = int(np.ceil(len(sample_index)/float(batch_size)))
    if seed is not None:
        np.random.seed(seed)
    
    if shuffle:
        np.random.shuffle(sample_index)
    
    counter = 0
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        x_trn, y_trn = FUNC_generate_patches(train_patches_df.iloc[batch_index], FUNC_read_wrap_img, params)
        
        if flat_output:
           y_trn = np.reshape(y_trn, (y_trn.shape[0], y_trn.shape[1], y_trn.shape[2]*y_trn.shape[3])) 
           y_trn = np.transpose(y_trn, (0,2,1))
        
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
    
    
def paste_preds2Dpatches(predict_patches_df, preds, final_size, isz2D, crop2D=(0.0,0.0)):
    
    pred_msk = np.zeros((preds.shape[-3], final_size[0], final_size[1]))
    pred_msk_ctr = np.zeros_like(pred_msk).astype(np.uint8)
    px_crop2D = (int(crop2D[0] * isz2D[0]), int(crop2D[1] * isz2D[1]))
    lim = (pred_msk.shape[-2], pred_msk.shape[-1])

    for i, i_row in enumerate(predict_patches_df.itertuples()):
        mask2D_patch = preds[i,:,:,:]
        cx0, cx1, cy0, cy1 = i_row.x0, i_row.x1, i_row.y0, i_row.y1
        mx0, mx1, my0, my1 = 0, mask2D_patch.shape[-2], 0, mask2D_patch.shape[-1]
        if cx0 > 0:
            cx0 += px_crop2D[0]
            mx0 += px_crop2D[0]
        if cy0 > 0:
            cy0 += px_crop2D[1]
            my0 += px_crop2D[1]
        if cx1 < lim[0]:
            cx1 -= px_crop2D[0]
            mx1 -= px_crop2D[0]
        if cy1 < lim[1]:
            cy1 -= px_crop2D[1]
            my1 -= px_crop2D[1]
        pred_msk[:, cx0:cx1, cy0:cy1] += mask2D_patch[:, mx0:mx1, my0:my1]
        pred_msk_ctr[:, cx0:cx1, cy0:cy1] += 1

    pred_msk = np.divide(pred_msk, pred_msk_ctr)

    return pred_msk

