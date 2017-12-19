# For number crunching
import numpy as np
import math

import random
from PIL import Image
from PIL import ImageEnhance
from scipy import ndimage
import cv2

'''
### Transformation(seed = None, channels = None)
# RandomVerticalFlip(p=0.5)
# RandomHorizontalFlip(p=0.5)
# RandomNormRotation(rot = [0, 90, 180, 270], p=1)
# Scale(factor=0.5)
# RandomScale(range_size = None, range_factor = None, p=1)
# Crop( bbox=(50,50,100,100) )
# CenterCrop( size=(100,100))
# RandomCrop( size=(100,100), p=1)
# Rotate(ang=None)
# RandomRotate(factor_ang=10)
# Move(XYmove=(0,0))
# RandomMove(XYmax_move=(0,0)) / RandomMove(radius=0, inner_circle=True)
# Shear(XYshear=(0.0, 0.0))
# RandomShear(XYmax_shear=(0,0)) / RandomShear(radius=0, inner_circle=True)
# RandomShuffleChannels(p=1.0)
# ToArray(dtype)
'''

def standarize_image(img, DT_mean, DT_std, on_site=False):
    tmp_img = img if on_site else np.copy(img)
    for i_band in range(tmp_img.shape[0]):
        tmp_img[i_band] = (tmp_img[i_band] - DT_mean[i_band]) / DT_std[i_band]
    return tmp_img

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, msk=None):
        if msk is not None:
            for t in self.transforms:
                img, msk = t(img, msk)
            return img, msk
        
        for t in self.transforms:
            img = t(img)
        return img
        
    
class Transformation(object):
    
    def __init__(self, seed = None, channels = None ):
        '''
        Args:
            channels: None, 'first', 'last'
        '''

        if seed is not None:
            random.seed(seed)
            
        assert channels in [None, 'first', 'last']
        self.channels = channels
        
        self.p = 1  # probability
    
    def __call__(self, img, msk=None):
        """
        Args:
            img: Image or images to be transformed.
                 - PIL Image
                 - numpy array (channels, x, y) or (x, y, channels)
                 - list of images
                 - numpy array (imgs, channels, x, y) or (imgs, x, y, channels)
        Returns:
            Image or images transformed.
        """
        if isinstance(img, list):  # List of images
            result_img = []
            result_msk = []
            for i in range(len(img)):
                if msk is not None:
                    result = self.transform(img[i], msk[i]) 
                    result_img.append(result[0])
                    result_msk.append(result[1])
                else:
                    result_img.append(self.transform(img[i]))
            if msk is not None:
                return result_img, result_msk
            return result_img
            
        
        if type(img) == np.ndarray and len(img.shape) > 3 :  # nd array of multiple images        
            result_img = []
            result_msk = []
            for i in range(img.shape[0]):
                if msk is not None:
                    result = self.transform(img[i], msk[i]) 
                    result_img.append(result[0])
                    result_msk.append(result[1])
                else:
                    result_img.append(self.transform(img[i]))
            if msk is not None:
                return np.array(result_img), np.array(result_msk)
            return np.array(result_img)
        
        return self.transform(img, msk)
    
    def get_channels(self, img):
        # To automaticly detect the channels position, we will assume that nb_ch is smaller the img size
        if img.shape[0] < img.shape[2]:
            channels = 'first'
        else:
            channels = 'last'
        self.channels = channels  # Save, so assume that will not change this para meter for this object
        return channels
            
    def transform(self, img, msk=None):
        
        if random.random() < self.p:
                
            if 'PIL' in str(type(img)):  # PIL Image
                return self.PIL_transf(img, msk)
            
            # Asumme img is np.ndarray class
            self.channels = self.channels or self.get_channels(img)
            return self.NPY_transf(img, msk)
            
        if msk is not None:
            return img, msk
        return img
    
    def PIL_transf(self, img, msk=None):
        if msk is not None:
            return img, msk
        return img
    
    def NPY_transf(self, img, msk=None):
        if msk is not None:
            return img, msk
        return img


class RandomVerticalFlip(Transformation):
    """Vertically flip the given 2D image randomly"""
    
    def __init__(self, p = 0.5, *args, **kwargs):
        '''
        Args:
            p: probability
            channels: None, 'first', 'last'
        '''
        super(RandomVerticalFlip, self).__init__(*args, **kwargs)
        
        assert p >= 0 and p <= 1
        self.p = p
        
    def PIL_transf(self, img, msk=None):
        if msk is not None:
            return img.transpose(Image.FLIP_LEFT_RIGHT), msk.transpose(Image.FLIP_LEFT_RIGHT)
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    
    def NPY_transf(self, img, msk=None):
        axis = 1 if self.channels == 'first' else 0
        if msk is not None:
            return np.flip(img, axis=axis), np.flip(msk, axis=axis)
        return np.flip(img, axis=axis)
    

class RandomHorizontalFlip(Transformation):
    """Horizontally flip the given 2D image randomly"""
    
    def __init__(self, p = 0.5, *args, **kwargs):
        '''
        Args:
            p: probability
        '''
        super(RandomHorizontalFlip, self).__init__(*args, **kwargs)
        
        assert p >= 0 and p <= 1
        self.p = p
        
    def PIL_transf(self, img, msk=None):
        if msk is not None:
            return img.transpose(Image.FLIP_TOP_BOTTOM), msk.transpose(Image.FLIP_TOP_BOTTOM)
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    
    def NPY_transf(self, img, msk=None):
        axis = 2 if self.channels == 'first' else 1
        if msk is not None:
            return np.flip(img, axis=axis), np.flip(msk, axis=axis)
        return np.flip(img, axis=axis)


class RandomNormRotation(Transformation):
    """Normal rotation of the given 2D image randomly"""
    
    def __init__(self, rot = [0, 90, 180, 270], p = 1, *args, **kwargs):
        '''
        Args:
            rot: list of possible rotations, in degrees
            p: probability
        '''
        super(RandomNormRotation, self).__init__(*args, **kwargs)
        
        assert [s in [0, 90, 180, 270] for s in rot]
        self.rot = rot
        PIL_dict = {0:None, 90:Image.ROTATE_270, 180:Image.ROTATE_180, 270:Image.ROTATE_90}
        self.PIL_rot = [PIL_dict[s] for s in self.rot]
        NPY_dict = {0:None, 90:1, 180:2, 270:3}
        self.NPY_rot = [NPY_dict[s] for s in self.rot]
        
        assert p >= 0 and p <= 1
        self.p = p
        
    def PIL_transf(self, img, msk=None):
        rnd_rot = random.choice(self.PIL_rot)
        if rnd_rot:
            if msk is not None:
                return img.transpose(rnd_rot), msk.transpose(rnd_rot)
            return img.transpose(rnd_rot)
        if msk is not None:
            return img, msk
        return img

    def NPY_transf(self, img, msk=None):
        axes = (1,2) if self.channels == 'first' else (0,1)
        rnd_rot = random.choice(self.NPY_rot)
        if rnd_rot:
            if msk is not None:
                return np.rot90(img, rnd_rot, axes=axes), np.rot90(msk, rnd_rot, axes=axes)
            return np.rot90(img, rnd_rot, axes=axes)
        if msk is not None:
            return img, msk
        return img


class Scale(Transformation):
    """Scale the given 2D image"""
    
    def __init__(self, size = None, factor = None, 
                 img_interpolation='cubic', msk_interpolation='nearest', 
                 method='ndimage', *args, **kwargs):
        '''
        Args:
            size: final size, number or 2Dtuple
            factor: scale factor, float
            interpolation: 'nearest', 'linear', 'cubic', 'antialias', 'area'
            method: 'ndimage', 'PIL', 'CV2'  only when img is an array
        '''
        super(Scale, self).__init__(*args, **kwargs)
        
        assert not ((size and factor) or (not size and not factor))
        self.size = size
        self.factor = float(factor) if factor is not None else factor
        self.img_interp = img_interpolation
        self.msk_interp = msk_interpolation
        
        PIL_dict = {'nearest': Image.NEAREST, 'linear': Image.BILINEAR, 
                     'cubic': Image.BICUBIC, 'antialias': Image.ANTIALIAS,
                     'area': None}
        self.PIL_itrp = PIL_dict[self.img_interp]
        self.PIL_itrp_msk = PIL_dict[self.msk_interp]
        
        ndimage_dict = {'nearest': 0, 'linear': 1, 
                     'cubic': 2, 'antialias': None,
                     'area': None}
        self.ndimage_itrp = ndimage_dict[self.img_interp]
        self.ndimage_itrp_msk = ndimage_dict[self.msk_interp]
        
        CV2_dict = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 
                     'cubic': cv2.INTER_CUBIC, 'antialias': None,
                     'area': cv2.INTER_AREA}
        self.CV2_itrp = CV2_dict[self.img_interp]
        self.CV2_itrp_msk = CV2_dict[self.msk_interp]
        
        assert method in ['PIL', 'ndimage', 'CV2']
        self.method = method
    
    def PIL_Scale(self, img, (ow,oh), interp):
        return img.resize((ow, oh), interp) 
    
    def CV2_Scale(self, img, (ow, oh), interp):
        return cv2.resize(img, (ow, oh), interp)

    def NPY_transf_PIL(self, img, msk=None):
        if self.channels == 'first':
            axis = 0
            w, h = img.shape[1], img.shape[2]
        else:
            axis = 2
            w, h = img.shape[0], img.shape[1] 
        
        if self.factor is not None:
            ow = w * self.factor
            oh = h * self.factor
        elif type(self.size) == tuple:
            ow, oh = self.size
        else:
            ow, oh = self.size, self.size

        ow, oh = int(round(ow, 0)), int(round(oh, 0))
        if w == ow and h == oh:
            if msk is not None:
                return img, msk
            return img
        
        result_img = []
        if self.channels == 'first':
            for i_ch in range(img.shape[axis]):
                tmp_img = img[i_ch,:,:]
                tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                tmp_rst = self.PIL_Scale(tmp_im, (ow,oh), self.PIL_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
        else:
            for i_ch in range(img.shape[axis]):
                tmp_img = img[:,:,i_ch]
                tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                tmp_rst = self.PIL_Scale(tmp_im, (ow,oh), self.PIL_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
            result_img = np.transpose(result_img, (1,2,0))
                
        if msk is not None:
            result_msk = []
            if self.channels == 'first':
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[i_ch,:,:]
                    tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                    tmp_rst = self.PIL_Scale(tmp_im, (ow,oh), self.PIL_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
            else:
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[:,:,i_ch]
                    tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                    tmp_rst = self.PIL_Scale(tmp_im, (ow,oh), self.PIL_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
                result_msk = np.transpose(result_msk, (1,2,0))
                
            return result_img, result_msk
        
        return result_img

    def NPY_transf_CV2(self, img, msk=None):
        if self.channels == 'first':
            axis = 0
            w, h = img.shape[1], img.shape[2]
        else:
            axis = 2
            w, h = img.shape[0], img.shape[1] 
        
        if self.factor is not None:
            ow = w * self.factor
            oh = h * self.factor
        elif type(self.size) == tuple:
            ow, oh = self.size
        else:
            ow, oh = self.size, self.size

        ow, oh = int(round(ow, 0)), int(round(oh, 0))
        if w == ow and h == oh:
            if msk is not None:
                return img, msk
            return img
        
        result_img = []
        if self.channels == 'first':
            for i_ch in range(img.shape[axis]):
                tmp_img = img[i_ch,:,:]
                tmp_im = np.transpose(tmp_img.astype(np.float32), (1,0))
                tmp_rst = self.CV2_Scale(tmp_im, (ow,oh), self.CV2_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
        else:
            for i_ch in range(img.shape[axis]):
                tmp_img = img[:,:,i_ch]
                tmp_im = np.transpose(tmp_img.astype(np.float32), (1,0))
                tmp_rst = self.CV2_Scale(tmp_im, (ow,oh), self.CV2_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
            result_img = np.transpose(result_img, (1,2,0))
                
        if msk is not None:
            result_msk = []
            if self.channels == 'first':
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[i_ch,:,:]
                    tmp_im = np.transpose(tmp_img.astype(np.float32), (1,0))
                    tmp_rst = self.CV2_Scale(tmp_im, (ow,oh), self.CV2_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
            else:
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[:,:,i_ch]
                    tmp_im = np.transpose(tmp_img.astype(np.float32), (1,0))
                    tmp_rst = self.CV2_Scale(tmp_im, (ow,oh), self.CV2_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
                result_msk = np.transpose(result_msk, (1,2,0))
                
            return result_img, result_msk
        
        return result_img
    
    def NPY_transf_ndimage(self, img, msk=None):
        if self.channels == 'first':
            axis = 0
            w, h = img.shape[1], img.shape[2]
        else:
            axis = 2
            w, h = img.shape[0], img.shape[1] 
        
        if self.factor is not None:
            fw = self.factor
            fh = self.factor
        elif type(self.size) == tuple:
            fw = self.size[0]/float(w)
            fh = self.size[1]/float(h)
        else:
            fw = self.size/float(w)
            fh = self.size/float(h)

        if fw == 1.0 and fh == 1.0:
            if msk is not None:
                return img, msk
            return img
        
        result_img = []
        if self.channels == 'first':
            for i_ch in range(img.shape[axis]):
                tmp_img = img[i_ch,:,:]
                tmp_img = tmp_img.astype(np.float32)
                tmp_rst = ndimage.zoom(tmp_img, (fw, fh), order=self.ndimage_itrp, mode='constant')
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img)).astype(img.dtype)
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
        else:
            for i_ch in range(img.shape[axis]):
                tmp_img = img[:,:,i_ch]
                tmp_img = tmp_img.astype(np.float32)
                tmp_rst = ndimage.zoom(tmp_img, (fw, fh), order=self.ndimage_itrp, mode='constant')
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img)).astype(img.dtype)
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
            result_img = np.transpose(result_img, (1,2,0))
                
        if msk is not None:
            result_msk = []
            if self.channels == 'first':
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[i_ch,:,:]
                    tmp_img = tmp_img.astype(np.float32)
                    tmp_rst = ndimage.zoom(tmp_img, (fw, fh), order=self.ndimage_itrp_msk, mode='constant')
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img)).astype(msk.dtype)
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
            else:
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[:,:,i_ch]
                    tmp_img = tmp_img.astype(np.float32)
                    tmp_rst = ndimage.zoom(tmp_img, (fw, fh), order=self.ndimage_itrp_msk, mode='constant')
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img)).astype(msk.dtype)
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
                result_msk = np.transpose(result_msk, (1,2,0))
                
            return result_img, result_msk
        
        return result_img
    
    def PIL_transf(self, img, msk=None):
        w, h = img.size
        if self.factor is not None:
            ow = w * self.factor
            oh = h * self.factor
        elif type(self.size) == tuple:
            ow, oh = self.size
        else:
            ow, oh = self.size, self.size
            
        ow, oh = int(round(ow, 0)), int(round(oh, 0))
        if w == ow and h == oh:
            if msk is not None:
                return img, msk
            return img
        if msk is not None:
            return self.PIL_Scale(img, (ow,oh), self.PIL_itrp), self.PIL_Scale(msk, (ow,oh), self.PIL_itrp_msk)
        return self.PIL_Scale(img, (ow,oh), self.PIL_itrp)       

    def NPY_transf(self, img, msk=None):
        if self.method == 'PIL':
            return self.NPY_transf_PIL(img, msk)
        elif self.method == 'ndimage':
            return self.NPY_transf_ndimage(img, msk)
        elif self.method == 'CV2':
            return self.NPY_transf_CV2(img, msk)


class RandomScale(Scale):
    """Scale the given 2D image randomly"""

    def __init__(self, range_size = None, range_factor = None, p = 1, *args, **kwargs):
        '''
        Args:
            range_size: 2Dtuple (max, min) or ((max, min), (max, min))
            range_factor: scale factor, float or 2Dtuple, +- value
            interpolation: 'nearest', 'linear', 'cubic', 'antialias', 'area'
        '''
        
        assert not ((range_size and range_factor) or (not range_size and not range_factor))
        self.range_size = range_size
        self.range_factor = range_factor
                
        super(RandomScale, self).__init__(size=None, factor=1.0, *args, **kwargs)
        
        assert p >= 0 and p <= 1
        self.p = p
        
    def __call__(self, img, msk=None):
        
        if self.range_factor is not None:
            self.factor = 1 + random.uniform(-self.range_factor, self.range_factor)
            self.size = None
        elif type(self.range_size[0]) != tuple:
            self.size = random.randint(int(self.range_size[0]), int(self.range_size[1]))
            self.factor = None
        else:
            self.size = (random.randint(int(self.range_size[0][0]), int(self.range_size[0][1])), 
                         random.randint(int(self.range_size[1][0]), int(self.range_size[1][1])))
            self.factor = None
        
        return super(RandomScale, self).__call__(img, msk)

        
        
class Crop(Transformation):
    """Crop the given 2D image"""
    
    def __init__(self, bbox, *args, **kwargs):
        '''
        Args:
            bbox: (x0, y0, x1, y1), in pixels
        '''
        
        super(Crop, self).__init__(*args, **kwargs)
        
        self.bbox = bbox
    
    def get_bbox(self, img_size):
        return self.bbox
    
    def PIL_transf(self, img, msk=None):
        bbox = self.get_bbox(img.size)
        if msk is not None:
            return img.crop(bbox), msk.crop(bbox)
        else:
            return img.crop(bbox)
        
    def NPY_transf(self, img, msk=None):
        if self.channels == 'first':
            x0, y0, x1, y1 = self.get_bbox((img.shape[1], img.shape[2]))
            if msk is not None:
                return img[:, x0:x1, y0:y1], msk[:, x0:x1, y0:y1]
            else:
                return img[:, x0:x1, y0:y1]
        else:
            x0, y0, x1, y1 = self.get_bbox((img.shape[0], img.shape[1]))
            if msk is not None:
                return img[x0:x1, y0:y1, :], msk[x0:x1, y0:y1, :],
            else:
                return img[x0:x1, y0:y1, :]


class CenterCrop(Crop):
    """Crop the given 2D image randomly"""
    
    def __init__(self, size, *args, **kwargs):
        '''
        Args:
            size: (x, y), in pixels
        '''
        
        super(CenterCrop, self).__init__(bbox = None, *args, **kwargs)
        
        self.size = size if type(size) == tuple else (size, size)
    
    def get_bbox(self, img_size):
        x0 = int((img_size[0]-self.size[0])/2.0)
        y0 = int((img_size[1]-self.size[1])/2.0)
        x1 = x0 + self.size[0]
        y1 = y0 + self.size[1]
        return x0, y0, x1, y1
    

class RandomCrop(Crop):
    """Crop the given 2D image randomly"""
    
    def __init__(self, size, p = 1, *args, **kwargs):
        '''
        Args:
            size: (x, y), in pixels
        '''
        
        super(RandomCrop, self).__init__(bbox = None, *args, **kwargs)
        
        self.size = size if type(size) == tuple else (size, size)
        
        assert p >= 0 and p <= 1
        self.p = p
    
    def get_bbox(self, img_size):
        x0 = random.randint(0, img_size[0]-self.size[0])
        y0 = random.randint(0, img_size[1]-self.size[1])
        x1 = x0 + self.size[0]
        y1 = y0 + self.size[1]
        return x0, y0, x1, y1


class Fit(Transformation):
    """Scale the given 2D image"""
    
    def __init__(self, size = None, 
                 img_interpolation='cubic', msk_interpolation='nearest', 
                 method='ndimage', *args, **kwargs):
        '''
        Args:
            size: final size, number or 2Dtuple
            factor: scale factor, float
            interpolation: 'nearest', 'linear', 'cubic', 'antialias', 'area'
            method: 'ndimage', 'PIL', 'CV2'  only when img is an array
        '''
        super(Fit, self).__init__(*args, **kwargs)
        
        self.size = size
        self.img_interp = img_interpolation
        self.msk_interp = msk_interpolation
        
        PIL_dict = {'nearest': Image.NEAREST, 'linear': Image.BILINEAR, 
                     'cubic': Image.BICUBIC, 'antialias': Image.ANTIALIAS,
                     'area': None}
        self.PIL_itrp = PIL_dict[self.img_interp]
        self.PIL_itrp_msk = PIL_dict[self.msk_interp]
        
        ndimage_dict = {'nearest': 0, 'linear': 1, 
                     'cubic': 2, 'antialias': None,
                     'area': None}
        self.ndimage_itrp = ndimage_dict[self.img_interp]
        self.ndimage_itrp_msk = ndimage_dict[self.msk_interp]
        
        CV2_dict = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 
                     'cubic': cv2.INTER_CUBIC, 'antialias': None,
                     'area': cv2.INTER_AREA}
        self.CV2_itrp = CV2_dict[self.img_interp]
        self.CV2_itrp_msk = CV2_dict[self.msk_interp]
        
        assert method in ['PIL', 'ndimage', 'CV2']
        self.method = method

    def get_size(self, img_size):

        factor = max([(img_size[0]/ float( self.size[0])), (img_size[1]/ float( self.size[1]))])
        ow = img_size[0] / factor
        oh = img_size[1] / factor
        ow, oh = int(round(ow, 0)), int(round(oh, 0))
        
        x0 = int((img_size[0]/ factor-self.size[0])/2.0)
        y0 = int((img_size[1]/ factor-self.size[1])/2.0)
        x1 = x0 + self.size[0]
        y1 = y0 + self.size[1]
        
        return (ow, oh), (x0, y0, x1, y1)

    def PIL_Fit(self, img, (ow,oh), interp):
        (ow, oh), bbox = self.get_size(img.size)
        tmp = img.resize((ow, oh), interp) 
        return tmp.crop(bbox) 

    def NPY_transf_PIL(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_CV2(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_ndimage(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def PIL_transf(self, img, msk=None):
        ow, oh = self.get_size(img.size)
        if msk is not None:
            return self.PIL_Fit(img, (ow,oh), self.PIL_itrp), self.PIL_Fit(msk, (ow,oh), self.PIL_itrp_msk)
        return self.PIL_Fit(img, (ow,oh), self.PIL_itrp) 

    def NPY_transf(self, img, msk=None):
        if self.method == 'PIL':
            return self.NPY_transf_PIL(img, msk)
        elif self.method == 'ndimage':
            return self.NPY_transf_ndimage(img, msk)
        elif self.method == 'CV2':
            return self.NPY_transf_CV2(img, msk)

class Rotate(Transformation):
    """Rotate the given 2D image"""
    
    def __init__(self, ang = None, reshape=False, 
                 img_interpolation='cubic', msk_interpolation='nearest', 
                 method='ndimage', *args, **kwargs):
        '''
        Args:
            ang: angle, degrees, clockwise
            factor: scale factor, float
            interpolation: 'nearest', 'linear', 'cubic', 'antialias', 'area'
        '''
        super(Rotate, self).__init__(*args, **kwargs)

        self.ang = ang
        self.reshape = reshape
        self.img_interp = img_interpolation
        self.msk_interp = msk_interpolation
        
        PIL_dict = {'nearest': Image.NEAREST, 'linear': Image.BILINEAR, 
                     'cubic': Image.BICUBIC}
        self.PIL_itrp = PIL_dict[self.img_interp]
        self.PIL_itrp_msk = PIL_dict[self.msk_interp]
        
        ndimage_dict = {'nearest': 0, 'linear': 1, 
                     'cubic': 2}
        self.ndimage_itrp = ndimage_dict[self.img_interp]
        self.ndimage_itrp_msk = ndimage_dict[self.msk_interp]
        
        CV2_dict = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 
                     'cubic': cv2.INTER_CUBIC}
        self.CV2_itrp = CV2_dict[self.img_interp]
        self.CV2_itrp_msk = CV2_dict[self.msk_interp]
        
        assert method in ['PIL', 'ndimage', 'CV2']
        self.method = method
    
    def get_ang(self):
        return self.ang

    def PIL_Rotate(self, img, ang, interp):
        return img.rotate(-ang, interp, expand=self.reshape) 
    
    def CV2_Rotate(self, img, ang, interp):
        h, w = img.shape
        xc = int(w/2.0)
        yc = int(h/2.0)
        center = (xc, yc)
        rot = cv2.getRotationMatrix2D(center, -ang, 1.0)
        
        if self.reshape:
            abs_cos, abs_sin = abs(rot[0,0]), abs(rot[0,1])
            bound_x = int(h*abs_sin + w*abs_cos)
            bound_y = int(h*abs_cos + w*abs_sin)
            xc = int(bound_x/2.0)
            yc = int(bound_y/2.0)
            center = (xc, yc)
            rot = cv2.getRotationMatrix2D(center, -ang, 1.0)
            rot_move = np.dot(rot, np.array([(bound_x-w)*0.5, (bound_y-h)*0.5, 0]))
            rot[0,2] += rot_move[0]
            rot[1,2] += rot_move[1]
            result = cv2.warpAffine(img, rot, dsize=(bound_x, bound_y), flags=interp)
        else:
            result = cv2.warpAffine(img, rot, dsize=(w,h), flags=interp)
        return result

    def NPY_transf_PIL(self, img, msk=None):
        if self.channels == 'first':
            axis = 0
            w, h = img.shape[1], img.shape[2]
        else:
            axis = 2
            w, h = img.shape[0], img.shape[1] 
        
        ang = self.get_ang()
        
        result_img = []
        if self.channels == 'first':
            for i_ch in range(img.shape[axis]):
                tmp_img = img[i_ch,:,:]
                tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                tmp_rst = self.PIL_Rotate(tmp_im, ang, self.PIL_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
        else:
            for i_ch in range(img.shape[axis]):
                tmp_img = img[:,:,i_ch]
                tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                tmp_rst = self.PIL_Rotate(tmp_im, ang, self.PIL_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
            result_img = np.transpose(result_img, (1,2,0))
                
        if msk is not None:
            result_msk = []
            if self.channels == 'first':
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[i_ch,:,:]
                    tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                    tmp_rst = self.PIL_Rotate(tmp_im, ang, self.PIL_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
            else:
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[:,:,i_ch]
                    tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                    tmp_rst = self.PIL_Rotate(tmp_im, ang, self.PIL_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
                result_msk = np.transpose(result_msk, (1,2,0))
                
            return result_img, result_msk
        
        return result_img

    def NPY_transf_CV2(self, img, msk=None):
        if self.channels == 'first':
            axis = 0
            w, h = img.shape[1], img.shape[2]
        else:
            axis = 2
            w, h = img.shape[0], img.shape[1] 
        
        ang = self.get_ang()
        
        result_img = []
        if self.channels == 'first':
            for i_ch in range(img.shape[axis]):
                tmp_img = img[i_ch,:,:]
                tmp_im = np.transpose(tmp_img.astype(np.float32), (1,0))
                tmp_rst = self.CV2_Rotate(tmp_im, ang, self.CV2_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
        else:
            for i_ch in range(img.shape[axis]):
                tmp_img = img[:,:,i_ch]
                tmp_im = np.transpose(tmp_img.astype(np.float32), (1,0))
                tmp_rst = self.CV2_Rotate(tmp_im, ang, self.CV2_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
            result_img = np.transpose(result_img, (1,2,0))
                
        if msk is not None:
            result_msk = []
            if self.channels == 'first':
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[i_ch,:,:]
                    tmp_im = np.transpose(tmp_img.astype(np.float32), (1,0))
                    tmp_rst = self.CV2_Rotate(tmp_im, ang, self.CV2_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
            else:
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[:,:,i_ch]
                    tmp_im = np.transpose(tmp_img.astype(np.float32), (1,0))
                    tmp_rst = self.CV2_Rotate(tmp_im, ang, self.CV2_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
                result_msk = np.transpose(result_msk, (1,2,0))
                
            return result_img, result_msk
        
        return result_img

    def NPY_transf_ndimage(self, img, msk=None):
        if self.channels == 'first':
            axis = 0
            w, h = img.shape[1], img.shape[2]
        else:
            axis = 2
            w, h = img.shape[0], img.shape[1] 
        
        ang = self.get_ang()
        
        result_img = []
        if self.channels == 'first':
            for i_ch in range(img.shape[axis]):
                tmp_img = img[i_ch,:,:]
                tmp_img = tmp_img.astype(np.float32)
                tmp_rst = ndimage.rotate(tmp_img, ang, reshape=self.reshape, 
                                       order=self.ndimage_itrp, mode='constant')
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img)).astype(img.dtype)
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
        else:
            for i_ch in range(img.shape[axis]):
                tmp_img = img[:,:,i_ch]
                tmp_img = tmp_img.astype(np.float32)
                tmp_rst = ndimage.rotate(tmp_img, ang, reshape=self.reshape, 
                                       order=self.ndimage_itrp, mode='constant')
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img)).astype(img.dtype)
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
            result_img = np.transpose(result_img, (1,2,0))
                
        if msk is not None:
            result_msk = []
            if self.channels == 'first':
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[i_ch,:,:]
                    tmp_img = tmp_img.astype(np.float32)
                    tmp_rst = ndimage.rotate(tmp_img, ang, reshape=self.reshape, 
                                       order=self.ndimage_itrp_msk, mode='constant')
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img)).astype(msk.dtype)
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
            else:
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[:,:,i_ch]
                    tmp_img = tmp_img.astype(np.float32)
                    tmp_rst = ndimage.rotate(tmp_img, ang, reshape=self.reshape, 
                                       order=self.ndimage_itrp_msk, mode='constant')
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img)).astype(msk.dtype)
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
                result_msk = np.transpose(result_msk, (1,2,0))
                
            return result_img, result_msk
        
        return result_img

    def PIL_transf(self, img, msk=None):
        ang = self.get_ang()
        if msk is not None:
            return self.PIL_Rotate(img, ang, self.PIL_itrp), self.PIL_Rotate(msk, ang, self.PIL_itrp_msk)
        return self.PIL_Rotate(img, ang, self.PIL_itrp)       

    def NPY_transf(self, img, msk=None):
        if self.method == 'PIL':
            return self.NPY_transf_PIL(img, msk)
        elif self.method == 'ndimage':
            return self.NPY_transf_ndimage(img, msk)
        elif self.method == 'CV2':
            return self.NPY_transf_CV2(img, msk)
        
        
class RandomRotate(Rotate):
    """Crop the given 2D image randomly"""
    
    def __init__(self, factor_ang, p = 1, *args, **kwargs):
        '''
        Args:
            factor_ang: ang: angle, degrees, +- value
        '''
        
        super(RandomRotate, self).__init__(ang = None, *args, **kwargs)
        
        self.factor_ang = factor_ang
        
        assert p >= 0 and p <= 1
        self.p = p
    
    def get_ang(self):
        return random.uniform(-self.factor_ang, self.factor_ang)


class Move(Transformation):
    """Move the given 2D image"""
    
    def __init__(self, XYmove, 
                 img_interpolation='cubic', msk_interpolation='nearest', 
                 method='ndimage', *args, **kwargs):
        '''
        Args:
            XYmove: (x,y), in pixels, translation.
            interpolation: 'nearest', 'linear', 'cubic'
        '''
        
        super(Move, self).__init__(*args, **kwargs)
        
        self.XYmove = XYmove
        self.img_interp = img_interpolation
        self.msk_interp = msk_interpolation
        
        PIL_dict = {'nearest': Image.NEAREST, 'linear': Image.BILINEAR, 
                     'cubic': Image.BICUBIC}
        self.PIL_itrp = PIL_dict[self.img_interp]
        self.PIL_itrp_msk = PIL_dict[self.msk_interp]
        
        ndimage_dict = {'nearest': 0, 'linear': 1, 
                     'cubic': 2}
        self.ndimage_itrp = ndimage_dict[self.img_interp]
        self.ndimage_itrp_msk = ndimage_dict[self.msk_interp]
        
        CV2_dict = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 
                     'cubic': cv2.INTER_CUBIC}
        self.CV2_itrp = CV2_dict[self.img_interp]
        self.CV2_itrp_msk = CV2_dict[self.msk_interp]
        
        assert method in ['PIL', 'ndimage', 'CV2']
        self.method = method
    
    def get_XYmove(self, img_size):
        return self.XYmove
    
    def PIL_Move(self, img, XYmove, interp):
        data = (1, 0, -XYmove[0], 0, 1, -XYmove[1])
        return img.transform(img.size, Image.AFFINE, data, interp)
    
    def NPY_Move(self, img, XYmove):
        if self.channels == 'first':
            x_s = img.shape[-2]
            y_s = img.shape[-1]
        else:
            x_s = img.shape[0]
            y_s = img.shape[1]
        
        x0, y0, = 0, 0
        xi0, yi0 = max(0, x0+XYmove[0]), max(0, y0+XYmove[1])
        xi1, yi1 = min(x_s, x0+XYmove[0]+x_s), min(y_s, y0+XYmove[1]+y_s)
        xf0, yf0 = max(0, x0-XYmove[0]), max(0, y0-XYmove[1])
        xf1, yf1 = min(x_s, x0-XYmove[0]+x_s), min(y_s, y0-XYmove[1]+y_s)
        
        result = np.zeros_like(img)  
        if self.channels == 'first':
            result[:, xi0:xi1, yi0:yi1] = img[:, xf0:xf1, yf0:yf1]
        else:
            result[xi0:xi1, yi0:yi1, :] = img[xf0:xf1, yf0:yf1, :]
        return result

    def NPY_transf_PIL(self, img, msk=None):
        if self.channels == 'first':
            axis = 0
            w, h = img.shape[1], img.shape[2]
        else:
            axis = 2
            w, h = img.shape[0], img.shape[1] 
        
        XYmove = self.get_XYmove(img.size)
        
        result_img = []
        if self.channels == 'first':
            for i_ch in range(img.shape[axis]):
                tmp_img = img[i_ch,:,:]
                tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                tmp_rst = self.PIL_Move(tmp_im, XYmove, self.PIL_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
        else:
            for i_ch in range(img.shape[axis]):
                tmp_img = img[:,:,i_ch]
                tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                tmp_rst = self.PIL_Move(tmp_im, XYmove, self.PIL_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
            result_img = np.transpose(result_img, (1,2,0))
                
        if msk is not None:
            result_msk = []
            if self.channels == 'first':
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[i_ch,:,:]
                    tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                    tmp_rst = self.PIL_Move(tmp_im, XYmove, self.PIL_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
            else:
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[:,:,i_ch]
                    tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                    tmp_rst = self.PIL_Move(tmp_im, XYmove, self.PIL_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
                result_msk = np.transpose(result_msk, (1,2,0))
                
            return result_img, result_msk
        
        return result_img

    def NPY_transf_CV2(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_ndimage(self, img, msk=None):
        if self.channels == 'first':
            w, h = img.shape[1], img.shape[2]
        else:
            w, h = img.shape[0], img.shape[1] 
        XYmove = self.get_XYmove((w,h))
        if msk is not None:
            return self.NPY_Move(img, XYmove), self.NPY_Move(msk, XYmove)
        return self.NPY_Move(img, XYmove) 
    
    def PIL_transf(self, img, msk=None):
        XYmove = self.get_XYmove(img.size)
        if msk is not None:
            return self.PIL_Move(img, XYmove, self.PIL_itrp), self.PIL_Move(msk, XYmove, self.PIL_itrp_msk)
        return self.PIL_Move(img, XYmove, self.PIL_itrp)  

    def NPY_transf(self, img, msk=None):
        if self.method == 'PIL':
            return self.NPY_transf_PIL(img, msk)
        else:
            return self.NPY_transf_NPY(img, msk)
        

class RandomMove(Move):
    """Crop the given 2D image randomly"""
    
    def __init__(self, XYmax_move=None, radius=None, inner_circle=True, p = 1, *args, **kwargs):
        '''
        Args:
            XYmove: (x,y), in pixels, translation. If given, don't pass radius
            radius: int, radius of circuference
            inner_circle: bool, if take desplazements inside the circle or just the perimeter
        '''
        
        super(RandomMove, self).__init__(XYmove = None, *args, **kwargs)
        
        assert (XYmax_move is None or radius is None) and not (XYmax_move == radius)
        
        self.XYmax_move = XYmax_move
        self.radius = radius
        self.inner_circle = inner_circle
        

        
        assert p >= 0 and p <= 1
        self.p = p
    
    def get_XYmove(self, img_size):
        if self.radius is not None:
            # get desplazament coords
            ang, dist = np.random.sample(2)
            ang = math.radians(360*ang)
            dist = self.radius*dist if self.inner_circle else self.radius
            dx = int(np.round(dist*math.cos(ang)))
            dy = int(np.round(dist*math.sin(ang)))
            XYmove  = (dx, dy)
        else:
            dx, dy = np.random.sample(2) *2 -1
            dx = int(np.round(self.XYmax_move[0]*dx))
            dy = int(np.round(self.XYmax_move[1]*dy))
            XYmove = (dx, dy)
            self.XYmove = XYmove
        return XYmove
    

class Shear(Transformation):
    """Move the given 2D image"""
    
    def __init__(self, XYshear, 
                 img_interpolation='cubic', msk_interpolation='nearest', 
                 method='ndimage', *args, **kwargs):
        '''
        Args:
            XYmshear: (dx,dy), ratio, shear.
            interpolation: 'nearest', 'linear', 'cubic'
        '''
        
        super(Shear, self).__init__(*args, **kwargs)
        
        self.XYshear = XYshear
        self.img_interp = img_interpolation
        self.msk_interp = msk_interpolation
        
        PIL_dict = {'nearest': Image.NEAREST, 'linear': Image.BILINEAR, 
                     'cubic': Image.BICUBIC}
        self.PIL_itrp = PIL_dict[self.img_interp]
        self.PIL_itrp_msk = PIL_dict[self.msk_interp]
        
        ndimage_dict = {'nearest': 0, 'linear': 1, 
                     'cubic': 2}
        self.ndimage_itrp = ndimage_dict[self.img_interp]
        self.ndimage_itrp_msk = ndimage_dict[self.msk_interp]
        
        CV2_dict = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 
                     'cubic': cv2.INTER_CUBIC}
        self.CV2_itrp = CV2_dict[self.img_interp]
        self.CV2_itrp_msk = CV2_dict[self.msk_interp]
        
        assert method in ['PIL', 'ndimage', 'CV2']
        self.method = method
    
    def get_XYshear(self, img_size):
        return self.XYshear
    
    def PIL_Shear(self, img, XYshear, interp):
        PIL_matrix = [1, -XYshear[0], 0,
                      -XYshear[1], 1, 0]
        PIL_img = img
        def transform(x, y, (a, b, c, d, e, f)=PIL_matrix):
            return a*x + b*y + c, d*x + e*y + f

        # calculate output size
        w, h = PIL_img.size
        xx = []
        yy = []
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x, y = transform(x, y)
            xx.append(x)
            yy.append(y)
        w = int(math.ceil(max(xx)) - math.floor(min(xx)))
        h = int(math.ceil(max(yy)) - math.floor(min(yy)))
        
        # adjust center
        x, y = transform(w / 2.0, h / 2.0)
        PIL_matrix[2] = PIL_img.size[0] / 2.0 - x
        PIL_matrix[5] = PIL_img.size[1] / 2.0 - y
        
        PIL_rst = PIL_img.transform((w, h), Image.AFFINE, PIL_matrix, interp)
        # center crop
        x0 = int((PIL_rst.size[0]-PIL_img.size[0])/2.0)
        y0 = int((PIL_rst.size[1]-PIL_img.size[1])/2.0)
        x1 = x0 + PIL_img.size[0]
        y1 = y0 + PIL_img.size[1]
        PIL_rst = PIL_rst.crop((x0, y0, x1, y1))
        return PIL_rst

    def NPY_transf_PIL(self, img, msk=None):
        if self.channels == 'first':
            axis = 0
            w, h = img.shape[1], img.shape[2]
        else:
            axis = 2
            w, h = img.shape[0], img.shape[1] 
        
        XYshear = self.get_XYshear(img.size)
        
        result_img = []
        if self.channels == 'first':
            for i_ch in range(img.shape[axis]):
                tmp_img = img[i_ch,:,:]
                tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                tmp_rst = self.PIL_Shear(tmp_im, XYshear, self.PIL_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
        else:
            for i_ch in range(img.shape[axis]):
                tmp_img = img[:,:,i_ch]
                tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                tmp_rst = self.PIL_Shear(tmp_im, XYshear, self.PIL_itrp)
                tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                result_img.append(tmp_rst)
            result_img = np.array(result_img)
            result_img = np.transpose(result_img, (1,2,0))
                
        if msk is not None:
            result_msk = []
            if self.channels == 'first':
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[i_ch,:,:]
                    tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                    tmp_rst = self.PIL_Shear(tmp_im, XYshear, self.PIL_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
            else:
                for i_ch in range(msk.shape[axis]):
                    tmp_img = msk[:,:,i_ch]
                    tmp_im = Image.fromarray(np.transpose(tmp_img.astype(np.float32), (1,0)))
                    tmp_rst = self.PIL_Shear(tmp_im, XYshear, self.PIL_itrp_msk)
                    tmp_rst = np.transpose(np.array(tmp_rst).astype(tmp_img.dtype), (1,0))
                    tmp_rst = np.clip(tmp_rst, np.min(tmp_img), np.max(tmp_img))
                    result_msk.append(tmp_rst)
                result_msk = np.array(result_msk)
                result_msk = np.transpose(result_msk, (1,2,0))
                
            return result_img, result_msk
        
        return result_img
    
    def NPY_transf_CV2(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_ndimage(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def PIL_transf(self, img, msk=None):
        XYshear = self.get_XYshear(img.size)
        if msk is not None:
            return self.PIL_Shear(img, XYshear, self.PIL_itrp), self.PIL_Shear(msk, XYshear, self.PIL_itrp_msk)
        return self.PIL_Shear(img, XYshear, self.PIL_itrp)  

    def NPY_transf(self, img, msk=None):
        if self.method == 'PIL':
            return self.NPY_transf_PIL(img, msk)
        elif self.method == 'ndimage':
            return self.NPY_transf_ndimage(img, msk)
        elif self.method == 'CV2':
            return self.NPY_transf_CV2(img, msk)
        

class RandomShear(Shear):
    """Crop the given 2D image randomly"""
    
    def __init__(self, XYmax_shear=None, radius=None, inner_circle=True, p = 1, *args, **kwargs):
        '''
        Args:
            XYshear: (dx,dy), ratio, shear. If given, don't pass radius
            radius: int, radius of circuference
            inner_circle: bool, if take desplazements inside the circle or just the perimeter
        '''
        
        super(RandomShear, self).__init__(XYshear = None, *args, **kwargs)
        
        assert (XYmax_shear is None or radius is None) and not (XYmax_shear == radius)
        
        self.XYmax_shear = XYmax_shear
        self.radius = radius
        self.inner_circle = inner_circle
        

        
        assert p >= 0 and p <= 1
        self.p = p
    
    def get_XYshear(self, img_size):
        if self.radius is not None:
            # get desplazament coords
            ang, dist = np.random.sample(2)
            ang = math.radians(360*ang)
            dist = self.radius*dist if self.inner_circle else self.radius
            dx = dist*math.cos(ang) / float(img_size[0])
            dy = dist*math.sin(ang) / float(img_size[1])
            XYshear  = (dx, dy)
        else:
            dx, dy = np.random.sample(2) *2 -1
            dx = self.XYmax_shear[0]*dx
            dy = self.XYmax_shear[1]*dy
            XYshear = (dx, dy)
            self.XYshear = XYshear
        return XYshear

class Bright(Transformation):
    """Bright the given 2D image"""
    
    def __init__(self, factor, 
                 method='ndimage', *args, **kwargs):
        '''
        Args:
            factor: factor, depends on method
            interpolation: 'nearest', 'linear', 'cubic'
        '''
        
        super(Bright, self).__init__(*args, **kwargs)
        
        self.factor = factor
        
        assert method in ['PIL', 'ndimage', 'CV2']
        self.method = method
    
    def get_factor(self, img_size):
        return self.factor
    
    def PIL_Bright(self, img, factor):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def NPY_transf_PIL(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_CV2(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_ndimage(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def PIL_transf(self, img, msk=None):
        factor = self.get_factor(img.size)
        if msk is not None:
            return self.PIL_Bright(img, factor), msk
        return self.PIL_Bright(img, factor)

    def NPY_transf(self, img, msk=None):
        if self.method == 'PIL':
            return self.NPY_transf_PIL(img, msk)
        elif self.method == 'ndimage':
            return self.NPY_transf_ndimage(img, msk)
        elif self.method == 'CV2':
            return self.NPY_transf_CV2(img, msk)

class RandomBright(Bright):
    """Bright the given 2D image randomly"""
    
    def __init__(self, factor_range=None, p = 1, *args, **kwargs):
        '''
        Args:
            factor_range: (min,max). 
        '''
        
        super(RandomBright, self).__init__(factor = None, *args, **kwargs)
        
        self.factor_range = factor_range
        
        assert p >= 0 and p <= 1
        self.p = p
    
    def get_factor(self, img_size):
        fmin = self.factor_range[0]
        fmax = self.factor_range[1]
        f = random.uniform(fmin, fmax)
        self.factor = f
        return f

class Contrast(Transformation):
    """Contrast the given 2D image"""
    
    def __init__(self, factor, 
                 method='ndimage', *args, **kwargs):
        '''
        Args:
            factor: factor, depends on method
            interpolation: 'nearest', 'linear', 'cubic'
        '''
        
        super(Contrast, self).__init__(*args, **kwargs)
        
        self.factor = factor
        
        assert method in ['PIL', 'ndimage', 'CV2']
        self.method = method
    
    def get_factor(self, img_size):
        return self.factor
    
    def PIL_Contrast(self, img, factor):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def NPY_transf_PIL(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_CV2(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_ndimage(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def PIL_transf(self, img, msk=None):
        factor = self.get_factor(img.size)
        if msk is not None:
            return self.PIL_Contrast(img, factor), msk
        return self.PIL_Contrast(img, factor)

    def NPY_transf(self, img, msk=None):
        if self.method == 'PIL':
            return self.NPY_transf_PIL(img, msk)
        elif self.method == 'ndimage':
            return self.NPY_transf_ndimage(img, msk)
        elif self.method == 'CV2':
            return self.NPY_transf_CV2(img, msk)

class RandomContrast(Contrast):
    """Bright the given 2D image randomly"""
    
    def __init__(self, factor_range=None, p = 1, *args, **kwargs):
        '''
        Args:
            factor_range: (min,max). 
        '''
        
        super(RandomContrast, self).__init__(factor = None, *args, **kwargs)
        
        self.factor_range = factor_range
        
        assert p >= 0 and p <= 1
        self.p = p
    
    def get_factor(self, img_size):
        fmin = self.factor_range[0]
        fmax = self.factor_range[1]
        f = random.uniform(fmin, fmax)
        self.factor = f
        return f

class Color(Transformation):
    """Color the given 2D image"""
    
    def __init__(self, factor, 
                 method='ndimage', *args, **kwargs):
        '''
        Args:
            factor: factor, depends on method
            interpolation: 'nearest', 'linear', 'cubic'
        '''
        
        super(Color, self).__init__(*args, **kwargs)
        
        self.factor = factor
        
        assert method in ['PIL', 'ndimage', 'CV2']
        self.method = method
    
    def get_factor(self, img_size):
        return self.factor
    
    def PIL_Color(self, img, factor):
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    
    def NPY_transf_PIL(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_CV2(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def NPY_transf_ndimage(self, img, msk=None):
        raise ValueError('Method not implemented')
    
    def PIL_transf(self, img, msk=None):
        factor = self.get_factor(img.size)
        if msk is not None:
            return self.PIL_Color(img, factor), msk
        return self.PIL_Color(img, factor)

    def NPY_transf(self, img, msk=None):
        if self.method == 'PIL':
            return self.NPY_transf_PIL(img, msk)
        elif self.method == 'ndimage':
            return self.NPY_transf_ndimage(img, msk)
        elif self.method == 'CV2':
            return self.NPY_transf_CV2(img, msk)

class RandomColor(Color):
    """Bright the given 2D image randomly"""
    
    def __init__(self, factor_range=None, p = 1, *args, **kwargs):
        '''
        Args:
            factor_range: (min,max). 
        '''
        
        super(RandomColor, self).__init__(factor = None, *args, **kwargs)
        
        self.factor_range = factor_range
        
        assert p >= 0 and p <= 1
        self.p = p
    
    def get_factor(self, img_size):
        fmin = self.factor_range[0]
        fmax = self.factor_range[1]
        f = random.uniform(fmin, fmax)
        self.factor = f
        return f    
    
class RandomShuffleChannels(Transformation):
    """Shuffle the image channels randomly"""
    
    def __init__(self, p = 1.0, *args, **kwargs):
        '''
        Args:
            p: probability
            channels: None, 'first', 'last'
        '''
        super(RandomShuffleChannels, self).__init__(*args, **kwargs)
        
        assert p >= 0 and p <= 1
        self.p = p
        
    def PIL_transf(self, img, msk=None):
        tmp = list(img.split())
        random.shuffle(tmp)
        random.shuffle(tmp)
        tmp_img = Image.merge("RGB", tmp)
        if msk is not None:
            return tmp_img, msk
        return tmp_img
    
    def NPY_transf(self, img, msk=None):
        if self.channels == 'first':
            ch = range(0,img.shape[0])
            random.shuffle(ch)
            random.shuffle(ch)
            tmp_img = img[ch, ...]
        else:
            ch = range(0,img.shape[-1])
            random.shuffle(ch)
            random.shuffle(ch)
            tmp_img = img[..., ch]
        if msk is not None:
            return tmp_img, msk
        return tmp_img

    
class ToArray(Transformation):
    """Convert to numpy array"""
    
    def __init__(self, img_dtype = np.float32, msk_dtype = np.float32, pos_channels='first', 
                 img_DT_mean=None, img_DT_std=None, *args, **kwargs):
        '''
        Args:
            dtype: numpy array dtype
            channels: None, 'first', 'last'
        '''
        super(ToArray, self).__init__(*args, **kwargs)
        
        self.img_dtype = img_dtype
        self.msk_dtype = msk_dtype
        self.img_DT_mean = img_DT_mean
        self.img_DT_std = img_DT_std
        assert pos_channels in ['first', 'last']
        self.pos_channels = pos_channels

        
    def PIL_transf(self, img, msk=None):
        result_img = np.array(img)
        if len(result_img.shape) == 2:
            result_img = result_img[..., np.newaxis]
        if self.pos_channels == 'first':
            result_img = np.transpose(result_img, (2,1,0))
        else:
            result_img = np.transpose(result_img, (1,0,2))
    
        result_img = result_img.astype(self.img_dtype)
        #standarize
        if (self.img_DT_mean is not None) and (self.img_DT_std is not None):
            result_img = standarize_image(result_img, self.img_DT_mean, self.img_DT_std, on_site=True)
            
        if msk is not None:
            result_msk = np.array(msk)
            if len(result_msk.shape) == 2:
                result_msk = result_msk[..., np.newaxis]
            if self.pos_channels == 'first':
                result_msk = np.transpose(result_msk, (2,1,0))
            else:
                result_msk = np.transpose(result_msk, (1,0,2))
            return result_img, result_msk.astype(self.msk_dtype)    
        
        return result_img
    
    def NPY_transf(self, img, msk=None):
        result_img = img.astype(self.img_dtype)
        #standarize
        if (self.img_DT_mean is not None) and (self.img_DT_std is not None):
            img = standarize_image(img, self.img_DT_mean, self.img_DT_std, on_site=True)   
        if msk is not None:
            return result_img, msk.astype(self.msk_dtype)
        return result_img