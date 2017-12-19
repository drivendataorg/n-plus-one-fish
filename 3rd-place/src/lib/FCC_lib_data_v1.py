
# For number crunching
import numpy as np
import pandas as pd

# For dealing images
import imageio

from PIL import Image
import matplotlib.pyplot as plt

# Misc.
import math
import os
import json
import copy

#################################
# LOGGER

class Logger(object):
    """ Logger class.
    """
    def __init__(self, filename, orig_stdout):
        self.filename = filename
        self.terminal = orig_stdout
        self.logfile = open(self.filename, "w")
        self.logfile.close()
        self.log = False
        self.verbose = True
        self.vnext = False
        self.linebuf = ''
        self.buff = []
        
    def openlog(self):
        if not self.log:
            self.log = True

    def write(self, message):
        if self.verbose:
            self.terminal.write(message)
        if (self.vnext & (not self.verbose)):
            self.terminal.write(message)
            self.vnext = False
        if self.log:
            self.linebuf = ''

            for c in message:
                if c == '\r':
                    self.linebuf = ''
                    readFile = open(self.filename)
                    lines = readFile.readlines()
                    readFile.close()
                    w = open(self.filename,'w')
                    w.writelines([item for item in lines[:-1]])
                    w.close()
                else:
                    self.linebuf += c

            self.logfile = open(self.filename, "a")
            self.logfile.write(self.linebuf)
            self.logfile.close()  
    
    def closelog(self):
        self.log = False

    def flush(self):
        pass


#################################
# VIDEO CLASS

class vidDATA(object):
    """ Video data class for N+1 fish, N+2 fish competition """
    def __init__(self, video_path, annot_path=None, read_data=True):
        
        # Store video & parameters
        self.video_path = video_path
        self.annot_path = annot_path
        self.vi = None  #imageo object   
        self.raw_annot = None  # dict
        self.annot = None
        
        # ROI and patches
        self.roi = None
        self.patches = None
        
        # Read data
        if read_data:
            self.load_vi(retur=False)
            if annot_path is not None:
                self.load_annot(retur=False)
    
    def load_vi(self, retur=True):
        """ Read video files using ffmpeg pluging as imageo object """
        vi = imageio.get_reader(self.video_path,  'ffmpeg')
        self.vi = vi
        if retur:
            return vi

    def load_annot(self, retur=True):
        """ Load annotation from json files """
        with open(self.annot_path) as data_file:
            raw_annot = json.load(data_file)
        self.raw_annot = raw_annot
        
        # Process annotations
        annot = dict()
        annot['frames'] = [int(s['frame']) for s in raw_annot['detections']]
        annot['ids'] = [int(s['id']) for s in raw_annot['detections']]
        annot['species'] = [str(s['subspecies']).lower() for s in raw_annot['tracks']]
        annot['bbox'] = [(min([int(s['x']), int(s['w'])]),
                        min([int(s['y']), int(s['h'])]),
                        max([int(s['x']), int(s['w'])]),
                        max([int(s['y']), int(s['h'])])) for s in raw_annot['detections']]
        self.annot= annot

        if retur:
            return annot

    def load_frames(self, nb_frames=9):
        """ Extract a given number of frames from video (imageo object).
            For visualization purposes.
        """
        # get images
        total_frames = len(self.vi)
        gap = int(total_frames/float(nb_frames+1))
        imgs = []
        for i in range(nb_frames):
            imgs.append(self.vi.get_data(gap*(i+1)))
        return imgs
    
    def show_frames(self, nb_frames=9):
        """ Show a given number of frames from video (imageo object) """
        imgs = self.load_frames(nb_frames)
        
        # plot images
        nbx = int(math.sqrt(nb_frames))
        nby = int(np.ceil(nb_frames/float(nbx)))
        fig,axes = plt.subplots(nbx,nby,figsize=[16,16])
        ax = axes.ravel()
    
        for i, img in enumerate(imgs):
            ax[i].imshow(img)
        plt.show()

    def load_frames_annot(self):
        """ Extract frames with annotations from video (imageo object) """
        if self.annot is None:
            self.load_annot(retur=False)
        imgs = []
        for i, frame in enumerate(self.annot['frames']):
            imgs.append(self.vi.get_data(frame))
            
        return imgs
    
    def show_frames_annot(self, title='ANNOTATIONS', grid = None, size = (16,16)):
        """ Show frames with annotations from video (imageo object) """
        import matplotlib.patches as patches
        imgs = self.load_frames_annot()
        nb_frames = len(imgs)
        
        # plot images
        if grid is None:
            nbx = int(math.sqrt(nb_frames))
            nby = int(np.ceil(nb_frames/float(nbx)))
        else:
            nbx, nby = grid
        fig,axes = plt.subplots(nbx,nby,figsize=size)
        fig.suptitle(title)
        ax = axes.ravel()
    
        for i, img in enumerate(imgs):
            ax[i].imshow(img)
            ititle = '{} - {}'.format(self.annot['frames'][i], self.annot['species'][i])
            ax[i].set_title(ititle) 
            coords = self.annot['bbox'][i]
            ax[i].add_patch(patches.Arrow(coords[0], coords[1], 
                coords[2]-coords[0], coords[3]-coords[1], fill=False))
        plt.show()


#################################
# DATA CLASS

class FishDATA(object):
    """ Data class for N+1 fish, N+2 fish competition """
    def __init__(self):
        
        # LOAD GLOBAL PATHS & SETTINGS
        path_settings_file = "SETTINGS_path.json"
        with open(path_settings_file) as data_file:
            PATH_SETTINGS = json.load(data_file)
        path_settings_file = "SETTINGS_exec.json"
        with open(path_settings_file) as data_file:
            EXEC_SETTINGS = json.load(data_file)
        
        # Paths
        self.path_settings = PATH_SETTINGS
        self.exec_settings = EXEC_SETTINGS
        self.cache = EXEC_SETTINGS['cache'] == "True"
        self.verbose = EXEC_SETTINGS['verbose'] == "True" 
        self.raw_data = str(PATH_SETTINGS['path_data_raw'])
        self.pp_data = str(PATH_SETTINGS['path_L0_preprocess'])
        self.paths = {
            # Source paths
            'train'        : os.path.join(self.raw_data, 'train_videos', '{video_id}.mp4'),
            'test'         : os.path.join(self.raw_data, 'test_videos', '{video_id}.mp4'),
            'train_annot'  : os.path.join(self.raw_data, 'train_videos', '{video_id}.json'),
            'annotations'  : os.path.join(self.raw_data, 'training.csv'),
            'sample'       : os.path.join(self.raw_data, 'submission_format_zeros.csv'),
            
            } 
        
        # Classes
        self.clss_nb = 7
        self.clss_names = [
            'species_fourspot',   # 0
            'species_grey sole',  # 1
            'species_other',      # 2
            'species_plaice',     # 3
            'species_summer',     # 4
            'species_windowpane', # 5
            'species_winter']     # 6
        self.clss_none = ['species_none',]
        
        # Annotations
        self.annotations = pd.read_csv(self.paths['annotations'])
        self.annotations = self.annotations.assign(species_none = \
                                                   (np.isnan(self.annotations.fish_number)).astype(np.uint16))
        self.S1_target = None
        self.S1_target_v2 = None
    
    def path(self, name, **kwargs):
        """ Return path to various source files """
        path = self.paths[name].format(**kwargs)
        return path

    def load_vidDATA(self, itype, video_id, read_annotations=False, read_data=True):
        """ Return vidDATA object given video_id and itype [train/test] """
        video_path = self.path(itype, video_id=video_id)
        annot_path = self.path('train_annot', video_id=video_id) if read_annotations else None
        return vidDATA(video_path, annot_path, read_data=read_data)

    def get_S1_target_v2(self):
        """ Function to extract S1 target (stage 1, ROI).
            The target is obtained as addition of frames annotation for each video.
            Return a line: center coordinates, angle, length, first and end points coordinates.
        """
        if self.S1_target_v2 is not None:
            return self.S1_target_v2

        df = copy.copy(self.annotations)
        df = df.assign(spec = np.sum(df[[u'species_fourspot', u'species_grey sole',
           u'species_other', u'species_plaice', u'species_summer',
           u'species_windowpane', u'species_winter']].values, axis=1))
    
        # remove no fish frame
        df = df.dropna(how="any", inplace=False)
        
        # remove no length fish
        df = df[df.length > 0]
        
        # Draw lines & get target's center & angle
        from PIL import Image, ImageDraw
        from skimage.measure import regionprops
        import math
        video_ids = np.unique(df.video_id)
        ndf = []
        for video_id in video_ids:
            mini_df = df[df.video_id == video_id]
            im = Image.new('1', (1280, 720))
            draw = ImageDraw.Draw(im)
            for i_row in mini_df.itertuples():
                draw.line((i_row.x1, i_row.y1, i_row.x2, i_row.y2), fill=255, width=10)
            msk = np.transpose(np.array(im), (1,0)).astype(int)
            region = regionprops(msk)[0]
            bbox = region.bbox
            center = int((bbox[0]+bbox[2])/2.0), int((bbox[1]+bbox[3])/2.0)
            xc = center[0]
            yc = center[1]
            ang = np.round(region.orientation * 180 / math.pi).astype(int)+90
            dist = np.round(region.major_axis_length).astype(int)
            x1 = xc + (int(dist/2.0 * math.cos(ang/180.0*math.pi)))
            x2 = xc - (int(dist/2.0 * math.cos(ang/180.0*math.pi)))
            y1 = yc + (int(dist/2.0 * math.sin(ang/180.0*math.pi)))
            y2 = yc - (int(dist/2.0 * math.sin(ang/180.0*math.pi)))
            max_frame = mini_df.frame.values[np.argmax(mini_df.length.values)]
            
            row = [video_id, xc, yc, ang, dist, x1, x2, y1, y2, max_frame]
            ndf.append(row)
        ndf = pd.DataFrame(ndf, columns = ['video_id', 'xc', 'yc', 'ang', 'dist', 'x1', 'x2', 'y1', 'y2', 
                                           'max_frame'])

        self.S1_target_v2 = ndf
        return self.S1_target_v2
    
    def extract_patch(self, img, center, ang, size=(448, 224), convert_BnW=False, returnPIL=False):
        """ Extract patch from image, given the center coordinates, angle and final patch size.
        """
        from PIL import ImageChops
        im = Image.fromarray(img)
        
        # center image
        im = ImageChops.offset(im, -int(center[0]-im.size[0]/2.0), -int(center[1]-im.size[1]/2.0))
        # rotate image
        im = im.rotate(ang, expand=0)
        # crop image
        bbox = (int((im.size[0]-size[0])/2.0), int((im.size[1]-size[1])/2.0), 0, 0)
        bbox = (bbox[0], bbox[1], bbox[0]+size[0], bbox[1]+size[1])
        im = im.crop(bbox)
        
        if convert_BnW:
            # Convert to Black & White
            im = im.convert('L')
            if returnPIL:
                return im
            img = np.array(im)
            img = img[..., np.newaxis]
            return np.transpose(img, (2,1,0))
        
        if returnPIL:
            return im
        return np.transpose(np.array(im), (2,1,0))

    def extract_patch_PIL(self, im, center, ang, size=(448, 224), convert_BnW=False):
        """ Extract patch from image, given the center coordinates, angle and final patch size.
            Return PIL Image object.
        """
        from PIL import ImageChops
        
        # center image
        im = ImageChops.offset(im, -int(center[0]-im.size[0]/2.0), -int(center[1]-im.size[1]/2.0))
        # rotate image
        im = im.rotate(ang, expand=0)
        # crop image
        bbox = (int((im.size[0]-size[0])/2.0), int((im.size[1]-size[1])/2.0), 0, 0)
        bbox = (bbox[0], bbox[1], bbox[0]+size[0], bbox[1]+size[1])
        im = im.crop(bbox)
        
        if convert_BnW:
            # Convert to Black & White
            im = im.convert('L')

        return im


#################################
# DATASET

def read_dsetID(use_cache=False, force_cache=False, save_cache=False):
    '''Search images in folders and return table with IDs and other features'''
    
    import re
    
    # Start data class & variables
    Data = FishDATA()
    df = None
    
    # Try cache
    if not os.path.exists(Data.pp_data):
        os.makedirs(Data.pp_data)
    cache_filename = os.path.join(Data.pp_data, 'dsetID.csv.gz')
    if use_cache | force_cache:
        try:
            df = pd.read_csv(cache_filename)
            return df
        except:
            if force_cache:
                raise Exception ("File dseID not in cache")
    
    # Read images/videos
    itypes = ['train','test']
    for itype in itypes:
        video_path = Data.path(itype, video_id = '{video_id}')
        video_ext = os.path.splitext(video_path)[1]
        video_path = re.sub('{video_id}', '', os.path.splitext(video_path)[0])
        try:
            videos_id = [os.path.splitext(s)[0] for s in os.listdir(video_path) if os.path.splitext(s)[1]==video_ext]
        except:
            continue
        tmp_df = pd.DataFrame({'video_id':videos_id, 'itype':itype}, 
                                columns = ['video_id','itype'])
        df = tmp_df if df is None else pd.concat([df, tmp_df])
        
        
    # Add train folds: F2
    filename = os.path.join(Data.pp_data, 'Kfolds_F2.csv.gz')
    try:
        fk = pd.read_csv(filename)
        df = pd.merge(df, fk, how='outer', on='video_id')
        df.loc[df.index[df.itype=='test'], 'F2'] = 'ALL'
    except:
        None

    # Add train folds: Fs3
    filename = os.path.join(Data.pp_data, 'Kfolds_Fs3.csv.gz')
    try:
        fk = pd.read_csv(filename)
        df = pd.merge(df, fk, how='outer', on='video_id')
        df.loc[df.index[df.itype=='test'], 'Fs3'] = 'ALL'
    except:
        None
    
    # Sort 
    df = df.sort_values(by=['itype','video_id'])
    df = df.reset_index(drop=True)
    
    # Exclude Training
    bad_train_ids = [ ]
    df = df.assign(exclude = np.array([s in bad_train_ids for s in df.video_id.values.tolist()]))

    # Save to cache
    if save_cache:
        df.to_csv(cache_filename, index=False, compression='gzip')
    
    return df   


#################################
# EVALUATION

def dice_coef(pred, true, thr=0.5):
    bpred = pred >= thr
    return 2*np.sum(((bpred+true) == 2)) / (np.sum(bpred) + np.sum(true))


import editdistance
from sklearn.metrics import roc_auc_score, r2_score
COLUMNS = ['frame',
           'video_id',
           'fish_number',
           'length',
           'species_fourspot',
           'species_grey sole',
           'species_other',
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']
SPECIES = COLUMNS[4:]
SPECIES_COL_IDX = np.arange(4, len(COLUMNS))
VIDEO_ID_IDX = 1
FISH_NUMBER_IDX = 2
LENGTH_IDX = 3

def fish_metric_mod(actual_all_vid, predicted_all_vid, a_l=0.1, a_n=0.6, a_s=0.3, species_prefix='species_',
                    return_results=False):
    # From https://github.com/drivendataorg/metrics/blob/master/metrics.py#L129-L227, modified
    """ Reference implementation for the N+1 fish, N+2 fish competition
        evaluation metric.
        Implemented in pure numpy for performance gains over pandas.
    """

    def get_fish_order(fish_numbers, species_probs):
        """ Gets a sequence of fish from the ordering of fish numbers
            and the species probabilities
        """
        sequence = []

        unique_fish = np.unique(fish_numbers[~np.isnan(fish_numbers)])

        for fishy in unique_fish:
            mask = (fish_numbers == fishy)
            this_fish = species_probs[mask, :]

            col_maxes = np.nanmax(this_fish, axis=0)

            species = SPECIES[np.argmax(col_maxes)]
            sequence.append(species)

        return sequence

    def levenfish(act_fish_numbers, act_species, pred_fish_numbers, pred_species):
        """ Edit distance for a sequence of fishes in the competition
            submission format.
        """
        return editdistance.eval(get_fish_order(act_fish_numbers, act_species),
                                 get_fish_order(pred_fish_numbers, pred_species))

    video_ids = actual_all_vid[:, VIDEO_ID_IDX].ravel()

    actual_fish_numbers = actual_all_vid[:, FISH_NUMBER_IDX].astype(np.float64)
    pred_fish_numbers = predicted_all_vid[:, FISH_NUMBER_IDX].astype(np.float64)

    actual_lengths = actual_all_vid[:, LENGTH_IDX].astype(np.float64)
    pred_lengths = predicted_all_vid[:, LENGTH_IDX].astype(np.float64)

    actual_species = actual_all_vid[:, SPECIES_COL_IDX].astype(np.float64)
    pred_species = predicted_all_vid[:, SPECIES_COL_IDX].astype(np.float64)

    uniq_video_ids = np.unique(video_ids)
    per_video_scores = np.zeros_like(uniq_video_ids, dtype=np.float64)
    
    rst_video_id = []
    rst_edit = []
    rst_species = []
    rst_length = []
    rst_score = []

    for ix, vid_idx in enumerate(uniq_video_ids):
        this_vid_mask = (video_ids == vid_idx)

        # edit distance scoring
        n_fish = np.nanmax(actual_fish_numbers[this_vid_mask])

        actual_fn = actual_fish_numbers[this_vid_mask]
        pred_fn = pred_fish_numbers[this_vid_mask]

        actual_spec = actual_species[this_vid_mask, :]
        pred_spec = pred_species[this_vid_mask, :]

        edit_error = 1 - (levenfish(actual_fn, actual_spec, pred_fn, pred_spec) / n_fish)
        edit_error = np.clip(edit_error, 0, 1)
        edit_component = a_n * edit_error

        # only test length and species against frames where we
        # have actual fish labeled
        annotated_frames = ~np.isnan(actual_fn)

        # species identification scoring
        def _auc(a, p):
            try:
                return roc_auc_score(a, p)
            except ValueError:
                mae = np.mean(np.abs(a - p))
                return ((1 - mae) / 2) + 0.5

        aucs = [_auc(actual_spec[annotated_frames, c],
                     pred_spec[annotated_frames, c])
                for c in range(actual_species.shape[1])]

        # normalize to 0-1
        species_auc = 2 * (np.mean(aucs) - 0.5)
        species_auc = np.clip(species_auc, 0, 1)
        species_component = a_s * species_auc

        # we have "no-fish" annotations where all of the species are zero
        # these are only relevant for the species classification task. We'll
        # ignore these for the length task.
        only_fish_annotations = (np.nan_to_num(actual_species.sum(axis=1)) > 0) & this_vid_mask

        # length scoring
        length_r2 = r2_score(actual_lengths[only_fish_annotations],
                             pred_lengths[only_fish_annotations])

        length_r2 = np.clip(length_r2, 0, 1)
        length_component = a_l * length_r2

        per_video_scores[ix] = length_component + edit_component + species_component
        
        rst_video_id.append(vid_idx)
        rst_edit.append(edit_component)
        rst_species.append(species_component)
        rst_length.append(length_component)
        rst_score.append(length_component + edit_component + species_component)

    if return_results:
        rst = pd.DataFrame({'video_id':rst_video_id,
                            'edit_component':rst_edit,
                            'species_component':rst_species,
                            'length_component':rst_length,
                            'score':rst_score
                            })
        return rst
    return np.mean(per_video_scores)

#################################
# FEATURE EXTRACTION

def mov_avg(x, Nprev, Nnxt):
    '''
    Custom Moving Average
    '''
    N = Nprev + 1 + Nnxt
    xp = np.concatenate([np.repeat(x[0], Nprev), x, np.repeat(x[-1], Nnxt)])
    xc = np.convolve(xp, np.ones((N,))/N, mode='valid')
    return xc


def mov_func(x, Nprev, Nnxt, func):
    '''
    Custom Moving Function
    '''
    N = Nprev + 1 + Nnxt
    xp = np.concatenate([np.repeat(x[0], Nprev), x, np.repeat(x[-1], Nnxt)])
    xc = np.zeros_like(x)
    for i in range(xc.shape[0]):
        xc[i] = func(xp[i:(i+N)])
    return xc