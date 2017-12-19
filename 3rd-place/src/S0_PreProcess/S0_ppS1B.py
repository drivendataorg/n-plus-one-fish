
# For number crunching
import numpy as np
import pandas as pd

# Misc
import sys
from tqdm import tqdm

# Project Library
from src.lib import FCC_lib_data_v1 as ld
from src.lib import FCC_lib_preprocess_v1 as pp

# READ & PREPARE DATA 
print('-'*40)
print('READING DATA')
Data = ld.FishDATA()
dsetID = ld.read_dsetID()

# PREPROCESSING FILES
print('-'*40)
print('PREPROCESSING FILES: ppS1B')
itera = tqdm(enumerate(dsetID.itertuples()), total=len(dsetID), unit='vid', file=sys.stdout)
for i, row in itera:
    itype = row.itype
    video_id = row.video_id
    
    # Read image.
    img_raw = pp.ppS1B(itype, video_id=video_id, Data=Data, 
                       use_cache=True, cache_only_training=False, verbose=False)

print('FINISHED')
print('-'*40)