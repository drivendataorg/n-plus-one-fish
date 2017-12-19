# Set working directory
import os
directory = "./"
os.chdir(directory) 

theano_flags = 'THEANO_FLAGS=device=cuda0'


##############################################################
source = './src/S9_PO_code_Predictions_Optimization'
predict_args = '--max_cores 16 '
os.system('{} python {}/S9_PO_AxA01_submission.py {}'.format(theano_flags, source, predict_args))  # 10 min


