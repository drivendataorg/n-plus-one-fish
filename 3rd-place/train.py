# Set working directory
import os
directory = "./"
os.chdir(directory) 

theano_flags = 'THEANO_FLAGS=device=cuda'



##############################################################
### S0_PP - Pre Process Files                                # 6 hours
##############################################################
source = './src/S0_PreProcess'
os.system('python {}/S0_ppS1B.py'.format(source))  # 6 hours



##############################################################
### S1_ROI - Video_ROI_Detection                             # 10 hours
##############################################################
source = './src/S1_ROI_code_Video_ROI_Detection'

train_args = ''
#train_args += '--debug_mode '  # Uncomment to test training code reducing training samples and batches
os.system('{} python {}/S1_ROI_NN_AxC01_train.py {}'.format(theano_flags, source, train_args))  # 9 hours

predict_args = ' '
os.system('{} python {}/S1_ROI_NN_AxC01_predict.py {}'.format(theano_flags, source, predict_args))  # 1 hour



##############################################################
### S2_VSE - Video Secuence Extraction                       # 36.5 hours
##############################################################
source = './src/S2_VSE_code_Video_Sequence_Extraction'

train_args = ''
train_args +=  '-f COMPLETE '  # COMPLETE to train each fold and the whole dataset
train_args += '--max_cores 16 '
#train_args += '--debug_mode'  # Uncomment to test training code reducing training samples and batches
os.system('{} python {}/S2_VSE_NN_AxB01_train.py {}'.format(theano_flags, source, train_args))  # 1.5 hours
os.system('{} python {}/S2_VSE_NN_BxB01_train.py {}'.format(theano_flags, source, train_args))  # 5 hours
os.system('{} python {}/S2_VSE_NN_DxB01_train.py {}'.format(theano_flags, source, train_args))  # 7 hours
os.system('{} python {}/S2_VSE_NN_FxB01_train.py {}'.format(theano_flags, source, train_args))  # 7 hours

predict_args = '--max_cores 6 '  # set to 1 if issues when reading using ffmpeg plugin and multithreading
os.system('{} python {}/S2_VSE_NN_AxB01_predict.py {}'.format(theano_flags, source, predict_args))  # 4 hours
os.system('{} python {}/S2_VSE_NN_BxB01_predict.py {}'.format(theano_flags, source, predict_args))  # 4 hours
os.system('{} python {}/S2_VSE_NN_DxB01_predict.py {}'.format(theano_flags, source, predict_args))  # 4 hours
os.system('{} python {}/S2_VSE_NN_FxB01_predict.py {}'.format(theano_flags, source, predict_args))  # 4 hours



##############################################################
### S7_FL - Fish Length  ***** SEGMENTATION                  # 26.5 hours
##############################################################
source = './src/S7_FL_code_Fish_Length'

train_args = ''
train_args +=  '-f COMPLETE '  # COMPLETE to train each fold and the whole dataset
train_args += '--max_cores 16 '
#train_args += '--debug_mode'  # Uncomment to test training code reducing training samples and batches
os.system('{} python {}/S7_FL_NN_AxA10_train.py {}'.format(theano_flags, source, train_args))   # 3.5 hours

predict_args = ''
os.system('{} python {}/S7_FL_NN_AxA10_predict.py {}'.format(theano_flags, source, predict_args))   # 23 hours



##############################################################
### S5_FC - Fish Classification                              # 26 hours
##############################################################
source = './src/S5_FC_code_Fish_Classification'
train_args = ''
train_args +=  '-f COMPLETE '  # COMPLETE to train each fold and the whole dataset
train_args += '--max_cores 16 '
#train_args += '--debug_mode'  # Uncomment to test training code reducing training samples and batches
os.system('{} python {}/S5_FC_NN_QxG02_train.py {}'.format(theano_flags,source, train_args))   # 2.5 hours
os.system('{} python {}/S5_FC_NN_QxJ04_train.py {}'.format(theano_flags, source, train_args))   # 3 hours

predict_args = ''
#predict_args += '--debug_mode'  # Uncomment to test predicting code reducing number of files to make predictions
os.system('{} python {}/S5_FC_NN_QxG02_predict.py {}'.format(theano_flags, source, predict_args))   # 11.5 hours
os.system('{} python {}/S5_FC_NN_QxJ04_predict.py {}'.format(theano_flags, source, predict_args))   # 9 hours



##############################################################
### S7_FL - Fish Length                                      # 29 hours  
##############################################################
source = './src/S7_FL_code_Fish_Length'
train_args = ''
train_args +=  '-f COMPLETE '  # COMPLETE to train each fold and the whole dataset
train_args += '--max_cores 16 '
#train_args += '--debug_mode'  # Uncomment to test training code reducing training samples and batches
os.system('{} python {}/S7_FL_NN_BxA01_train.py {}'.format(theano_flags, source, train_args))   #  5.5 hours
os.system('{} python {}/S7_FL_NN_BxA02_train.py {}'.format(theano_flags, source, train_args))   #  5.5 hours

predict_args = ''
#predict_args += '--debug_mode'  # Uncomment to test predicting code reducing number of files to make predictions
os.system('{} python {}/S7_FL_NN_BxA01_predict.py {}'.format(theano_flags, source, predict_args))   # 9 hours
os.system('{} python {}/S7_FL_NN_BxA02_predict.py {}'.format(theano_flags, source, predict_args))   # 9 hours



##############################################################
### S3_VSEL2 - Video Sequence Extraction LAYER 2
##############################################################
source = './src/S3_VSEL2_code_Video_Sequence_Extraction_Layer2'
train_args = ''
train_args += '--max_cores 16 '
#train_args += '--debug_mode'  # Uncomment to test training code reducing training samples and rounds
os.system('{} python {}/S3_VSEL2_SKxET_CxA01_train.py {}'.format(theano_flags, source, train_args))   #  9 min
os.system('{} python {}/S3_VSEL2_SKxET_DxA01_train.py {}'.format(theano_flags, source, train_args))   #  25 min



##############################################################
### S6_FCL2 - Fish Classification LAYER 2
##############################################################
source = './src/S6_FCL2_code_Fish_Classification_Layer2'
train_args = ''
train_args += '--max_cores 16 '
#train_args += '--debug_mode'  # Uncomment to test training code reducing training samples and rounds
os.system('{} python {}/S6_FCL2_SKxET_BxA20_train.py {}'.format(theano_flags, source, train_args))   #  3.5 min



##############################################################
### S8_FLL2 - Fish Length LAYER 2
##############################################################
source = './src/S8_FLL2_code_Fish_Length_Layer2'
train_args = ''
train_args += '--max_cores 16 '
#train_args += '--debug_mode'  # Uncomment to test training code reducing training samples and rounds
os.system('{} python {}/S8_FLL2_SKxET_AxA02_train.py {}'.format(theano_flags, source, train_args))   #  1.5 min

