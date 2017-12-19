import numpy as np
import pandas as pd
import sys
import imp
import time
import threading


#####
#     threadsafe_generator
#####

#https://github.com/fchollet/keras/issues/1638
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


#####
#     METRICS & TOOLS
#####


def jaccard_coef(y_true, y_pred, smooth = 1e-12):
    """ jaccard coeficient implementation for keras """
    from keras import backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    sum_ = K.sum(y_true_f + y_pred_f)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def dice_coef(y_true, y_pred, smooth = 1.):
    """ dice coeficient implementation for keras """
    from keras import backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
       
def similarity_coef(y_true, y_pred, smooth = 1e-12, coef=(1,1,1,1), weights=None,
                    loss=False, drop=None, axis=None,
                    binary_cross_coef=None):
    """ Custom similarity coeficient for keras 
        For dice coeficient, coef = (2,1,1,2)
        For jaccard coeficient, coef = (1,1,1,1)
    """

    from keras import backend as K
    
    if drop is not None:
        y_true = y_true[:, : , :-drop]
        y_pred = y_pred[:, : , :-drop]
    
    if weights is not None:
        axis = [0,1]
        
    if axis is None:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        tp = K.sum(y_true_f * y_pred_f)
        fp = K.sum(y_pred_f) - tp
        fn = K.sum(y_true_f) - tp
    else:
        tp = K.sum(y_true * y_pred, axis=axis)
        fp = K.sum(y_pred, axis=axis) - tp
        fn = K.sum(y_true, axis=axis) - tp
        
    cof = (coef[0]*tp + smooth) / (coef[1]*fn + coef[2]*fp + coef[3]*tp +smooth)
    if weights is not None:
        rst = K.mean(cof * weights)
    else:
        rst = K.mean(cof)
    
    if binary_cross_coef is not None:
        binary_cross = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
        rst = rst - binary_cross_coef * binary_cross
    
    if loss:
        rst = - rst
        
    return rst 

def binary_cross(y_true, y_pred): # Output must be (batch_size, x*y, channels)
    # Eliminate last class
    from keras import backend as K
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
 

#####
#     SEGMENTATION - MODELS
#####

def get_UNET_C_r1(channels, isz, classes, args_dict={}):
    # C: customizable UNET, with inception modules
    # r1: version
    
    #libraries
    from keras.models import Model
    from keras.layers import Input,Activation, Dropout, BatchNormalization, concatenate, Reshape, Permute, Lambda
    from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2DTranspose
    from keras.layers.merge import add
    from keras.layers.advanced_activations import ELU, LeakyReLU
    from keras.optimizers import Adam
    import inspect
     
    # Functions 
    
    def NConvolution2D(cc_inputs, nb_conv, nb_filter, args_dict):
        
        # NConvolution2D - Parameters
        batch_norm = args_dict.get('batch_norm', False)
        bn_pos = args_dict.get('bn_pos', 'before')
        conv_activ = args_dict.get('conv_activ', 'all')  # all, end
        pad = args_dict.get('pad', 0)
        kernel = args_dict.get('kernel', 3)
        conv_strides = args_dict.get('conv_strides', 1)
        conv_padding = args_dict.get('conv_padding', 'same')
        init = args_dict.get('init', 'glorot_uniform')
        activation = args_dict.get('activation', 'relu')
        
        if pad > 0:
            res = ZeroPadding2D(padding=(pad,pad))(cc_inputs)
        else:
            res = cc_inputs
        
        for i_conv in range(nb_conv):
            res = Conv2D(nb_filter, (kernel, kernel) , strides=(conv_strides, conv_strides), 
                            kernel_initializer=init, padding=conv_padding)(res)  
            if conv_activ == 'all':
                if batch_norm and bn_pos == 'before':
                    res = BatchNormalization(axis=1)(res)
                res = Activation(activation)(res)
                if batch_norm and bn_pos == 'after':
                    res = BatchNormalization(axis=1)(res)
                
        return res
    
    def NConvDilation2D(cc_inputs, nb_conv, nb_filter, args_dict):
        
        # NConvolution2D - Parameters
        batch_norm = args_dict.get('batch_norm', False)
        bn_pos = args_dict.get('bn_pos', 'before')
        pad = args_dict.get('pad', 0)
        kernel = args_dict.get('kernel', 3)
        conv_strides = args_dict.get('conv_strides', 1)
        conv_padding = args_dict.get('conv_padding', 'same')
        init = args_dict.get('init', 'glorot_uniform')
        activation = args_dict.get('activation', 'relu')
        
        if pad > 0:
            res = ZeroPadding2D(padding=(pad,pad))(cc_inputs)
        else:
            res = cc_inputs
        
        res = Conv2D(nb_filter, (kernel, kernel) , strides=(conv_strides, conv_strides), 
                            kernel_initializer=init, padding=conv_padding)(res)  
        
        for i in range(1, nb_conv):
            dr = 2 * i
            res = Conv2D(nb_filter, (kernel, kernel) , strides=(conv_strides, conv_strides), 
                            kernel_initializer=init, padding=conv_padding, dilation_rate=dr)(res)         
        
        if batch_norm and bn_pos == 'before':
            res = BatchNormalization(axis=1)(res)
        res = Activation(activation)(res)
        if batch_norm and bn_pos == 'after':
            res = BatchNormalization(axis=1)(res)

        return res

    def NConcDilation2D(cc_inputs, nb_conv, nb_filter, args_dict):
        
        # NConvolution2D - Parameters
        batch_norm = args_dict.get('batch_norm', False)
        bn_pos = args_dict.get('bn_pos', 'before')
        pad = args_dict.get('pad', 0)
        kernel = args_dict.get('kernel', 3)
        conv_strides = args_dict.get('conv_strides', 1)
        conv_padding = args_dict.get('conv_padding', 'same')
        init = args_dict.get('init', 'glorot_uniform')
        activation = args_dict.get('activation', 'relu')
        
        if pad > 0:
            res = ZeroPadding2D(padding=(pad,pad))(cc_inputs)
        else:
            res = cc_inputs
        res_conc = []
        res = Conv2D(nb_filter, (kernel, kernel) , strides=(conv_strides, conv_strides), 
                            kernel_initializer=init, padding=conv_padding)(res)  
        res_conc.append(res)
        for i in range(1, nb_conv):
            dr = 2 * i
            res = Conv2D(nb_filter, (kernel, kernel) , strides=(conv_strides, conv_strides), 
                            kernel_initializer=init, padding=conv_padding, dilation_rate=dr)(res)         
            res_conc.append(res)
            
        res = concatenate(res_conc, axis=1)
        
        if batch_norm and bn_pos == 'before':
            res = BatchNormalization(axis=1)(res)
        res = Activation(activation)(res)
        if batch_norm and bn_pos == 'after':
            res = BatchNormalization(axis=1)(res)

        return res

    def inception_block(inputs, depth, 
                        splitted=False, activation='relu'):
        
        assert depth % 16 == 0
        actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
        
        c1_1 = Conv2D(depth/4, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
        
        c2_1 = Conv2D(depth/8*3, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
        c2_1 = actv()(c2_1)
        if splitted:
            c2_2 = Conv2D(depth/2, (1, 3), kernel_initializer='he_normal', padding='same')(c2_1)
            c2_2 = BatchNormalization(axis=1)(c2_2)
            c2_2 = actv()(c2_2)
            c2_3 = Conv2D(depth/2, (3, 1), kernel_initializer='he_normal', padding='same')(c2_2)
        else:
            c2_3 = Conv2D(depth/2, (3, 3), kernel_initializer='he_normal', padding='same')(c2_1)
        
        c3_1 = Conv2D(depth/16, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
        #missed batch norm
        c3_1 = actv()(c3_1)
        if splitted:
            c3_2 = Conv2D(depth/8, (1, 5), kernel_initializer='he_normal', padding='same')(c3_1)
            c3_2 = BatchNormalization(axis=1)(c3_2)
            c3_2 = actv()(c3_2)
            c3_3 = Conv2D(depth/8, (5, 1), kernel_initializer='he_normal', padding='same')(c3_2)
        else:
            c3_3 = Conv2D(depth/8, (5, 5), kernel_initializer='he_normal', padding='same')(c3_1)
        
        p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(inputs)
        c4_2 = Conv2D(depth/8, (1, 1), kernel_initializer='he_normal', padding='same')(p4_1)
        
        res = concatenate([c1_1, c2_3, c3_3, c4_2], axis=1)
        res = BatchNormalization(axis=1)(res)
        res = actv()(res)
        return res    
        
    def _shortcut(_input, residual):
        stride_width = _input._keras_shape[2] / residual._keras_shape[2]
        stride_height = _input._keras_shape[3] / residual._keras_shape[3]
        equal_channels = residual._keras_shape[1] == _input._keras_shape[1]
    
        shortcut = _input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(residual._keras_shape[1], (1, 1),
                                     strides=(stride_width, stride_height),
                                     kernel_initializer="he_normal", padding="valid")(_input)
    
        return add([shortcut, residual])

    def rblock(inputs, num, depth, scale=0.1):    
        residual = Conv2D(depth, (num, num), padding='same')(inputs)
        residual = BatchNormalization(axis=1)(residual)
        residual = Lambda(lambda x: x*scale, output_shape = lambda x: x)(residual)
        res = _shortcut(inputs, residual)
        return ELU()(res)  

    # Unet - Parameters
    convs   = args_dict.get('convs', [ [2, 2, 2, 2], 1, [2, 2, 2, 2] ])  #down-mid-up
    filters = args_dict.get('filters', [ [16, 32, 64, 128], 256, [128, 64, 32, 16] ])  #down-mid-up
    drops   = args_dict.get('drops', [ [0, 0, 0, 0], 0, [0, 0, 0, 0] ])   #down-mid-up
        
    down_size = args_dict.get('down_size', 2)
    up_size = args_dict.get('up_size', 2)
    up_drop_pos = args_dict.get('up_drop_pos', 'after_conv')  #after_conv, before_conv
    learnable = args_dict.get('learnable', False)
    strides = args_dict.get('strides', False)
    strides_conc = args_dict.get('strides_conc', 'conv')  # conv, pool, input, 
    strides_type = args_dict.get('strides_type', '')  # '', residual, vgg
    strides_vgg_convs = args_dict.get('strides_vgg_convs', [ 1, 1, 1, 1 ])
    strides_vgg_filters = args_dict.get('strides_vgg_filters', [ 8, 16, 32, 64 ])
    
    type_filters = args_dict.get('type_filters', '')  #'', ConvDila, ConcDila, inception
    conv_activ = args_dict.get('conv_activ', 'all')  # all, end
    
    init = args_dict.get('init', 'glorot_uniform')
    activation = args_dict.get('activation', 'relu')
    batch_norm = args_dict.get('batch_norm', False)
    bn_pos = args_dict.get('bn_pos', 'before')
    
    final_activation = args_dict.get('final_activation', 'sigmoid')
    flat_output = args_dict.get('flat_output', False)
    compile_conf = args_dict.get('compile_conf', 0)
    optimizer = args_dict.get('optimizer', Adam())
    optimizer = optimizer() if inspect.isclass(optimizer) else optimizer
    loss = args_dict.get('loss', 'binary_crossentropy')
    metrics = args_dict.get('metrics', [])
    
    inputs = Input((channels, isz[0], isz[1]))
    
    # Down sampling
    net = inputs
    down_inputs = []
    down_convs = []
    down_pools = []
    down_pools.append(None)
    for i in range(len(filters[0])):
        down_inputs.append(net)
        if type_filters == 'ConvDila':
            conv = NConvDilation2D(net, convs[0][i], filters[0][i], args_dict)
            down_convs.append(conv)
        if type_filters == 'ConcDila':
            conv = NConcDilation2D(net, convs[0][i], filters[0][i], args_dict)
            down_convs.append(conv)
        if type_filters == 'inception':
            conv = inception_block(net, filters[0][i], splitted=True, activation=activation)    
            down_convs.append(conv)
        else:
            conv = NConvolution2D(net, convs[0][i], filters[0][i], args_dict)
            down_convs.append(conv)    
            if conv_activ == 'end':
                if batch_norm and bn_pos == 'before':
                    conv = BatchNormalization(axis=1)(conv)
                conv = Activation(activation)(conv)
                if batch_norm and bn_pos == 'after':
                    conv = BatchNormalization(axis=1)(conv)
            
        if learnable:
            pool = Conv2D(filters[0][i], (3, 3) , strides=(down_size, down_size), 
                          kernel_initializer=init, padding='same')(conv)
        else:
            pool = MaxPooling2D(pool_size=(down_size, down_size))(conv)
            
        down_pools.append(pool)
        net = pool
        if drops[0][i] > 0:
            net = Dropout(drops[0][i])(net)
    
    # Mid
    if type_filters == 'ConvDila':
        conv = NConvDilation2D(net, convs[1], filters[1], args_dict)
    if type_filters == 'ConcDila':
        conv = NConcDilation2D(net, convs[1], filters[1], args_dict)
    if type_filters == 'inception':
        conv = inception_block(net, filters[1], splitted=True, activation=activation) 
    else:
        conv = NConvolution2D(net, convs[1], filters[1], args_dict)   
        if conv_activ == 'end':
            if batch_norm and bn_pos == 'before':
                conv = BatchNormalization(axis=1)(conv)
            conv = Activation(activation)(conv)
            if batch_norm and bn_pos == 'after':
                conv = BatchNormalization(axis=1)(conv)
            
    net = conv
    if drops[1] > 0:
        net = Dropout(drops[1])(net)
    
    # Up sampling
    inv = list(reversed(range(len(filters[0]))))
    up_ch = filters[1]
    for i in range(len(filters[2])):   
        if learnable:
            up = Conv2DTranspose(up_ch, (3, 3) , strides=(2, 2), kernel_initializer=init, 
                           padding="same")(net)
        else:
            up = UpSampling2D(size=(up_size, up_size))(net)
        if strides:
            if strides_conc == 'conv':
                net_conc = down_convs[inv[i]]
            elif strides_conc == 'pool':
                net_conc = down_pools[inv[i]]
            elif strides_conc == 'input':
                net_conc = down_inputs[inv[i]]
            
            if strides_type == 'residual' and net_conc is not None:
                net_conc = rblock(net_conc, 1, filters[2][i])
            elif strides_type == 'vgg' and net_conc is not None:
                net_conc = NConvDilation2D(net_conc, strides_vgg_convs[inv[i]], strides_vgg_filters[inv[i]], args_dict)
            
            if net_conc is None:
                conc = up
            else:
                conc = concatenate([up, net_conc], axis=1)
                
        else:
            conc = up
        
        if up_drop_pos == 'before_conv' and drops[2][i] > 0:
            conc = Dropout(drops[2][i])(conc)
        
        if type_filters == 'ConvDila':
            conv = NConvDilation2D(conc, convs[2][i], filters[2][i], args_dict)
        if type_filters == 'ConcDila':
            conv = NConcDilation2D(conc, convs[2][i], filters[2][i], args_dict)
        if type_filters == 'inception':
            conv = inception_block(conc, filters[2][i], splitted=True, activation=activation)  
        else:
            conv = NConvolution2D(conc, convs[2][i], filters[2][i], args_dict)
            if conv_activ == 'end':
                if batch_norm and bn_pos == 'before':
                    conv = BatchNormalization(axis=1)(conv)
                conv = Activation(activation)(conv)
                if batch_norm and bn_pos == 'after':
                    conv = BatchNormalization(axis=1)(conv)
        up_ch = filters[2][i]    
        
        net = conv
        if up_drop_pos == 'after_conv' and drops[2][i] > 0:
            net = Dropout(drops[2][i])(net)
    
    if flat_output:
        final = Conv2D(classes, (1, 1) , strides=(1, 1), kernel_initializer=init, padding="same")(net)  
        final = Reshape((classes,isz[0]*isz[1]))(final)
        final = Permute((2,1))(final)
        final = Activation(final_activation)(final) 
    else:
        final = Conv2D(classes, (1, 1) , strides=(1, 1), kernel_initializer=init, padding="same")(net)  
        final = Activation(final_activation)(final) 
    
    model = Model(inputs=inputs, outputs=final)
    
    if compile_conf == 0:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if compile_conf == 1:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, sample_weight_mode="temporal")

    return model

#####
#     CLASSIFICATION - MODELS
#####

def get_CNN_C_r0(channels, isz, classes, args_dict={}):
    # C: adding SeparableConv2D
    # r0: revision
    
    #libraries
    from keras.models import Model
    from keras.layers import Input,Activation, Dropout, BatchNormalization, Dense, Flatten
    from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, SpatialDropout2D, SeparableConv2D
    from keras.optimizers import Adam
    import inspect
     
    # Functions 
    def conv_module(cc_inputs, nb_filter, separableConv=False, args_dict={}):
        
        # conv_module - Parameters
        batch_norm = args_dict.get('batch_norm', False)
        activation = args_dict.get('activation', 'relu')
        init = args_dict.get('init', 'glorot_uniform')
        depth_multiplier = args_dict.get('depth_multiplier', 1)
        
        default = {'pad': 0,
                   'kernel': 3,
                   'conv_strides': 1,
                   'conv_padding': "same"}
        conv_Blk_args = args_dict.get('conv_Blk_args', default)
        pad = conv_Blk_args.get('pad', default.get('pad'))
        kernel = conv_Blk_args.get('kernel', default.get('kernel'))
        conv_strides = conv_Blk_args.get('conv_strides', default.get('conv_strides'))
        conv_padding = conv_Blk_args.get('conv_padding', default.get('conv_padding'))
        
        if pad > 0:
            res = ZeroPadding2D(padding=(pad,pad))(cc_inputs)
        else:
            res = cc_inputs
        
        if separableConv:
            res = SeparableConv2D(nb_filter, (kernel, kernel) , strides=(conv_strides, conv_strides), 
                            kernel_initializer=init, padding=conv_padding,
                            depth_multiplier=depth_multiplier)(res)        
        else:
            res = Conv2D(nb_filter, (kernel, kernel) , strides=(conv_strides, conv_strides), 
                            kernel_initializer=init, padding=conv_padding)(res) 
            
        if batch_norm:
            res = BatchNormalization(axis=1)(res)
        res = Activation(activation)(res)

        return res
    
    def conv_Blk(cc_inputs, nb_convs, nb_filter, pool=True, drop=0.5, separable=False, args_dict={}):
        
        # conv_module - Parameters
        default = {'pool_size': 2, 
                   'pool_strides': 2,
                   'pool_padding': "same"}
        conv_Blk_args = args_dict.get('conv_Blk_args', default)
        pool_size    = conv_Blk_args.get('pool_size', default.get('pool_size'))
        pool_strides = conv_Blk_args.get('pool_strides', default.get('pool_strides'))
        pool_padding = conv_Blk_args.get('pool_padding', default.get('pool_padding'))
        
        res = cc_inputs
        for i in range(nb_convs):
            res = conv_module(res, nb_filter, separable, args_dict)
            
        if pool:
            res =  MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_strides, pool_strides), 
                           padding=pool_padding)(res)
        
        if drop > 0:
            res = Dropout(drop)(res)
        
        return res

    def dense_Blk(cc_inputs, n_size, drop=0.5, args_dict={}):
        
        # Dense block - Parameters
        batch_norm = args_dict.get('batch_norm', False)
        init = args_dict.get('init', 'glorot_uniform')
        activation = args_dict.get('activation', 'relu')
        
        res = cc_inputs
        
        res = Dense(n_size, kernel_initializer=init)(res)
        if batch_norm:
            res = BatchNormalization(axis=1)(res)
        res = Activation(activation)(res)
        if drop > 0:
            res = Dropout(drop)(res)

        return res
        
    # squeezeNet - Parameters
    channels_drop = args_dict.get('channels_drop', None)
    conv_Blks = args_dict.get('conv_Blks', [[1, 64, True,  None, False],
                                            [1, 64, True,  None, False],
                                            [1, 64, False,  0.5, False],
                                            ])   
    dense_Blks = args_dict.get('dense_Blks', [[256, 0.5],
                                              ])  
    init = args_dict.get('init', 'glorot_uniform')
    final_activation = args_dict.get('final_activation', 'softmax')
    compile_conf = args_dict.get('compile_conf', 0)
    optimizer = args_dict.get('optimizer', Adam())
    optimizer = optimizer() if inspect.isclass(optimizer) else optimizer
    loss = args_dict.get('loss', 'binary_crossentropy')
    metrics = args_dict.get('metrics', [])

                                        
    inputs = Input((channels, isz[0], isz[1]))  # (None, ch, x, y)
    
    if channels_drop is not None:
        inputsD = SpatialDropout2D(channels_drop)(inputs) 
    else:
        inputsD = inputs
    
    conv = inputsD
    for i_Blk in conv_Blks:
        conv = conv_Blk(conv, nb_convs   = i_Blk[0], 
                              nb_filter  = i_Blk[1], 
                              pool       = i_Blk[2],  
                              drop       = i_Blk[3], 
                              separable  = i_Blk[4], 
                              args_dict=args_dict)
    
    dens = Flatten()(conv)
    for i_Blk in dense_Blks:
        dens = dense_Blk(dens, n_size  = i_Blk[0],
                                 drop    = i_Blk[1], 
                                 args_dict=args_dict)
    
    final = Dense(classes, kernel_initializer=init, activation=final_activation)(dens) 
    
    model = Model(inputs=inputs, outputs=final)
    
    if compile_conf == 0:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model     