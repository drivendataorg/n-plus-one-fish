# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


def preprocess_batch(batch):
    batch /= 256.0
    batch -= 0.5
    return batch


def dice_coef(y_true, y_pred):
    from keras import backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    from keras import backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def single_conv_layer(x, size, dropout, batch_norm):
    from keras.layers import Convolution2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import Activation, SpatialDropout2D
    conv = Convolution2D(size, 3, 3, border_mode='same')(x)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=1)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def ZF_UNET_1280_720_V2_SINGLE_OUTPUT_SMALL(dropout_val=0.0, batch_norm=True):
    from keras.models import Model
    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
    from keras.layers.core import Activation
    filters = 8

    inputs = Input((3, 720, 1280))
    conv_720_1280 = single_conv_layer(inputs, 1 * filters, dropout_val, batch_norm)
    pool_240_320 = MaxPooling2D(pool_size=(3, 4))(conv_720_1280)

    conv_240_320 = single_conv_layer(pool_240_320, 2 * filters, dropout_val, batch_norm)
    pool_80_80 = MaxPooling2D(pool_size=(3, 4))(conv_240_320)

    conv_80_80 = single_conv_layer(pool_80_80, 4 * filters, dropout_val, batch_norm)
    pool_40_40 = MaxPooling2D(pool_size=(2, 2))(conv_80_80)

    conv_40_40 = single_conv_layer(pool_40_40, 8 * filters, dropout_val, batch_norm)
    pool_20_20 = MaxPooling2D(pool_size=(2, 2))(conv_40_40)

    conv_20_20 = single_conv_layer(pool_20_20, 16 * filters, dropout_val, batch_norm)
    pool_10_10 = MaxPooling2D(pool_size=(2, 2))(conv_20_20)

    conv_10_10 = single_conv_layer(pool_10_10, 32 * filters, dropout_val, batch_norm)
    pool_5_5 = MaxPooling2D(pool_size=(2, 2))(conv_10_10)

    conv_5_5 = single_conv_layer(pool_5_5, 64 * filters, dropout_val, batch_norm)

    up_10_10 = merge([UpSampling2D(size=(2, 2))(conv_5_5), conv_10_10], mode='concat', concat_axis=1)
    conv_up_10_10 = single_conv_layer(up_10_10, 32 * filters, dropout_val, batch_norm)

    up_20_20 = merge([UpSampling2D(size=(2, 2))(conv_up_10_10), conv_20_20], mode='concat', concat_axis=1)
    conv_up_20_20 = single_conv_layer(up_20_20, 16 * filters, dropout_val, batch_norm)

    up_40_40 = merge([UpSampling2D(size=(2, 2))(conv_up_20_20), conv_40_40], mode='concat', concat_axis=1)
    conv_up_40_40 = single_conv_layer(up_40_40, 8 * filters, dropout_val, batch_norm)

    up_80_80 = merge([UpSampling2D(size=(2, 2))(conv_up_40_40), conv_80_80], mode='concat', concat_axis=1)
    conv_up_80_80 = single_conv_layer(up_80_80, 4 * filters, dropout_val, batch_norm)

    up_240_320 = merge([UpSampling2D(size=(3, 4))(conv_up_80_80), conv_240_320], mode='concat', concat_axis=1)
    conv_up_240_320 = single_conv_layer(up_240_320, 2 * filters, dropout_val, batch_norm)

    up_720_1280 = merge([UpSampling2D(size=(3, 4))(conv_up_240_320), conv_720_1280], mode='concat', concat_axis=1)
    conv_up_720_1280 = single_conv_layer(up_720_1280, 1 * filters, dropout_val, batch_norm)

    conv_final = Convolution2D(1, 1, 1)(conv_up_720_1280)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(input=inputs, output=conv_final)
    return model