# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import h5py
import numpy as np
import sys
sys.setrecursionlimit(5000)

WEIGHTS_PATH = '../weights/'


def get_learning_rate(cnn_type):
    if cnn_type == 'RESNET50' or cnn_type == 'RESNET50_DENSE_LAYERS':
        return 0.00003
    elif cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V3_DENSE_LAYERS':
        return 0.00003
    elif cnn_type == 'DENSENET_121':
        return 0.00003
    else:
        print('Error Unknown CNN type for learning rate!!')
        exit()
    return 0.00005


def get_optim(cnn_type, optim_type, learning_rate=-1):
    from keras.optimizers import SGD
    from keras.optimizers import Adam

    if learning_rate == -1:
        lr = get_learning_rate(cnn_type)
    else:
        lr = learning_rate
    if optim_type == 'Adam':
        optim = Adam(lr=lr)
    else:
        optim = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    return optim


def get_random_state(cnn_type):
    return 69


def get_input_shape(cnn_type):
    if cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V3_DENSE_LAYERS' or cnn_type == 'INCEPTION_V4' or cnn_type == 'XCEPTION':
        return (299, 299)
    elif cnn_type == 'SQUEEZE_NET':
        return (227, 227)
    return (224, 224)


# Tuned for 8 GB of GPU memory
def get_batch_size(cnn_type):
    if cnn_type == 'RESNET50' or cnn_type == 'RESNET50_DENSE_LAYERS':
        return 32
    if cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V3_DENSE_LAYERS':
        return 20
    if cnn_type == 'DENSENET_121':
        return 20
    return -1


def normalize_image_vgg16(img):
    img[:, 0, :, :] -= 103.939
    img[:, 1, :, :] -= 116.779
    img[:, 2, :, :] -= 123.68
    return img


def normalize_image_inception(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def normalize_image_densenet(img):
    img[:, 0, :, :] = (img[:, 0, :, :] - 103.94) * 0.017
    img[:, 1, :, :] = (img[:, 1, :, :] - 116.78) * 0.017
    img[:, 2, :, :] = (img[:, 2, :, :] - 123.68) * 0.017
    return img


def preprocess_input_overall(cnn_type, x):
    if cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V3_DENSE_LAYERS' or cnn_type == 'INCEPTION_V4' or cnn_type == 'XCEPTION':
        return normalize_image_inception(x.astype(np.float32))
    if 'DENSENET' in cnn_type:
        return normalize_image_densenet(x.astype(np.float32))
    return normalize_image_vgg16(x.astype(np.float32))


def RESNET_50(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.resnet50 import ResNet50
    from keras.models import Model

    base_model = ResNet50(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    return model


# Batch 40 OK
def Inception_V3(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model

    base_model = InceptionV3(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    return model


def DenseNet121(classes_number, final_layer_activation):
    from a01_densenet_121 import DenseNet_121
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Model

    base_model = DenseNet_121(reduction=0.5, weights_path=WEIGHTS_PATH + 'densenet121_weights_th.h5')
    x = base_model.layers[-3].output
    del base_model.layers[-2:]
    x = Dense(classes_number, activation=final_layer_activation, name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    # print(model.summary())
    return model


def get_pretrained_model(cnn_type, classes_number, optim_name='Adam', learning_rate=-1, final_layer_activation='sigmoid'):
    import keras
    K = keras.backend.backend()
    if K == 'tensorflow':
        print('Update dim ordering to "tf"')
        keras.backend.set_image_dim_ordering('tf')

    if cnn_type == 'RESNET50':
        model = RESNET_50(classes_number)
    elif cnn_type == 'INCEPTION_V3':
        model = Inception_V3(classes_number)
    elif cnn_type == 'DENSENET_121':
        model = DenseNet121(classes_number, final_layer_activation)
    else:
        model = None
        print('Unknown CNN type: {}'.format(cnn_type))
        exit()

    optim = get_optim(cnn_type, optim_name, learning_rate)
    if final_layer_activation == 'sigmoid':
        model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['binary_crossentropy'])
    else:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['binary_crossentropy', 'accuracy'])

    return model

'''
DenseNet: https://github.com/flyyufelix/DenseNet-Keras
ResNet-101: https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294
ResNet-152: https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6
SqueezeNet: https://github.com/rcmalli/keras-squeezenet
Inception v4: https://github.com/titu1994/Inception-v4/releases
VGG16: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
VGG19: https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
Other Keras models: https://keras.io/applications/
'''