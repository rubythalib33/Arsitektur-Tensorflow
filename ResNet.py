import tensorflow as tf
from keras.models import Sequential, Model,
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Add, Input, ZeroPadding2D, AveragePooling2D
from keras.initializers import glorot_uniform

def identity_block(X,f,filters, stage, block):
    #defining name basis
    conv_name_base = 'res' +str(stage)+block+'_branch'
    bn_name_base = 'bn' +str(stage)+block+'_branch'

    #Retrive Filters
    F1,F2,F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1,1), strides = (1,1), padding='valid', name = conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f,f), strides = (1,1), padding='same', name = conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1,1), strides = (1,1), padding='valid', name = conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' +str(stage)+block+'_branch'
    bn_name_base = 'bn' +str(stage)+block+'_branch'

    F1,F2,F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1,1), strides = (s,s), name = conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f,f), strides = (1,1),padding='same', name = conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1,1), strides = (1,1),padding='valid', name = conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides = (s,s), name = conv_name_base+'1',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base+'1')(X_shortcut)
    X = Add() ([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape = (224,224,3), classes = 200):
    X_input = Input(input_shape)

    #Zero padding
    X = ZeroPadding2D((3,3))(X_input)

    #stage 1
    X = Conv2D(64,(7,7),strides=(2,2), name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)

    #stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage = 2, block='a', s=1)
    X = identity_block(X, 3, [64,64,256], stage=2, block='b')
    X = identity_block(X,3,[64,64,256], stage = 2, block = 'c')

    #stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage = 3, block='a', s=2)
    X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X,3,filters=[128, 128, 512], stage = 3, block = 'c')
    X = identity_block(X,3,filters=[128, 128, 512], stage = 3, block = 'd')

    #stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage = 4, block='a', s=2)
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X,3,filters=[256, 256, 1024], stage = 4, block = 'c')
    X = identity_block(X,3,filters=[256, 256, 1024], stage = 4, block = 'd')
    X = identity_block(X,3,filters=[256, 256, 1024], stage = 4, block = 'f')

    #stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage = 5, block='a', s=2)
    X = identity_block(X, 3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X,3,filters=[512, 512, 2048], stage = 5, block = 'c')

    X = AveragePooling2D((2,2), name = 'avg_pool')(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc'+str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
