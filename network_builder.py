import tensorflow as tf

import keras
import keras.backend as K

from keras.utils import np_utils, multi_gpu_model
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import glorot_normal, RandomNormal, Zeros
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from resnet import ResnetBuilder

class NetworkBuilder:
    def __init__(self):
        pass

    def build_simplenet(self, shape, num_classes):
        self.model = Sequential()

        # Block 1
        self.model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=glorot_normal(), input_shape=shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 2
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 3
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 4
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        # First Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        self.model.add(Dropout(0.2))
        
        
        # Block 5
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 6
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 7
        self.model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
        # Second Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        
        # Block 8
        self.model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 9
        self.model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        # Third Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        
        
        # Block 10
        self.model.add(Conv2D(512, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        # Block 11  
        self.model.add(Conv2D(2048, (1,1), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 12  
        self.model.add(Conv2D(256, (1,1), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(Activation('relu'))
        # Fourth Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        self.model.add(Dropout(0.2))

        # Block 13
        self.model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(Activation('relu'))
        # Fifth Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))

        # Final Classifier
        self.model.add(Flatten())
        self.model.add(Dense(num_classes, activation='softmax'))        
        
        print ('[+] SimpleNet model built')
        return self.model

    def build_vgg_16(self, shape, num_classes):
        self.model = Sequential()

        # Block 1
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=shape))
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 2
        self.model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 3
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 4
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 5
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Final Classifier
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))        
        
        print ('[+] VGG-16 model built')
        return self.model