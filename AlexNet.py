# coding=utf-8
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np
form keras.datasets import mnist

seed = 7
np.random.seed(seed)
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number,227*227)
    x_test = x_test.reshape(x_test.shape[0],227*227)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #
    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)
    x_train = x_train
    x_test = x_test
    #
    x_train = x_train/255
    x_test = x_test/255
    return (x_train,y_train),(x_test,y_test)
(x_train,y_train),(x_test,y_test) = load_data()
model = Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(227, 227, 1), padding='valid', activation='relu',
                 kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
result = model.evaluate(x_test,y_test)
print('\nTest Acc',result[1])