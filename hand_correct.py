# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:03:24 2019

@author: reshma
"""


#correct
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import cv2

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test) 

num_classes = 10


model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10)


model.save('digit.h5')


#run
image=cv2.imread('pic3.png',cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(28,28))
ret, img_pred = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
img_pred = img_pred. reshape ( 1 , 28 , 28 , 1 ) . astype ( 'float32' )
img = img_pred / 255.0


pred = model. predict_classes ( img )
pred_proba = model. predict_proba ( img )

pred_proba = "% .2f %%" % (pred_proba [0] [pred] * 100) 

print( pred[0] , "with confidence of" , pred_proba )

