#!/usr/bin/python
#coding:utf-8
__author__ = 'kinoshita kenta'
__email__  = 'ybs0306748@gmail.com'

import numpy as np
import keras
import os
import cv2

from keras.models import load_model

path = "C:\\Users\\Tom\\Desktop\\人臉辨識程式"
fpath01 = 'test_images'
dataset = os.listdir(fpath01)

h = 32
w = 32

#算出總資料筆數
inum = 0
for i in range(1,len(dataset)+1):
    s1 = str(i)
    dataset_class = os.listdir(fpath01)
    inum += 1

#x_train, y_train初始化
x_test = np.zeros([inum,1024], dtype='float32')

#把照片讀進 x_test
inum = 0
for i in range(1,len(dataset)+1):
    s1 = str(i)
    x_imgs = cv2.imread(fpath01 + '/' + s1 + '_test.jpg',0)
    x_imgs = x_imgs.astype('float32')/255
    x_imgs = np.reshape(x_imgs, h * w, 1)
    x_test[inum,:] = x_imgs.copy()
    inum += 1

#載入model並丟入test img
model = load_model('mlp_face_model.h5')
Y = model.predict(x_test,verbose=1)
#local = np.argmax(Y)
#Y1 = np.where(Y > 0.9,1,0)

#算位於第幾個分類
fpath01 = 'img_sample'
dataset = os.listdir(fpath01)

for i in range(0,inum):
	local = np.argmax(Y[i])
	print (local+1)

#    print (Y1[i,:])
