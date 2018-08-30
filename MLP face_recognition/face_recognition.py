#!/usr/bin/python
#coding:utf-8
__author__ = 'kinoshita kenta'
__email__  = 'ybs0306748@gmail.com'

import numpy as np
import keras
import os
import cv2
#import pydot
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

#專案路徑
path = "C:\\Users\\Tom\\Desktop\\人臉辨識程式\\MLP face_recognition"
#path = "C:\\Users\\Owner\\Desktop\\人臉辨識程式\\MLP face_recognition"
#修改程式工作路徑
os.chdir(path)

#train data路徑名
fpath01 = 'img_sample'
#列出img_sample裡的檔案  ->list
flist = os.listdir(fpath01)

#照片長寬
h = 32
w = 32

#遍歷所有資料夾 算出總資料筆數 ->inum
inum = 0
for i in range(1,len(flist)+1):
    #路徑資料型態是字串 所以i要int轉str
    s1 = str(i)
    #資料夾檔案list
    filelist = os.listdir(fpath01 + '/' + s1)
    #inum += len(filelist)
    for j in range(1,len(filelist)+1):
        inum += 1


#x_train, y_train初始化
x_train = np.zeros([inum,h*w], dtype='float32')
y_train = np.zeros([inum])


#把照片讀進 x_train
inum = 0
for i in range(1,len(flist)+1):
    #路徑資料型態是字串 所以i要int轉str
    s1 = str(i)
    #資料夾檔案list
    filelist = os.listdir(fpath01 + '/' + s1)

    #每個資料夾裡的各個檔案
    for j in range(1,len(filelist)+1):
        #同理 int轉str
        s2 = str(j)
        #使用OpenCV讀取圖片 (路徑 + 讀取模式) 讀取模式 0為灰階 1為全彩
        x_imgs = cv2.imread(fpath01 + '/' + s1 + '/' + s2 + '.jpg',0)
        #x_imgs的值壓縮至0~1之間
        x_imgs = x_imgs.astype('float32')/255
        #把照片向量拉平 (向量化)
        x_imgs = np.reshape(x_imgs, h * w, 1)
        #將照片加入至x_train
        x_train[inum,:] = x_imgs.copy()
        #給予y_train相對應標籤
        y_train[inum] = i-1
        inum += 1


#x_train = x_train.reshape(x_train.shape[0], 1, 32, 32).astype('float32')

#將y_train編碼至model需要的型態
y_train = keras.utils.to_categorical(y_train, i)


#建立Sequential model
model = Sequential()
#加入隱藏層 input為1024 output為512
model.add(Dense(512,activation='relu',input_shape = (h*w,)))
model.add(Dropout(0.2))
#加入隱藏層 input為上一層(512) output為512
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
#加入輸出層 input為上一層(512) output為i = 總分類數量(資料夾數)
model.add(Dense(i,activation='softmax'))

'''
model = Sequential()
#model.add(Conv2D(32, (5, 5), input_shape=(1, 32, 32), activation='relu'))
model.add(Conv2D(32, (2, 2), input_shape=(1, 1, 1024), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dense(inum, activation='softmax'))
'''

plot_model(model, to_file='model.png')

#顯示模型架構
model.summary()
with open('model_summary.txt', 'w') as f:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn = lambda x: f.write(x + '\n'))

#Model_checkpoint = ModelCheckpoint('mlp_face.hdf5', monitor='loss', verbose=1, save_best_only=True)

#編譯模型訓練目標 此處為多分類(categorical_crossentropy)
model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
#訓練模型
#model.fit(x_train, y_train,batch_size=5,epochs=150,verbose=1)
#儲存模型
model.save('mlp_face_model.h5')
