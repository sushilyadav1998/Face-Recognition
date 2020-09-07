# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:30:54 2019

@author: sushil
"""

#%%
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import time

#%%
Test_dir = 'TestData'
Train_dir = '(64)AE_Database'
Img_size = 64
LR = 0.01
Model_name = 'FaceRecognition-{}-{}.model'.format(LR,'6-conv-basic')


#%% Process data
def label_img(img):
    word_label = img.split('.')[0]
    if word_label =='Ariel_Sharon':
        return [1,0,0,0,0,0,0,0,0,0,0]
    elif word_label =='Colin_Powell':
        return [0,1,0,0,0,0,0,0,0,0,0]
    elif word_label =='George_Bush':
        return [0,0,1,0,0,0,0,0,0,0,0]
    elif word_label =='Gerhard_Schroed':
        return [0,0,0,1,0,0,0,0,0,0,0]
    elif word_label =='Hugo_Chavez':
        return [0,0,0,0,1,0,0,0,0,0,0]
    elif word_label =='Jacques_Chirac':
        return [0,0,0,0,0,1,0,0,0,0,0]
    elif word_label =='Jean_Chretien':
        return [0,0,0,0,0,0,1,0,0,0,0]
    elif word_label =='John_Ashcroft':
        return [0,0,0,0,0,0,0,1,0,0,0]
    elif word_label =='Junichiro_Koizumi':
        return [0,0,0,0,0,0,0,0,1,0,0]
    elif word_label =='Serena_Williams':
        return [0,0,0,0,0,0,0,0,0,1,0]
    elif word_label =='Tony_Blair':
        return [0,0,0,0,0,0,0,0,0,0,1]

#%%
def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(Train_dir)):
        label = label_img(img)
        if label==None:
            continue
        path = os.path.join(Train_dir,img)
        #print(path,label)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    #np.save('sc_train_data.npy',training_data)
    return training_data


#%%
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(Test_dir)):
        path = os.path.join(Test_dir,img)
        img_num = img.split('.')[0]
        img =cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
        testing_data.append([np.array(img),np.array(img_num)])
    np.save('sc_test_data.npy',testing_data)
    return testing_data

#%%Load Traing and Testing Data:
train_data = create_training_data()
#if already created the npy file use:
#train_data = np.load('sc_train_data.npy')
#%%

test_data = process_test_data()
#if already created use:
#test_data = np.load('sc_test_data.npy')
#%%
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()
convnet = input_data(shape=[None, Img_size, Img_size, 1], name='input')

convnet = conv_2d(convnet, 64, 11, activation='relu')
convnet = conv_2d(convnet, 96, 5, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 256, 3, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 384, 3, activation='relu')
convnet = conv_2d(convnet, 384, 3, activation='relu')
convnet = conv_2d(convnet, 256, 5, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 11, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

#%%
if os.path.exists('{}.meta'.format(Model_name)):
    model.load(Model_name)
    print("[INFO] Model Loaded")
    time.sleep(3.0)

#%%
#else:
train = train_data[:-50] 
test = train_data[-50:]
#%%
X = np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, Img_size, Img_size, 1)
test_y = np.array([i[1] for i in test])

#%%
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=Model_name)
model.save(Model_name)

#%%
import matplotlib.pyplot as plt
fig =plt.figure(figsize=(10,10))

for num,data in enumerate(test_data[:11]):
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3,4,num+1)
    og = img_data
    img_data = img_data.reshape(Img_size,Img_size,1)
    model_output = model.predict([img_data])
    print(model_output)
    if np.argmax(model_output) == 0: str_label = 'Ariel_Sharon'
    elif np.argmax(model_output) == 1: str_label = 'Colin_Powell'
    elif np.argmax(model_output) == 2: str_label = 'George_Bush'
    elif np.argmax(model_output) == 3: str_label = 'Gerhard_Schroede'
    elif np.argmax(model_output) == 4: str_label = 'Hugo_Chavez'
    elif np.argmax(model_output) == 5: str_label = 'Jacques_Chirac'
    elif np.argmax(model_output) == 6: str_label = 'Jean'
    elif np.argmax(model_output) == 7: str_label = 'John_Ashcroft'
    elif np.argmax(model_output) == 8: str_label = 'Junichiro_Koimuzi'
    elif np.argmax(model_output) == 9: str_label = 'Serena_Williams'
    elif np.argmax(model_output) == 10: str_label = 'Tony_Blair'
    
    else: str_label = 'cat'
    
    y = plt.imshow(og,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
    