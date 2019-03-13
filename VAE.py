# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:41:24 2019

@author: sushil
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models,layers
from keras import applications
import glob2 as glob
from numpy import random
from tqdm import tqdm
import os
import cv2
from random import shuffle
from scipy.misc import imresize, imsave

#%%
# dimensionality of the latents space 
embedding_dim = 64

#Input layer
input_img = layers.Input(shape=(4096,))  

#Encoding layer
encoded = layers.Dense(embedding_dim, activation='relu')(input_img)

#Decoding layer
decoded = layers.Dense(4096,activation='sigmoid')(encoded) 

autoencoder = models.Model(input_img,decoded)
autoencoder.summary()

#Encoder
encoder = models.Model(input_img,encoded)
encoder.summary()

#Decoder
encoded_input = layers.Input(shape=(embedding_dim,))
decoder_layers = autoencoder.layers[-1]  #applying the last layer
decoder = models.Model(encoded_input,decoder_layers(encoded_input))

print(input_img)
print(encoded)

#%%

autoencoder.compile(
    optimizer='adadelta',  #backpropagation Gradient Descent
    loss='binary_crossentropy'
)
#%%

Train_dir = 'Facedata'
Img_size = 64

#%%
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
    elif word_label =='John_Aschcroft':
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
        print(path,label)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('sc_train_data.npy',training_data)
    return training_data
    
#%%
    
train_data = create_training_data()


train = train_data[:-1] 
test = train_data[-5:]
#%%
x_train = np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
x_test = np.array([i[0] for i in train])

#%%
#Normalization
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

x_train = x_train.reshape((-1,4096))  
                                    
x_test = x_train.reshape((-1,4096))

print(x_train.shape,x_test.shape)

#%%

history = autoencoder.fit(x_train,x_train,epochs=500,batch_size=5,shuffle=True,
                validation_data=(x_test,x_test))
                
#%%
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()
#%%


encoded_imgs = encoder.predict(x_test) 
decoded_imgs = decoder.predict(encoded_imgs)  
print(encoded_imgs.shape,decoded_imgs.shape)

#%%

n = 50
for i in range(n):
    decoded=decoded_imgs[i].reshape((64,64))
    #cv2.imshow('output',decoded)
    cv2.waitKey(0)
    cv2.imwrite('Tony1/Tony_Blair.'+ str(i)+'.jpeg',255*decoded)
    #fig.savefig('face'+ str(i)+ '.jpg') use this to save the fig when plt is used
   

    
save_img()    
plt.show()
 

