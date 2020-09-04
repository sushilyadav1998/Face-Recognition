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
input_img = layers.Input(shape=(16384,))  

#Encoding layer
encoded = layers.Dense(embedding_dim, activation='relu')(input_img)

#Decoding layer
decoded = layers.Dense(16384,activation='sigmoid')(encoded) 

#Autoencoder --> in this API Model, we define the Input tensor and the output layer
#wraps the 2 layers of Encoder e Decoder
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

Train_dir = 'Ariel_Sharon'
Img_size = 128

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
#test_data = process_test_data()


train = train_data[:-1] 
test = train_data[-5:]
#%%
x_train = np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
x_test = np.array([i[0] for i in train])

#test_x = np.array([i[0] for i in test]).reshape(-1, Img_size, Img_size, 1)
#test_y = np.array([i[1] for i in test])

#%%
#from keras.datasets import mnist
#import numpy as np

#x_train, x_test = train  #underscore for unanimous label that we don't
                                    # want to keep im memory
#Normalization
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

x_train = x_train.reshape((-1,16384))  #to go from (60000,28,28) to new shape and -1 let
                                    #numpy to calculate the number for you
x_test = x_train.reshape((-1,16384))

print(x_train.shape,x_test.shape)

#%%

history = autoencoder.fit(x_train,x_train,epochs=100,batch_size=5,shuffle=True,
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
    
    
    #saveimg=cv2.resize(saveimg1,(300,300))
#%%


#if already created use:
#test_data = np.load('sc_test_data.npy')
#%%

#%%

#n = 450
#
#for i in range(n):
#    saveimg1 = encoded_imgs[64,64]
#    #saveimg=cv2.resize(saveimg1,(300,300))
#    FaceFileName = ("new_imgs/face_" + str(y) + ".jpg")
#    cv2.imwrite(FaceFileName, saveimg)
# 

##%%
#directory = "imgs"
#new_dir = "new_imgs"
#filepaths = []
#for dir_, _, files in os.walk(directory):
#    for fileName in files:
#        relDir = os.path.relpath(dir_, directory)
#        relFile = os.path.join(relDir, fileName)
#        filepaths.append(directory + "/" + relFile)
#        
#for i, fp in enumerate(filepaths):
#    img = imread(fp) #/ 255.0
#    img = imresize(img, (40, 40))
#    imsave(new_dir + "/" + str(i) + ".png", encoded_imgs[i])
#%%

n = 50
#plt.figure(figsize=(20,4))
for i in range(n):
#    ax = plt.subplot(2, n, i+1)
#    plt.imshow(x_test[i].reshape((64,64)),cmap='gray')
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
    
#    ax = plt.subplot(2,n,i+1+n)
#    plt.imshow(decoded_imgs[i].reshape((64,64)),cmap='gray')
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
    decoded=decoded_imgs[i].reshape((128,128))
    #cv2.imshow('output',decoded)
    #cv2.waitKey(0)
    cv2.imwrite('samples/S.'+ str(i)+'.jpeg',255*decoded)
    #fig.savefig('face'+ str(i)+ '.jpg')
   
  #%%  
#n=10
    
#fig = plt.figure()
#for i in range(n):
#    saveimg = decoded_imgs[i]
#    saveimg1=cv2.resize(saveimg, (64,64))
#    #cv2.imshow('output', saveimg1)
#    FaceFileName = ("new_imgs/face"+ str(i) + ".jpg")
#    cv2.imwrite(FaceFileName, saveimg1)
    #cv2.imwrite('output', saveimg1)
#(FaceFileName, resized_output)
#%%
#n=10
#
#def save_img():
#    for i in range(n):
#          
#        
#        
#        #decoded_imgs = autoencoder.predict(x_train)
#    #test_img = x_test_noisy[0]
##    resized_test_img = cv2.resize(test_img, (64,64))
##    cv2.imshow('input', resized_test_img)
##    cv2.waitKey(0)
#        output = x_test[i]
#        resized_output = cv2.resize(output, (64,64))
#        #cv2.imshow('output', resized_output)
#        cv2.waitKey(0)
#        cv2.imwrite('new_imgs/face'+ str(i)+'.jpg',resized_output)
#        
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
    #cv2.imwrite('test_results/denoised_image.jpg')   
    
#%%
    #plt.imshow(decoded_imgs[i].reshape((64,64)),cmap='gray')
    #fig.savefig("new_imgs/face"+ str(i) + ".jpg")
    
#    FaceFileName = ("new_imgs/face"+ str(i) + ".jpg")
#    cv2.imwrite(FaceFileName, saveimg1)
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
    
    
    
    
    
#    ax = plt.subplot(3,n,i+1+2*n)
#    plt.imshow(encoded_imgs[i].reshape((64,64)),cmap='gray')
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
    
save_img()    
plt.show()
 

