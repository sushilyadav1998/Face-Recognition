  # -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:31:06 2020

@author: Sushilkumar.Yadav
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.models import load_model
from keras.layers import Lambda, Input, Dense, Conv2D
from keras.models import Model
from keras.optimizers import adam
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import cv2
from random import shuffle
from tqdm import tqdm
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import models
import matplotlib.cm as cm
from keras.layers import Reshape, Conv2DTranspose
from keras.losses import mse
from collections import Counter

#%%

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#%%
def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """
    
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    print(z_mean.shape)
    plt.figure(figsize=(12,11))
    plt.scatter(z_mean[:,0], z_mean[:,1], c = test2)
    
    #%%
    #counter = Counter(test2)
    
    #%%
    
    ys = np.unique(test2)
    means = np.array([np.mean(z_mean[test2 == y, :], axis=0) for y in ys])
    plt.scatter(means[:, 0], means[:, 1], c=ys, s=200, edgecolors='black')

    #%%
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 50
    digit_size = 64
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-10, 10, n)
    grid_y = np.linspace(-10, 10, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            #%%
            #print(x_decoded.shape)
            #%%
            digit = x_decoded[0].reshape(digit_size, digit_size)
            #cv2.imwrite('New_VAE_Database/Tony_Blair/Tony_Blair.'+ str(i)+'.jpeg',255*digit)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 11))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

#%%
    
#n=5
#for i in range(n):
#    digit = x_decoded[i].reshape(64,64) 
#    cv2.waitKey(0)
#    cv2.imwrite('Tony1/Tony_Blair.'+ str(i)+'.jpeg',255*decoded)
#    plt.show()


#%%

#(64)AE_Database
#LFWdataset1
#(64)VAE_Self_DB
#Train_dir = r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\LFW_Database_ind\Tony_Blair'
Train_dir = r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\Database_NON_IR\Selfdatabase'
Img_size = 64 #256

LR = 0.01
def label_img(img):
    word_label = img.split('.')[0]
    if word_label =='Aleric_Pinto':
        return [1,0,0,0,0,0,0,0,0,0]
    elif word_label =='Jason_Malliss':
        return [0,1,0,0,0,0,0,0,0,0]
    elif word_label =='Jay_Patel':
        return [0,0,1,0,0,0,0,0,0,0]
    elif word_label =='Prachi_Tailor':
        return [0,0,0,1,0,0,0,0,0,0]
    elif word_label =='Priya_Thorat':
        return [0,0,0,0,1,0,0,0,0,0]
    elif word_label =='Rajnish_Tailor':
        return [0,0,0,0,0,1,0,0,0,0]
    elif word_label =='Rohit_Thange':
        return [0,0,0,0,0,0,1,0,0,0]
    elif word_label =='Sheetal_Yadav':
        return [0,0,0,0,0,0,0,1,0,0]
    elif word_label =='Shreya_Tembe':
        return [0,0,0,0,0,0,0,0,1,0]
    elif word_label =='Siddharth_Gaud':
        return [0,0,0,0,0,0,0,0,0,1]

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
        #(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        training_data.append([np.array(img),np.array(label)])
        #training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    #np.save('sc_train_data.npy',training_data) 
    return training_data


#%%Load Traing and Testing Data:
train_data = create_training_data()
train = train_data[:-50] 
test = train_data[-100:]
#%%
x_train= np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
y_train = np.array([i[1] for i in train])

x_test = np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
print(x_test.shape)
y_test= np.array([i[1] for i in train])


#%%
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
#%%
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#%%
test1 = []
def convert_to_1d():
    
    #testvalue=[]
    for i in range(2910):
        if (y_test[i] == np.array([1,0,0,0,0,0,0,0,0,0])).all():
            test1.append(0)
        elif (y_test[i] == np.array([0,1,0,0,0,0,0,0,0,0])).all():
            test1.append(1)
        elif (y_test[i] == np.array([0,0,1,0,0,0,0,0,0,0])).all():
            test1.append(2)
        elif (y_test[i] == np.array([0,0,0,1,0,0,0,0,0,0])).all():
            test1.append(3)
        elif (y_test[i] == np.array([0,0,0,0,1,0,0,0,0,0])).all():
            test1.append(4)
        elif (y_test[i] == np.array([0,0,0,0,0,1,0,0,0,0])).all():
            test1.append(5)
        elif (y_test[i] == np.array([0,0,0,0,0,0,1,0,0,0])).all():
            test1.append(6)
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,1,0,0])).all():
            test1.append(7)
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,0,1,0])).all():
            test1.append(8)
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,0,0,1])).all():
            test1.append(9)
            
    return test1
        
test2 = convert_to_1d()



#%%
# network parameters
input_shape = (original_dim, )
intermediate_dim = 3096
batch_size = 128
latent_dim = 2
epochs = 20 #default:50

# VAE model = encoder + decoder
# build encoder model

#%%
inputs = Input(shape=input_shape, name='encoder_input')

#x = inputs
#
#x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(x)
#x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
#
#shape = K.int_shape(x)
#%%
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

#x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
#x = Reshape((shape[1], shape[2], shape[3]))(x)
#
#x = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
#x = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
#
#outputs = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(x)
#

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    #opt = adam(lr=1e-3, decay=1e-9)
    vae.compile(optimizer='adam')
    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae_history = vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                shuffle=True,
                validation_data=(x_test, None))
        #vae.save("vae_mlp_mnist.h5")
        
        

#%%
    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")
    
    #%%
#encoder, decoder = models  
#print(x_test[0].shape)
##%% 
#x_encoded = encoder.predict(x_test[0])
#print(x_encoded.shape)
##%%
#x_decoded = decoder.predict(x_encoded)
#
#print(x_encoded.shape, x_decoded.shape)

#%%
#n = 50
#for i in range(n):
#    decoded = x_decoded[i].reshape(64,64)
#    #cv2.imshow('output',decoded)
#    cv2.waitKey(0)
#    cv2.imwrite('Tony1/Tony_Blair.'+ str(i)+'.jpeg',255*decoded)
#    #fig.savefig('face'+ str(i)+ '.jpg') use this to save the fig when plt is used
#    
#save_img()    
#plt.show()
#    
    
    