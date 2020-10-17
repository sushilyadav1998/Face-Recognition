# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:45:28 2020

@author: Sushilkumar.Yadav
"""

# prerequisites
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from scipy.stats import norm
import cv2
from random import shuffle
from tqdm import tqdm
import os
#%%
d1=r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\Bollywood-dataset\bollywood\ACTOR'
Img_size=28
def create_training_data(d1):
    training_data=[]
    for i in range(10):
        Train_dir = d1+str(i+1)+'\\'
        print(Train_dir)
        label=i
        for img in tqdm(os.listdir(Train_dir)):
            path = os.path.join(Train_dir,img)
        #print(path,label)
            img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
            training_data.append([np.array(img),np.array(label)])
        #shuffle(training_data)
    #np.save('sc_train_data.npy',training_data)
    return training_data
X=create_training_data(d1)
shuffle(X)
#%%
x_i = np.array([i[0] for i in X]).reshape(-1, Img_size, Img_size, 1)
y_i = np.array([i[1] for i in X])
x_train=x_i
y_train=y_i
x_test=x_train
y_test=y_train
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#%%
    
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# compute the number of labels
num_labels = len(np.unique(y_train))

# network parameters
input_shape = (image_size, image_size, 1)
label_shape = (num_labels, )

image_size = x_train.shape[1]
#%%
# network parameters
batch_size, n_epoch = 10, 50
n_hidden, z_dim = 256, 2

#%%
# encoder
x = Input(shape=(x_train.shape[1:]))
x_encoded = Dense(n_hidden, activation='relu')(x)
x_encoded = Dense(n_hidden//2, activation='relu')(x_encoded)

mu = Dense(z_dim)(x_encoded)
log_var = Dense(z_dim)(x_encoded)

#%%
# sampling function
def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps

z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])

#%%
# decoder
z_decoder1 = Dense(n_hidden//2, activation='relu')
z_decoder2 = Dense(n_hidden, activation='relu')
y_decoder = Dense(x_train.shape[1], activation='sigmoid')

z_decoded = z_decoder1(z)
z_decoded = z_decoder2(z_decoded)
y = y_decoder(z_decoded)

#%%
# loss
reconstruction_loss = objectives.binary_crossentropy(x, y) * x_train.shape[1]
kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
vae_loss = reconstruction_loss + kl_loss

# build model
vae = Model(x, y)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

#%%
# train
vae.fit(x_train,
       shuffle=True,
       epochs=n_epoch,
       batch_size=batch_size,
       validation_data=(x_test, None), verbose=1)


#%%
# build encoder
encoder = Model(x, mu)
encoder.summary()

#%%
# Plot of the digit classes in the latent space
x_te_latent = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_te_latent[:, 0], x_te_latent[:, 1], c=y_test)
ys = np.unique(y_test)
means = np.array([np.mean(x_te_latent[y_test == y, :], axis=0) for y in ys])
plt.scatter(means[:, 0], means[:, 1], c=ys, s=200, edgecolors='black')
plt.colorbar()
plt.show()

#%%
# build decoder
decoder_input = Input(shape=(z_dim,))
_z_decoded = z_decoder1(decoder_input)
_z_decoded = z_decoder2(_z_decoded)
_y = y_decoder(_z_decoded)
generator = Model(decoder_input, _y)
generator.summary()


#%%
# display a 2D manifold of the digits
n = 15 # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) 
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()