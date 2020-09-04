# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:10:55 2020

@author: Sushilkumar.Yadav
"""

import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt


#from keras.models import Sequential 

#%%
#(64)AE_Database
#LFWdataset1
#(64)VAE_Self_DB
Train_dir = r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\LFWdataset1'
Img_size = 64 #256

LR = 0.01
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
        #print(path,label)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
        
    shuffle(training_data)
    #np.save('sc_train_data.npy',training_data)
    return training_data


#%%Load Traing and Testing Data:
train_data = create_training_data()
train = train_data[:-50] 
test = train_data[-50:]
#%%
X_train= np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
Y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test]).reshape(-1, Img_size, Img_size, 1)
Y_test= np.array([i[1] for i in test])

#%%
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import models
#from keras import UpSampling2D, ZeroPadding2D, Input
batch_size = 5       
hidden_neurons = 1000
classes = 11    
epochs = 10

#%%
model = Sequential() 
model.add(Convolution2D(4, (3, 3), input_shape=(Img_size, Img_size, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
#model.add(Convolution2D(4, (3, 3)))  
#model.add(Activation('relu'))
#model.add(Convolution2D(4, (3, 3)))  
#model.add(Activation('relu'))
#model.add(Convolution2D(4, (3, 3))) 
#model.add(Activation('relu'))     
#model.add(Convolution2D(4, (3, 3)))     
#model.add(Activation('relu'))  
#model.add(Convolution2D(4, (3, 3)))     
#model.add(Activation('relu'))   
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))     

     
#model.add(MaxPooling2D(pool_size=(2, 2)))     
#model.add(Dropout(0.25))
               
model.add(Flatten())
 
model.add(Dense(hidden_neurons)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5))      
model.add(Dense(classes)) 
model.add(Activation('softmax'))
     
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.1, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1]*100) 

 #%%
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0005, 1.5])
plt.legend(loc='lower right')

#%%
from tensorflow.python.keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
model.summary()

#%%
#os.environ["PATH"] += os.pathsep + 'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\utils_libraries\Graphviz\bin\'
#plot_model(model, to_file='CNNModel.png')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


#%%

img_path = r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\LFWdataset1\George_Bush.26.jpeg'      
img = cv2.resize(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.
print(img.shape)

#%%

layer_outputs = [layer.output for layer in model.layers[:20]] # Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

#%%

activations = activation_model.predict(img) # Returns a list of five Numpy arrays: one array per layer activation
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 1], cmap='viridis')

#%%
layer_names = []
for layer in model.layers[:20]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 4
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
#%%
import tensorflow as tf
import matplotlib.pyplot as plt
fig =plt.figure(figsize=(10,10))
out=[]
for num,data in enumerate(test[:16]):
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(4,4,num+1)
    og = img_data
    img_data = img_data.reshape(1,Img_size,Img_size,1)
    img_data = tf.cast(img_data, tf.float32)
    model_output = model.predict([img_data])
    #print(model_output)
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
    
    if np.argmax(img_num) == 0: act_label = 'Ariel_Sharon'
    elif np.argmax(img_num) == 1: act_label = 'Colin_Powell'
    elif np.argmax(img_num) == 2: act_label = 'George_Bush'
    elif np.argmax(img_num) == 3: act_label = 'Gerhard_Schroede'
    elif np.argmax(img_num) == 4: act_label = 'Hugo_Chavez'
    elif np.argmax(img_num) == 5: act_label = 'Jacques_Chirac'
    elif np.argmax(img_num) == 6: act_label = 'Jean'
    elif np.argmax(img_num) == 7: act_label = 'John_Ashcroft'
    elif np.argmax(img_num) == 8: act_label = 'Junichiro_Koimuzi'
    elif np.argmax(img_num) == 9: act_label = 'Serena_Williams'
    elif np.argmax(img_num) == 10: act_label = 'Tony_Blair'
    else: str_label = 'cat'
    out.append(act_label)
    
    y = plt.imshow(og,cmap='gray')
    plt.title(str_label)
    plt.xlabel(act_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
print(out)