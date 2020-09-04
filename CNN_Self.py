# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:58:42 2020

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
Train_dir = r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\(64)VAE_Self_DB'
Img_size = 64 #256

LR = 0.01
def label_img(img):
    word_label = img.split('.')[0]
    if word_label =='Aleric_Pinto':
        return [1,0,0,0,0,0,0,0,0,0]
    elif word_label =='jason_Malliss':
        return [0,1,0,0,0,0,0,0,0,0]
    elif word_label =='Jay_patel':
        return [0,0,1,0,0,0,0,0,0,0]
    elif word_label =='Prachi_tailor':
        return [0,0,0,1,0,0,0,0,0,0]
    elif word_label =='Priya_Thorat':
        return [0,0,0,0,1,0,0,0,0,0]
    elif word_label =='Rajnish_Tailor':
        return [0,0,0,0,0,1,0,0,0,0]
    elif word_label =='Rohit_thange':
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
        training_data.append([np.array(img),np.array(label)])
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
plt.plot(history.history['loss'], label = 'Loss')
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
import matplotlib.pyplot as plt
fig =plt.figure(figsize=(10,10))
out=[]
for num,data in enumerate(test[:16]):
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(4,4,num+1)
    og = img_data
    img_data = img_data.reshape(1,Img_size,Img_size,1)
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