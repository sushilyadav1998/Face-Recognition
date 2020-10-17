import cv2
import sys
from tqdm import tqdm
import os
import numpy as np

# Get user supplied values
imagePath = sys.argv[0]
cascPath = r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

#%%
# Read the image
Img_size = 64
Train_dir=r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\Bollywood-dataset\ResizedTrainDataset\Actor1'
def image_load():
    imagedata = []
    for img in tqdm(os.listdir(Train_dir)):
        path = os.path.join(Train_dir,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(64,64))
        imagedata.append([np.array(img)])
    return imagedata

image_sample = image_load()
img_data = image_sample
image= np.array([i[0] for i in img_data]).reshape(-1, Img_size, Img_size, 1)
    #image = cv2.imread(img)
    #%%
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#%%
#image = cv2.resize(image1, (1080, 1920)) 
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#%%
# Detect faces in the image

faces = faceCascade.detectMultiScale(image, 1.3, 5)
#faces = faceCascade.load(
#    image,
#    scaleFactor=1.1,
#    minNeighbors=4,
#    minSize=(100, 100),
#    flags = 0
#)
print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    pt1 = (int(x), int(y))
    pt2 = (int(x + w), int(y + h))
    cv2.Rectangle(image, pt1, pt2, cv2.RGB(255, 0, 0), 5, 8, 0)
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 10), 2)
    saveimg1 = image[y:y+h, x:x+w]
    saveimg=cv2.resize(saveimg1,(64,64))
    FaceFileName = ("Bollywood-dataset\ResizedTrainDataset\cropped/actor1." + str(y) + ".jpg")
    cv2.imwrite(FaceFileName, saveimg)
    
#cv2.imshow("Faces found", image)

