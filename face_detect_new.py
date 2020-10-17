import cv2
import os
import sys
import numpy as np
facedata = sys.argv[0]
def facecrop(image):
    cascPath = r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascPath)
    img = cv2.imread(image)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(miniframe)
    counter = 0
    for f in faces:
        x, y, w, h = [ v for v in f ]
#        pt1 = (int(x), int(y))
#        pt2 = (int(x + w), int(y + h))
#        cv2.rectangle(image, pt1, pt2, (255, 0, 0))
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0))
        sub_face = img[y:y+h, x:x+w]
        saveimg=cv2.resize(sub_face,(64,64))
        #fname, ext = os.path.splitext(image)
        FaceFileName = (r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\Bollywood-dataset\ResizedTestDataset\cropped\actor13." + str(y) + ".jpg")
        cv2.imwrite(FaceFileName, saveimg)
        counter += 1
    return
#%%
for i in range(50):
    facecrop(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\Bollywood-dataset\ResizedTestDataset\ActorT13\AT13i"+str(1+i)+".jpg")
    