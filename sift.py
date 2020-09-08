# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 12:31:46 2020

@author: Sushilkumar.Yadav
"""

import cv2 
import matplotlib.pyplot as plt

# read images
#img11 = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\junichiro_koizumi.1.jpeg")
#img12 = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\New_VAE_Database\junichiro_koizumi\junichiro_koizumi.0.jpeg")

img11 = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\LFWdataset1\Ariel_Sharon.01.jpeg")
img12 = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\New_VAE_Database\Ariel_Sharon\Ariel_Sharon.49.jpeg")

#%%
img1 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
#%%
#sift
sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)


#%%
img_1 = cv2.drawKeypoints(img1,keypoints_1,img11)
plt.figure()
plt.imshow(img_1)

img_2 = cv2.drawKeypoints(img2,keypoints_2,img12)
plt.figure()
plt.imshow(img_2)

#%%
print(len(keypoints_1), len(keypoints_2))
#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.figure()
plt.imshow(img3),plt.show()