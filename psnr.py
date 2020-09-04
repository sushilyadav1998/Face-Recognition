# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 20:24:36 2020

@author: Sushilkumar.Yadav
"""

import numpy 
import math
import cv2
original = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\Junichiro_Koizumi\Junichiro_Koizumi.01.jpeg", cv2.IMREAD_GRAYSCALE)
contrast = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\New_VAE_Database\junichiro_koizumi\junichiro_koizumilat8.0.jpeg", cv2.IMREAD_GRAYSCALE)

#original = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Ariel_Sharon.jpeg")
#contrast = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Ariel_SharonVAE.jpeg",1)
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d=psnr(original,contrast)
print(d)