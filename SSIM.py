# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 20:39:32 2020

@author: Sushilkumar.Yadav
"""

import numpy as np
import numpy
import scipy.ndimage
#from scipy.ndimage import imread
from numpy.ma.core import exp
from scipy.constants.constants import pi
import cv2
img_mat_1 = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\Junichiro_Koizumi\Junichiro_Koizumi.01.jpeg", cv2.IMREAD_GRAYSCALE)
img_mat_2 = cv2.imread(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\New_VAE_Database\junichiro_koizumi\junichiro_koizumilat8.0.jpeg", cv2.IMREAD_GRAYSCALE)


#%%
print(img_mat_1.shape)
print(img_mat_2.shape)

#%%
#img_path_1= r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Ariel_Sharon.jpeg"
#img_path_2=r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Ariel_SharonVAE.jpeg"
#img_mat_1 = img_mat_1[:,:,None]
#img_mat_1 = cv2.imread(img_path_1,cv2.IMREAD_GRAYSCALE)
#img_mat_2 = cv2.imread(img_path_2,cv2.IMREAD_GRAYSCALE)
#%%
#print(img_mat_2.shape)

#%%

#import cv2
#from skimage import measure
#print(measure.compare_ssim(img_mat_2, img_mat_2))
#%%
#img_mat_1 = np.array(img_mat_1)
#img_mat_1 = img_mat_1.flatten()
#print(img_mat_1.shape)
#img_mat_2 = np.array(img_mat_2)
#img_mat_2 = img_mat_2.flatten()
#print(img_mat_2.shape)
#%%
'''
The function to compute SSIM
@param param: img_mat_1 1st 2D matrix
@param param: img_mat_2 2nd 2D matrix
'''
def compute_ssim(img_mat_1, img_mat_2):
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel=numpy.zeros((gaussian_kernel_width,gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))
 
    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1=img_mat_1.astype(numpy.float)
    img_mat_2=img_mat_2.astype(numpy.float)
    
    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
    img_mat_mu_1=img_mat_mu_1.squeeze()
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)
    img_mat_mu_2=img_mat_mu_2.squeeze()
    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
    img_mat_sigma_1_sq=img_mat_sigma_1_sq.squeeze()
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)
    img_mat_sigma_2_sq=img_mat_sigma_2_sq.squeeze()
    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)
    img_mat_sigma_12=img_mat_sigma_12.squeeze()
    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12;
    
    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2
    
    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=numpy.average(ssim_map)
 
    print(index)
 
    return index
 
    
compute_ssim(img_mat_1, img_mat_2)