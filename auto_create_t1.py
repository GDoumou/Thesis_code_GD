
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:42:00 2023

@author: gd19
"""


import nibabel as nib
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndimage
from numpy import isnan, isinf
import statistics
import math
import time
from sklearn.linear_model import LinearRegression
from skimage.morphology import dilation, erosion
from scipy.ndimage.morphology import binary_fill_holes
from scipy import ndimage


# start = time.time()

def find_neigh(matrix, i, j, k):
    neigh = [matrix[i-1][j-1][k], matrix[i-1][j][k], matrix[i-1][j+1][k], 
             matrix[i][j-1][k], matrix[i][j-1][k],
             matrix[i+1][j-1][k],matrix[i+1][j][k],matrix[i+1][j+1][k],
             matrix[i-1][j-1][k+1], matrix[i-1][j][k+1], matrix[i-1][j+1][k+1], 
             matrix[i][j-1][k+1], matrix[i][j-1][k+1],matrix[i][j][k+1],
             matrix[i+1][j-1][k+1],matrix[i+1][j][k+1],matrix[i+1][j+1][k+1],
             matrix[i-1][j-1][k-1], matrix[i-1][j][k-1], matrix[i-1][j+1][k-1], 
             matrix[i][j-1][k-1], matrix[i][j-1][k-1],
             matrix[i+1][j-1][k-1],matrix[i+1][j][k-1],matrix[i+1][j+1][k-1],matrix[i][j][k-1]]
    return neigh

def find_window(matrix, i, j, k):
    neigh = [matrix[i][j][k], matrix[i-1][j-1][k], matrix[i-1][j][k], matrix[i-1][j+1][k], 
             matrix[i][j-1][k], matrix[i][j-1][k],
             matrix[i+1][j-1][k],matrix[i+1][j][k],matrix[i+1][j+1][k],
             matrix[i-1][j-1][k+1], matrix[i-1][j][k+1], matrix[i-1][j+1][k+1], 
             matrix[i][j-1][k+1], matrix[i][j-1][k+1],matrix[i][j][k+1],
             matrix[i+1][j-1][k+1],matrix[i+1][j][k+1],matrix[i+1][j+1][k+1],
             matrix[i-1][j-1][k-1], matrix[i-1][j][k-1], matrix[i-1][j+1][k-1], 
             matrix[i][j-1][k-1], matrix[i][j-1][k-1],
             matrix[i+1][j-1][k-1],matrix[i+1][j][k-1],matrix[i+1][j+1][k-1],matrix[i][j][k-1]]
    return neigh


def find_coords(i,j,k):
    coords = [[i-1,j-1,k], [i-1,j,k], [i-1,j+1,k], 
             [i,j-1,k], [i,j-1,k],
             [i+1,j-1,k],[i+1,j,k],[i+1,j+1,k],
             [i-1,j-1,k+1], [i-1,j,k+1], [i-1,j+1,k+1], 
             [i,j-1,k+1], [i,j-1,k+1],[i,j,k+1],
             [i+1,j-1,k+1],[i+1,j,k+1],[i+1,j+1,k+1],
             [i-1,j-1,k-1], [i-1,j,k-1], [i-1,j+1,k-1], 
             [i,j-1,k-1], [i,j-1,k-1],
             [i+1,j-1,k-1],[i+1,j,k-1],[i+1,j+1,k-1],[i,j,k-1]]
    return coords

def threed_conv_sum(values):
     return np.sum(values)   


def remove_solitary_pixels(input_mask):  
    image_shape = np.shape(input_mask)
    input_mask_ = input_mask       
    kernel = np.ones((3,3,3))
    my_help = ndimage.generic_filter(input_mask_, threed_conv_sum, footprint=kernel)
    input_mask_[np.where(my_help == 1)] = 0
    return input_mask_


# tm_bin = remove_solitary_pixels(tm_bin)
# ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
# nib.save(ni_img,'tm_bin.nii')

# ------------------------------fill holes------------------------------------

def fill_holes(input_mask):  
    image_shape = np.shape(input_mask)
    input_mask_ = input_mask
    kernel = np.ones((3,3,3))
    my_help = ndimage.generic_filter(input_mask_, threed_conv_sum, footprint=kernel)
    input_mask_[np.where((input_mask==0) & (my_help == 26 ))] = 1
    return input_mask_


# tm_bin = fill_holes(tm_bin)
# ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
# nib.save(ni_img,'tm_bin.nii')

cur_dir = '/data/Georgia_data/nifti_MPM_anonym/to_reorient'
# cur_dir = '/data/auto_synth'
os.chdir(cur_dir)

files = os.listdir(cur_dir)

for f in files:
    file_dir = os.path.join(cur_dir,f)
    os.chdir(file_dir)

    # load R1 map
    
    img = nib.load('R1_masked_procin.nii')
    imgR1 = img.get_fdata()
    
    imgR1 = np.where(imgR1<0, 0, imgR1)
    where_are_NaNs = isnan(imgR1)
    imgR1[where_are_NaNs] = 0
    ni_img = nib.Nifti1Image(imgR1, img.affine, img.header)
    nib.save(ni_img,'imgR1.nii')
    
    img = nib.load('imgR1.nii')
    imgR1 = img.get_fdata()
    # imgR1 = imgR1[150:300,350:450,150:300] 
    imgR1 = np.nan_to_num(imgR1, copy=True, posinf=0, neginf=0)
    img_shape = np.shape(imgR1)
    
    tm_bin = np.zeros(img_shape)
    tm_bin[np.where(imgR1 > 0.1)] = 1
    tm_bin = binary_fill_holes(tm_bin)
    ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
    nib.save(ni_img,'tm_bin.nii')
    
    # footprint = np.ones((3,3,3)) 
    # eroded_img = erosion(tm_bin,footprint)
    # eroded_img = erosion(eroded_img,footprint)
    # ni_img = nib.Nifti1Image(eroded_img, img.affine, img.header)
    # nib.save(ni_img,'tm_bin.nii')
    
    tm_bin_ = nib.load('tm_bin.nii')
    tm_bin = tm_bin_.get_fdata()
    
    imgR1 = imgR1 * tm_bin
    ni_img = nib.Nifti1Image(imgR1, img.affine, img.header)
    nib.save(ni_img,'imgR1.nii')
    
    img = nib.load('imgR1.nii')
    imgR1 = img.get_fdata()
    
    # tm_bin = remove_solitary_pixels(tm_bin)
    # ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
    # nib.save(ni_img,'tm_bin.nii')
    
    # tm_bin = fill_holes(tm_bin)
    # ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
    # nib.save(ni_img,'tm_bin.nii')
    
    # convert R1 map to T1 map
    
    img01 = np.divide(1000, imgR1, out=np.zeros_like(imgR1), where=imgR1>0.1)
    img01 = np.nan_to_num(img01, copy=True, posinf=0, neginf=0)
    img01[np.where(img01>5000)]=5000
    # img01 = np.nan_to_num(img01, copy=True, posinf=0, neginf=0)
    where_are_NaNs = isnan(img01)
    img01[where_are_NaNs] = 0
    ni_img = nib.Nifti1Image(img01, img.affine, img.header)
    nib.save(ni_img,'t1.nii')
    os.chdir(cur_dir)