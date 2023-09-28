#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:36:01 2023

@author: gd19
"""


from skimage.measure import label 
import os
import nibabel as nib
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import math
from skimage.morphology import dilation, erosion
from scipy.ndimage.morphology import binary_fill_holes
from scipy import ndimage
from numpy import isnan, isinf
from skimage.measure import label 


# cur_dir = '/data/tu_gdoumou/IQT_data/leave_one_out'
# os.chdir(cur_dir)

# files = os.listdir(cur_dir)



# ------------------------------- Synth MPRAGE -----------------------------------------
cur_dir = '/data/Georgia_data/nifti_MPM_anonym/to_reorient/segmented'
# cur_dir = '/data/auto_synth'
os.chdir(cur_dir)

files = os.listdir(cur_dir)

for f in files:
    file_dir = os.path.join(cur_dir,f)
    os.chdir(file_dir)

    img = nib.load('R1_masked_procin.nii')
    img01 = img.get_fdata()
    # img01 = img01[150:300,350:450,150:300] 

    # img01 = img02[200:300,350:400,200:300]
    img_shape = np.shape(img01)

    #img = nib.load('gmc.nii')
    # img01 = img01[150:300,350:450,150:300] 
    # img_shape = np.shape(img01)
    
    img = nib.load(os.path.join(cur_dir,f,'gmc_updated_2.nii'))
    gmc_updated = img.get_fdata()
    # gmc_updated = gmc[150:300,350:450,150:300] 
    img = nib.load(os.path.join(cur_dir,f,'csf_updated_2.nii'))
    csf = img.get_fdata()
    # csf = csf[150:300,350:450,150:300] 
    # gmc_updated = gmc_updated-csf
    # ni_img = nib.Nifti1Image(gmc_updated, img.affine, img.header)
    # nib.save(ni_img,'gmc.nii')

    # img = nib.load('/data/recursive/DIR.nii')
    # DIR = img.get_fdata()
    # DIR = DIR[150:300,350:450,150:300]


    img_shape = np.shape(img01)

    # t1_img = (1/img01)*1000
    # t1_img[np.where(t1_img>7000)]=0
    # ni_img = nib.Nifti1Image(t1_img, img.affine, img.header)
    # nib.save(ni_img,'t1.nii')


    img = nib.load(os.path.join(cur_dir,f,'t1.nii'))
    t1_img = img.get_fdata()


    fermi_pos = np.zeros(img_shape)
    fermi_neg = np.zeros(img_shape)

    for z in range(1,img_shape[2]-1):
        for x in range(1,img_shape[0]-1):
            for y in range(1,img_shape[1]-1):
                 t1 = t1_img[x][y][z]
                 # x0_neg = 1100
                 x0_neg = 1450
                 dx_neg = 500
                 f_neg = 1/(1+math.exp(-(math.log(19)/dx_neg)*(t1-x0_neg)))
                 fermi_neg[x][y][z] = f_neg
                 # x0_pos = 2200
                 x0_pos = 2700
                 dx_pos = 700
                 f_pos = 1/(1+math.exp((math.log(19)/dx_pos)*(t1-x0_pos)))
                 fermi_pos[x][y][z] = f_pos
                              

    DIR = 1100*fermi_pos*fermi_neg
    # plt.imshow(DIR[:,25,:], cmap = "gray")  

    ni_img = nib.Nifti1Image(DIR, img.affine, img.header)
    nib.save(ni_img,'DIR.nii')


    def avg_func(values):
         return values.mean()                    
            

    kernel = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]])


    M1 = ndimage.generic_filter(csf, avg_func, footprint=kernel)

    M1[np.where(csf==1)] = 1

    M2 = 1-M1
    # plt.imshow(M2[:,25,:], cmap = "gray")  

    DIR_final = DIR * M2
    # plt.imshow(DIR_final[:,25,:], cmap = "gray") 


    ni_img = nib.Nifti1Image(DIR_final, img.affine, img.header)
    nib.save(ni_img,'DIR_final.nii')
    os.chdir(cur_dir)