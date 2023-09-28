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
cur_dir = '/data/Georgia_data/nifti_MPM_anonym/to_reorient/now'
# cur_dir = '/data/auto_synth'
os.chdir(cur_dir)

files = os.listdir(cur_dir)

for f in files:
    file_dir = os.path.join(cur_dir,f)
    os.chdir(file_dir)
    gm_ = nib.load('c1R1.nii')
    gm = gm_.get_fdata()
    wm = nib.load('c2R1.nii')
    wm = wm.get_fdata()
    csf = nib.load('c3R1.nii')
    csf = csf.get_fdata()
    # gm_ = nib.load('c1synth_mprage.nii')
    # gm = gm_.get_fdata()
    # wm = nib.load('c2synth_mprage.nii')
    # wm = wm.get_fdata()
    # csf = nib.load('c3synth_mprage.nii')
    # csf = csf.get_fdata()
    img_shape = np.shape(gm_)
    
    # mask = np.zeros(img_shape)
    # mask[np.where((gm>=0.70)|(wm>=0.70)|(csf>=0.70))]=1   
    mask = gm+wm+csf
    where_are_NaNs = isnan(mask)
    mask[where_are_NaNs] = 0
    imask = np.nan_to_num(mask, copy=True, posinf=0, neginf=0)
    
    mask_ = np.zeros(img_shape)
    mask_[np.where(mask > 0)] = 1
    mask_ = binary_fill_holes(mask_)
    # ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
    # nib.save(ni_img,'tm_bin.nii')
    
    #find largest blob (remove background)
    labels = label(mask_)
    if labels.max() != 0: # assume at least 1 CC
        largest_label = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    # smooth
    def avg_func(values):
         return values.mean()           
            
    kernel3 = np.ones([3,3,3])

    smask = ndimage.generic_filter(largest_label, avg_func, footprint=kernel3)

    # binarize above 0.5 smoothed mask
    smask_bin = np.zeros(img_shape)
    smask_bin[np.where(smask>=0.4)]=1
    
    # erode twice binarized smoothed mask

    # footprint = np.ones((3,3,3)) 
    # eroded_img = erosion(smask_bin,footprint)
    # eroded_img = erosion(eroded_img,footprint)

            
    ni_img = nib.Nifti1Image(smask_bin, gm_.affine, gm_.header)
    nib.save(ni_img,'smask.nii')
    os.chdir(cur_dir)
    
    
# -----------------------------auto_mask-------------------------------

# cur_dir = '/data/Georgia_data/HC/nifti'
# # cur_dir = '/data/auto_synth'
# os.chdir(cur_dir)

# files = os.listdir(cur_dir)

# for f in files:
#     file_dir = os.path.join(cur_dir,f)
#     os.chdir(file_dir)
#     smask_ = nib.load('resliced_smask.nii')
#     smask = smask_.get_fdata()
#     img_shape = np.shape(smask)
#     sub = os.listdir(file_dir)
#     for s in sub:
#         if s.startswith('resliced_') is True and s!='resliced_smask.nii':
#             R1_ = nib.load(s)
#             R1 = R1_.get_fdata()
#             R1_masked = R1 * smask
            
#             ni_img = nib.Nifti1Image(R1_masked, smask_.affine, smask_.header)
#             nib.save(ni_img,'R1_3T_procin.nii')
#     os.chdir(cur_dir)
    
    
cur_dir = '/data/Georgia_data/nifti_MPM_anonym/to_reorient'
# cur_dir = '/data/auto_synth'
os.chdir(cur_dir)

files = os.listdir(cur_dir)

for f in files:
    file_dir = os.path.join(cur_dir,f)
    os.chdir(file_dir)
    smask_ = nib.load('s_resliced_smask.nii')
    smask = smask_.get_fdata()
    img_shape = np.shape(smask)
    smask_bin = np.zeros(img_shape)
    smask_bin[np.where(smask>=0.4)]=1
    sub = os.listdir(file_dir)
    for s in sub:
        if s.startswith('resliced_') is True and s.endswith('R1.nii') and s!='resliced_smask.nii':
            R1_ = nib.load(s)
            R1 = R1_.get_fdata()
            R1_masked = R1 * smask_bin
            
            ni_img = nib.Nifti1Image(R1_masked, smask_.affine, smask_.header)
            nib.save(ni_img,'R1_masked_procin.nii')
    os.chdir(cur_dir)
    
# ---------------------------------create_folders-----------------------------    

cur_dir = '/data/Georgia_data/nifti_MPM_anonym/to_reconstruct_3
# cur_dir = '/data/auto_synth'
os.chdir(cur_dir)


lst = list(range(1,51))

for items in range(len(lst)):
    name = 'subject_'+ str(lst[items])
    os.mkdir(name)
    

import shutil
    
cur_dir = '/data/Georgia_data/nifti_MPM_anonym/to_reorient'
# cur_dir = '/data/auto_synth'
os.chdir(cur_dir)

files = os.listdir(cur_dir)

for item in files:
    file_name = item.split('-')[0]
    os.mkdir(os.path.join(cur_dir , file_name))
    shutil.copy(os.path.join(cur_dir , item), os.path.join(cur_dir , file_name))
    
    
    
import shutil
    
cur_dir = '/data/Georgia_data/nifti_MPM_anonym/to_reorient'
# cur_dir = '/data/auto_synth'
os.chdir(cur_dir)

files = os.listdir(cur_dir)

for item in files:
    os.chdir(os.path.join(cur_dir,item))
    file_name = 'R1_masked_procin.nii'
    os.mkdir(os.path.join('/data/Georgia_data/nifti_MPM_anonym/to_reconstruct_3' , item))
    shutil.copy(os.path.join(cur_dir,item, file_name), os.path.join('/data/Georgia_data/nifti_MPM_anonym/to_reconstruct_3',item,file_name))
    
    