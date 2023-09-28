
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
cur_dir = '/data/Georgia_data/nifti_MPM_anonym/to_sm_ce/now_2'
# cur_dir = '/data/auto_synth'
os.chdir(cur_dir)

files = os.listdir(cur_dir)

for f in files:
    file_dir = os.path.join(cur_dir,f)
    os.chdir(file_dir)

    img = nib.load('R1_masked_procin.nii')
    # img = nib.load('R1_masked_procin.nii')
    img01 = img.get_fdata()
    # img01 = img01[150:300,350:450,150:300] 

    # img01 = img02[200:300,350:400,200:300]
    img_shape = np.shape(img01)

    #img = nib.load('gmc.nii')
    # img01 = img01[150:300,350:450,150:300] 
    # img_shape = np.shape(img01)
    img = nib.load(os.path.join(cur_dir,f,'gmc_updated_2.nii'))
    gmc_updated = img.get_fdata()
    gmc_updated[np.where(gmc_updated>1)] = 1
    # gmc_updated = gmc[150:300,350:450,150:300] 
    img = nib.load(os.path.join(cur_dir,f,'csf_updated_2.nii'))
    csf = img.get_fdata()
    csf[np.where(csf>1)] = 1
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

    DIR_final = nib.load('DIR_final.nii')
    DIR_final = DIR_final.get_fdata()

     

    len_x = 10
    len_y = 10
    len_z = 10
    distance_map = np.zeros([len_x*2+1,len_y*2+1,len_z*2+1]) 
    for k in range(-len_z, len_z+1):
        for i in range(-len_x, len_x+1):
            for j in range(-len_y, len_y+1): 
                d = math.sqrt((-i)**2+(-j)**2+(-k)**2) 
                distance_map[i+len_x][j+len_y][k+len_z]=round(d/0.25)*0.25 
                              
    sd_map = np.zeros(img_shape)
    d_list = []


    for z in range(15,img_shape[2]-10):
        for x in range(15,img_shape[0]-10):
            for y in range(15,img_shape[1]-10):
                px = gmc_updated[x,y,z]
                if px == 1:
                   window = gmc_updated[x-10:x+11,y-10:y+11,z-10:z+11]
                   try:
                       sd = np.min(distance_map[np.nonzero(window-1)])
                   except ValueError: 
                       sd = np.max(distance_map)
                   sd_map[x][y][z]= sd
                   d_list.append([sd,x,y,z])
                    

    ni_img = nib.Nifti1Image(sd_map, img.affine, img.header)
    nib.save(ni_img,'sd_map.nii')

    ce_map = np.zeros(img_shape)
    sorted_d_list = sorted(d_list,reverse = True)
    sorted_d_list_len = len(sorted_d_list)

    for d in range(sorted_d_list_len):
        x = sorted_d_list[d][1]
        y = sorted_d_list[d][2]
        z = sorted_d_list[d][3]
        dist = sorted_d_list[d][0]
        if sd_map[x,y,z]>10:
            dist = 10
        if ce_map[x,y,z]==0 and sd_map[x,y,z]!=0:
            # if sd_map[x,y,z]<=5 and sd_map[x,y,z]!=0:
                # px_d = dist*2
                # ce_map[x,y,z]=px_d
                img = nib.load('sd_map.nii')
                sd_map = img.get_fdata()
                window = sd_map[x-10:x+11,y-10:y+11,z-10:z+11]
                window[np.where(distance_map <= dist)] = dist*2
                window[np.where(distance_map > dist)] = 0
                for k in range(-len_z, len_z+1):
                    for i in range(-len_x, len_x+1):
                        for j in range(-len_y, len_y+1): 
                            if ce_map[x+i][y+j][z+k]==0:
                                ce_map[x+i][y+j][z+k]=window[i+len_x][j+len_y][k+len_z] 
                                     

    # ni_img = nib.Nifti1Image(window, img.affine, img.header)
    # nib.save(ni_img,'window__.nii')

    ni_img = nib.Nifti1Image(ce_map, img.affine, img.header)
    nib.save(ni_img,'ce_map.nii')

    def avg_func(values):
         return values.mean()                    
            

    kernel = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]])
    kernel_5 = np.ones([5,5,5])

    ce_map_conv = ndimage.generic_filter(ce_map, avg_func, footprint=kernel_5)

    ni_img = nib.Nifti1Image(ce_map_conv, img.affine, img.header)
    nib.save(ni_img,'ce_map_conv.nii')


    # img = nib.load('ce_map_conv.nii')
    # ce_map_conv = img.get_fdata()
    # ce_map_conv = ce_map_conv[150:300,350:450,150:300] 

    # # img01 = img02[200:300,350:400,200:300]
    img_shape = np.shape(ce_map_conv)

    img = nib.load('DIR_final.nii')
    DIR_final = img.get_fdata()
    # DIR_final = DIR_final[150:300,350:450,150:300]

    # img = nib.load('ce_map_conv.nii')
    # ce_map_conv = img.get_fdata()

    def avg_func(values):
        avg = values.mean()
        return avg

    def std_func(values):
        std = values.std()
        return std

    kernel_9 = np.ones([9,9,9])


    # kernel_9[4,4] = 0

    ce_map_avP = ndimage.generic_filter(ce_map_conv, avg_func, footprint=kernel_9)

    ni_img = nib.Nifti1Image(ce_map_avP, img.affine, img.header)
    nib.save(ni_img,'ce_map_avP.nii')

    ce_map_stdP = ndimage.generic_filter(ce_map_conv, std_func, footprint=kernel_9)

    ni_img = nib.Nifti1Image(ce_map_stdP, img.affine, img.header)
    nib.save(ni_img,'ce_map_stdP.nii')

    img = nib.load('ce_map_avP.nii')
    ce_map_avP = img.get_fdata()
    # ce_map_avP = ce_map_avP[150:300,350:450,150:300]

    img = nib.load('ce_map_stdP.nii')
    ce_map_stdP = img.get_fdata()
    # ce_map_stdP = ce_map_stdP[150:300,350:450,150:300]

    # median_avP = np.median(ce_map_avP)  
    # median_avP = np.median(ce_map_avP[gmc_updated > 0]) 
    median_avP = np.median(ce_map_avP[img01 > 0]) 

    # median_stdP = np.median(ce_map_stdP)
    # median_stdP = np.median(ce_map_stdP[gmc_updated > 0])
    median_stdP = np.median(ce_map_stdP[img01 > 0]) 

    SL = 3
    Pmin = median_avP + SL*median_stdP

    p_lower_Pmin = ce_map_conv[ce_map_conv > Pmin]

    Pmax = np.percentile(p_lower_Pmin, 90)


    Enh_map_ce =  np.zeros(img_shape)
    AF = 4

    for z in range(img_shape[2]-1):
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                P = ce_map_conv[x][y][z]
                Enhancement  = (P-Pmin)/(Pmax-Pmin)
                if Enhancement>Pmax:
                    Enhancement=Pmax
                if P >= Pmin:
                    Enh_map_ce[x][y][z] = DIR_final[x][y][z] + (AF-1)*DIR_final[x][y][z]*Enhancement
                else:
                    Enh_map_ce[x][y][z] = DIR_final[x][y][z]

                
                #  else: 
                #       Enh_start_map[x][y][z] = P
                # if P<Pmax and P > Pmin:
                #     Enhancement  = (P-Pmin)/(Pmax-Pmin)
                #     Enh_start_map[x][y][z] = Enhancement
                #  else: 
                #       Enh_start_map[x][y][z] = P


    ni_img = nib.Nifti1Image(Enh_map_ce, img.affine, img.header)
    nib.save(ni_img,'Enh_map_ce.nii')
    os.chdir(cur_dir)
   