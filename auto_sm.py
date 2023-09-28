#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:47:47 2023

@author: gd19
"""

# -------------------------------------SM map----------------------------------
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

cur_dir = '/data/external_processed_884/do_ce_sm/do_sm_1'

files = os.listdir(cur_dir)

for f in files:
    file_dir = os.path.join(cur_dir,f)
    os.chdir(file_dir)    


    img = nib.load('t1.nii')
    img01 = img.get_fdata()
    # img01 = img01[150:300,350:450,150:300]
    # ni_img = nib.Nifti1Image(img01, img.affine, img.header)
    # nib.save(ni_img,'3T.nii')
    
    img = nib.load('combined_tm_updated_2.nii')
    combined_tm_updated = img.get_fdata()
    combined_tm_updated[np.where((combined_tm_updated>1) & (combined_tm_updated<2))] = 1
    combined_tm_updated[np.where((combined_tm_updated>=2) & (combined_tm_updated<3))] = 2
    combined_tm_updated[np.where(combined_tm_updated>=3)] = 3
    
    # combined_tm_updated = combined_tm_updated[150:300,350:450,150:300] 
    
    # ni_img = nib.Nifti1Image(combined_tm_updated, img.affine, img.header)
    # nib.save(ni_img,'combined_tm_updated.nii')
    
    img = nib.load('gmc_updated_2.nii')
    gmc_updated = img.get_fdata()
    gmc_updated[np.where(gmc_updated>1)] = 1
    
    img = nib.load('wm_updated_2.nii')
    wm_updated = img.get_fdata()
    wm_updated[np.where(wm_updated>1)] = 1
    
    gm_wm = gmc_updated + wm_updated
    ni_img = nib.Nifti1Image(gm_wm, img.affine, img.header)
    nib.save(ni_img,'gm_wm.nii')
    
    # gmc_updated = gmc_updated[150:300,350:450,150:300]
    
    # img01 = r1 #[150:300,350:450,150:300] 
    img_shape = np.shape(combined_tm_updated)
    # plt.imshow(test[:,25,:],cmap = 'gray')
    
    
    
    # img_tm = nib.load('combined_tm_updated.nii')
    # combined_tm_updated = img_tm.get_fdata()
    # combined_tm_updated = combined_tm_updated[150:300,350:450,150:300]
    
    L1 = np.zeros(img_shape) #4
    L2 = np.zeros(img_shape) #5
    L3 = np.zeros(img_shape) #6
    layers_mask = np.zeros(img_shape)
    
    for z in range(1,img_shape[2]-1):
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                px = combined_tm_updated[x, y, z]
                neighbors = find_neigh(combined_tm_updated, x, y, z)
                if px == 1:
                    if 2 in neighbors:
                        L1[x,y,z] = 4
                        
                elif px == 2:
                    if 1 in neighbors:
                        L2[x,y,z] = 5
                        
                    
    for z in range(1,img_shape[2]-1):
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                px = combined_tm_updated[x, y,z]
                if px == 2:
                    neighbors = find_neigh(L2, x, y, z)
                    if 5 in neighbors:
                        if L2[x,y,z]!=5:
                            L3[x,y,z] = 6
                            
            
    #layers_mask = L1+L2+L3          
    
    #layers_mask[np.where(L2==5)] = 5
    
    #layers_mask[np.where(layers_mask==4.988)] = 5
    
    layers_mask[np.where(L1==4)] = 4
    layers_mask[np.where(L2==5)] = 5
    layers_mask[np.where(L3==6)] = 6
    
    
    ni_img = nib.Nifti1Image(layers_mask, img.affine, img.header)
    nib.save(ni_img,'layers_mask.nii')
    
    
    L2_bin = np.zeros(img_shape)
    L2_bin[np.where(L2>0)] = 1
    
    L1_L2_bin  = np.zeros(img_shape)
    L1_L2_bin[np.where((L2>0) | (L1>0))] = 1
    
    ni_img = nib.Nifti1Image(L2_bin, img.affine, img.header)
    nib.save(ni_img,'L2_mask.nii')
    
    ni_img = nib.Nifti1Image(L1_L2_bin, img.affine, img.header)
    nib.save(ni_img,'L1_L2_mask.nii')
    
    gm_wm_t1s = np.zeros(img_shape)
    
    gm_wm_t1s[np.where(gm_wm!=0)] = img01[np.where(gm_wm!=0)]
    
    ni_img = nib.Nifti1Image(gm_wm_t1s, img.affine, img.header)
    nib.save(ni_img,'gm_wm_t1s.nii')
    
    # layers_mask = nib.load('/data/Georgia_data/redo/FCD_MPM_03/layers_mask.nii')
    # layers_mask = layers_mask.get_fdata()
    
    layers_mask_bin = np.zeros(img_shape)
    layers_mask_bin[np.where(layers_mask>0)]=1
    
    ni_img = nib.Nifti1Image(layers_mask_bin, img.affine, img.header)
    nib.save(ni_img,'layers_mask_bin.nii')
    
    gradient_gm_wm_map = np.zeros(img_shape)
    # gradient_pos_map = np.zeros(tm_shape)
    # gradient_neg_map = np.zeros(tm_shape)
    
    for z in range(1,img_shape[2]-1):
        z_front = z + 1
        z_back = z - 1
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                px = gm_wm_t1s[x, y, z]
                if px!=0:
                #if axial_wm[x,y] = 1 and axial_wm[x,y+1] = 2 or xial_wm[x,y] = 1 and axial_wm[x,y+1] = 2:
                    if img01[x+1,y,z]!=0 and img01[x-1,y,z]!=0:
                        g_pos_x = img01[x+1,y,z]-img01[x,y,z]
                        g_neg_x = img01[x,y,z]-img01[x-1,y,z]
                        g_x = np.mean([g_pos_x,g_neg_x])  
                    else:
                        g_x = 0
                
                    
                    if img01[x,y+1,z]!=0 and img01[x,y-1,z]!=0:
                        g_pos_y = img01[x,y+1,z]-img01[x,y,z]
                        g_neg_y = img01[x,y,z]-img01[x,y-1,z] 
                        g_y = np.mean([g_pos_y,g_neg_y])
                    else:
                        g_y = 0
             
                    if img01[x,y,z_front]!=0 and img01[x,y,z_back]!=0:
                        g_pos_z = img01[x,y,z_front]-img01[x,y,z]
                        g_neg_z = img01[x,y,z]-img01[x,y,z_back]
                        g_z = np.mean([g_pos_z,g_neg_z])
                    else:
                        g_z = 0
    
                    if g_x != 0 and g_x != 0 and g_x != 0:
                        g_hyp = math.sqrt(g_x**2+g_y**2+g_z**2)
                        
                        gradient_gm_wm_map[x,y,z]=g_hyp
                    
    ni_img = nib.Nifti1Image(gradient_gm_wm_map, img.affine, img.header)
    nib.save(ni_img,'gradient_gm_wm_map.nii')  
    
    
    def avg_and_std_func(values):
        avg = values.mean()
        std = values.std()
        P = avg * std
        return P
    
    def avg_func(values):
         return values.mean()   
    
    kernel9 = np.ones([9,9,9])                
            
    
    kernel3 = np.ones([3,3,3])
    
    
    gm_wm_average_std_grad = ndimage.generic_filter(gradient_gm_wm_map, avg_and_std_func, footprint=kernel3)
    
    P_layers = layers_mask_bin * gm_wm_average_std_grad
    
    P_layers_conv = ndimage.generic_filter(P_layers, avg_func, footprint=kernel3)
    
    ni_img = nib.Nifti1Image(P_layers_conv, img.affine, img.header)
    nib.save(ni_img,'P_layers_conv.nii')  
    
    # P_map_conv_avg = ndimage.generic_filter(P_map_conv, avg_func, footprint=kernel)
    
    # layers_mask = nib.load('/data/Georgia_data/redo/FCD_MPM_03/layers_mask.nii')
    # layers_mask = layers_mask.get_fdata()
    
    Pav_L2_img = np.zeros(img_shape)
    Pav_L2 = []
    
    #L2 = np.zeros(img_shape)
    
    for z in range(1,img_shape[2]-1):
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                px = L2_bin[x, y, z]
                if px == 1:
                    Pav_L2.append(P_layers_conv[x,y,z]) 
                    Pav_L2_img[x,y,z] = P_layers_conv[x,y,z]
                    #L2[x,y,z] = 1
    
    ni_img = nib.Nifti1Image(Pav_L2_img, img.affine, img.header)
    nib.save(ni_img,'Pav_L2_img.nii')   
            
    #ni_img = nib.Nifti1Image(L2, img.affine, img.header)
    #nib.save(ni_img,'L2.nii')      
    
    L2_grad = np.multiply(L2_bin, gradient_gm_wm_map)
    ni_img = nib.Nifti1Image(L2_grad, img.affine, img.header)
    nib.save(ni_img,'L2_grad.nii') 
    
    L1_L2_grad = np.multiply(L1_L2_bin, gradient_gm_wm_map)
    ni_img = nib.Nifti1Image(L1_L2_grad, img.affine, img.header)
    nib.save(ni_img,'L1_L2_grad.nii') 
    
    Pav_L2_median = np.median(Pav_L2)
    SM_mask = np.zeros(img_shape)
    for z in range(1,img_shape[2]-1):
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                if Pav_L2_img[x, y, z]!=0:
                # if layers_mask[x, y, z]==5:
                    Pav = Pav_L2_img[x, y, z]
                    SM = Pav_L2_median / Pav
                    if SM>=10:
                        SM=10
                    if SM<=10 and SM>0:
                        SM_mask[x,y, z] = SM
                        
    # img_sm = nib.load('sm_mask.nii')
    # SM_mask = img_sm.get_fdata()
            
    SM_mask_avg = ndimage.generic_filter(SM_mask, avg_func, footprint=kernel3)
    SM_mask_avg[np.where(SM_mask==0)]=0
    
    # img_l2 = nib.load('L2_mask.nii')
    # L2_bin = img_l2.get_fdata()
    # L2_bin[np.where(L2_bin>0)]=1
    # L_SM_mask = np.multiply(SM_mask_avg,L2_bin)
    
    # ni_img = nib.Nifti1Image(L_SM_mask, img.affine, img.header)
    # nib.save(ni_img,'sm_mask_L2.nii')
    
    ni_img = nib.Nifti1Image(SM_mask, img.affine, img.header)
    nib.save(ni_img,'sm_mask.nii')
    
    ni_img = nib.Nifti1Image(SM_mask_avg, img.affine, img.header)
    nib.save(ni_img,'sm_mask_avg.nii')
    
    # ------------------------extend L2--------------------------
    
    # ************change img_ to img for the whole pipeline****************
    
    # img_ = nib.load('gmc_updated.nii')
    # gmc_updated_test = img_.get_fdata()
    # gmc_updated_test = gmc_updated_test[150:300,350:450,150:300]
    
    # img_l2 = nib.load('L2_mask.nii')
    # L2 = img_l2.get_fdata()
    # L2_test = L2[150:300,350:450,150:300]
    
    # img_shape = np.shape(gmc_updated_test)
    
    # img_sm = nib.load('sm_mask_L2.nii')
    # sm_mask_test = img_sm.get_fdata()
    # L_SM_mask_test  = sm_mask_test[150:300,350:450,150:300]
    
    
    len_x = 10
    len_y = 10
    len_z = 10
    distance_map = np.zeros([len_x*2+1,len_y*2+1,len_z*2+1]) 
    for k in range(-len_z, len_z+1):
        for i in range(-len_x, len_x+1):
            for j in range(-len_y, len_y+1): 
                d = math.sqrt((-i*0.5)**2+(-j*0.5)**2+(-k*0.5)**2) 
                distance_map[i+len_x][j+len_y][k+len_z]=round(d/0.25)*0.25 
    
    sm_extended = np.zeros(img_shape)
    
    
    for z in range(15,img_shape[2]-10):
        for x in range(15,img_shape[0]-10):
            for y in range(15,img_shape[1]-10):
                px = gmc_updated[x,y,z]
                if px == 1:
                    window_gmc = gmc_updated[x-10:x+11,y-10:y+11,z-10:z+11]*100
                    window_l2 = L2[x-10:x+11,y-10:y+11,z-10:z+11]*4
                    window_sm = SM_mask_avg[x-10:x+11,y-10:y+11,z-10:z+11]
                    window_2 = window_gmc + window_l2 
                    window_3 = window_2 + distance_map
                    idx_list = []
                    val_list = []
                    for index, val in np.ndenumerate(window_3):
                        if val_list!=0 and val>=21 and val<100:
                            idx_list.append(index)
                            val_list.append(val)
                            min_val = min(val_list)
                            val_idx = val_list.index(min_val)
                            coords = idx_list[val_idx]
                            sm_extended[x,y,z] = window_sm[coords[0]][coords[1]][coords[2]]
                            
                    
                    # for k in range(-len_z, len_z+1):
                    #     for i in range(-len_x, len_x+1):
                    #         for j in range(-len_y, len_y+1): 
                    #             if ce_map[x+i][y+j][z+k]==0:
                    #                 ce_map[x+i][y+j][z+k]=window[i+len_x][j+len_y][k+len_z] 
     
    sm_extended_ = sm_extended+SM_mask_avg
    ni_img = nib.Nifti1Image(sm_extended, img.affine, img.header)
    nib.save(ni_img,'sm_extended.nii')   
    
    # ni_img = nib.Nifti1Image(sm_extended_, img.affine, img.header)
    # nib.save(ni_img,'sm_extended_.nii')  
    
    
    def avg_func(values):
         return values.mean()                    
            
    
    kernel = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]])
    
    
    sm_map_extended_conv = ndimage.generic_filter(sm_extended, avg_func, footprint=kernel)
    
    ni_img = nib.Nifti1Image(sm_map_extended_conv, img.affine, img.header)
    nib.save(ni_img,'sm_map_extended_conv_.nii')            
                
    # -------------------------------------------------------------------
    # L1_L2_mask = L1+L2         
    
    # ni_img = nib.Nifti1Image(L1_L2_mask, img.affine, img.header)
    # nib.save(ni_img,'L1_L2_mask.nii')
    
    # gradient_L1_L2 = np.multiply(gradient_map_3D,L1_L2_mask)
    # ni_img = nib.Nifti1Image(gradient_L1_L2, img.affine, img.header)
    # nib.save(ni_img,'L1_L2_gradient.nii') 
    
    # --------------------------Enh SM map -----------------------------
    sm_map_conv = nib.load('sm_map_extended_conv_.nii')
    sm_map_conv = sm_map_conv.get_fdata()
    
    DIR_final = nib.load('DIR_final.nii')
    DIR_final = DIR_final.get_fdata()
    # DIR_final = DIR_final[150:300,350:450,150:300] 
    
    gmc = nib.load('gmc_updated_2.nii')
    gmc = gmc.get_fdata()
    gmc_updated[np.where(gmc_updated>1)] = 1
    
    
    
    def avg_func(values):
        avg = values.mean()
        return avg
    
    def std_func(values):
        std = values.std()
        return std
    
    kernel_9 = np.ones([9,9,9])
    # kernel_9[4,4] = 0
    
    DIR_final_masked = np.zeros(img_shape)
    
    DIR_final_masked[np.where(gmc!=1)]=0
    
    average_DIR = ndimage.generic_filter(DIR_final_masked, avg_func, footprint=kernel_9)
    
    sm_map_avP = ndimage.generic_filter(sm_map_conv, avg_func, footprint=kernel_9)
    
    sm_map_stdP = ndimage.generic_filter(sm_map_conv, std_func, footprint=kernel_9)
    
    # median_avP = np.median(sm_map_avP)   
    
    # median_stdP = np.median(sm_map_stdP)
    
    
    median_avP = np.median(sm_map_avP[img01 > 0]) 
    
    median_stdP = np.median(sm_map_stdP[img01 > 0])
    
    
    SL = 3
    Pmin = median_avP + SL*median_stdP
    
    p_lower_Pmin = sm_map_conv[sm_map_conv > Pmin]
    
    Pmax = np.percentile(p_lower_Pmin, 90)
    
    AF = 4
    Enh_map_sm =  np.zeros(img_shape)
    
    for z in range(img_shape[2]-1):
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                P = sm_map_conv[x][y][z]
                Enhancement  = (P-Pmin)/(Pmax-Pmin)
                if Enhancement>Pmax:
                    Enhancement=Pmax
                if P >= Pmin:
                    Enh_map_sm[x][y][z] = DIR_final[x][y][z] + (AF-1)*DIR_final[x][y][z]*Enhancement
                else:
                    Enh_map_sm[x][y][z] = DIR_final[x][y][z]
    
    
    ni_img = nib.Nifti1Image(Enh_map_sm, img.affine, img.header)
    nib.save(ni_img,'Enh_map_sm_optim.nii')
    
    # Enh_image_sm = DIR_final + (AF-1)*DIR_final*Enh_start_map_sm
    
    # # plt.imshow(Enh_image[:,120,:],cmap = "gray")
    
    # ni_img = nib.Nifti1Image(Enh_image_sm, img.affine, img.header)
    # nib.save(ni_img,'Enh_image_sm.nii')
    os.chdir(cur_dir)