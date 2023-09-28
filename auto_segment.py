#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:29:48 2023

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



cur_dir = '/data/Georgia_data/HC/nifti'
# cur_dir = '/data/auto_synth'
os.chdir(cur_dir)

files = os.listdir(cur_dir)

for f in files:
    file_dir = os.path.join(cur_dir,f)
    os.chdir(file_dir)
    # img = nib.load('R1_masked_procin.nii')
    # imgR1 = img.get_fdata()

    # imgR1 = np.where(imgR1<0, 0, imgR1)
    # where_are_NaNs = isnan(imgR1)
    # imgR1[where_are_NaNs] = 0
    # ni_img = nib.Nifti1Image(imgR1, img.affine, img.header)
    # nib.save(ni_img,'imgR1.nii')

    # img = nib.load('imgR1.nii')
    # imgR1 = img.get_fdata()
    # # imgR1 = imgR1[150:300,350:450,150:300] 
    # imgR1 = np.nan_to_num(imgR1, copy=True, posinf=0, neginf=0)
    # img_shape = np.shape(imgR1)

    # tm_bin = np.zeros(img_shape)
    # tm_bin[np.where(imgR1 > 0.1)] = 1
    # tm_bin = binary_fill_holes(tm_bin)
    # ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
    # nib.save(ni_img,'tm_bin.nii')

    # # footprint = np.ones((3,3,3)) 
    # # eroded_img = erosion(tm_bin,footprint)
    # # eroded_img = erosion(eroded_img,footprint)
    # # ni_img = nib.Nifti1Image(eroded_img, img.affine, img.header)
    # # nib.save(ni_img,'tm_bin.nii')

    # tm_bin = eroded_img

    # imgR1 = imgR1 * tm_bin
    # ni_img = nib.Nifti1Image(imgR1, img.affine, img.header)
    # nib.save(ni_img,'imgR1.nii')

    # img = nib.load('imgR1.nii')
    # imgR1 = img.get_fdata()

    # tm_bin = remove_solitary_pixels(tm_bin)
    # ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
    # nib.save(ni_img,'tm_bin.nii')

    # tm_bin = fill_holes(tm_bin)
    # ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
    # nib.save(ni_img,'tm_bin.nii')

    # # convert R1 map to T1 map

    # img01 = np.divide(1000, imgR1, out=np.zeros_like(imgR1), where=imgR1>0.1)
    # img01 = np.nan_to_num(img01, copy=True, posinf=0, neginf=0)
    # img01[np.where(img01>5000)]=5000
    # # img01 = np.nan_to_num(img01, copy=True, posinf=0, neginf=0)
    # where_are_NaNs = isnan(img01)
    # img01[where_are_NaNs] = 0
    # ni_img = nib.Nifti1Image(img01, img.affine, img.header)
    # nib.save(ni_img,'t1.nii')
    
    







    # set gm/wm parameters (remeber to: check Ralf's code re percentages and how')

    img = nib.load('t1.nii')
    t1 = img.get_fdata()
    img_shape = np.shape(t1)

    img = nib.load('sresliced_smask.nii')
    smask = img.get_fdata()
    
    tm_bin = np.zeros(img_shape)
    tm_bin[np.where(smask>=0.4)]=1
    
    ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
    nib.save(ni_img,'tm_bin.nii')
    
    img = nib.load('tm_bin.nii')
    tm_bin = img.get_fdata()


    # t1_wm_min = 800
    # t1_wm_max = 1200
    # t1_gm_min =1350
    # t1_gm_max = 1750

    # delta_wm = 100
    # delta_csf = 100

    t1_wm_min = 700
    t1_wm_max = 1100
    t1_gm_min =1350
    t1_gm_max = 1750

    delta_wm = 100
    delta_csf = 100

    # t1_wm_min = 0.85
    # t1_wm_max = 1.54
    # t1_gm_min = 0.63
    # t1_gm_max = 0.75

    # T1 min(WM) = 650 ms, T1max(WM) = 1050 ms, T1 min(GM) = 1200 ms, T1max(GM) = 1600 ms 


    # tm_bin = img01
    # tm_bin[np.where(tm_bin > 0)] = 1
    # ni_img = nib.Nifti1Image(tm_bin, img.affine, img.header)
    # nib.save(ni_img,'tm_bin.nii')




    # -------------------------get edges function---------------------------------

    def get_edges(input_mask):
        image_shape = np.shape(input_mask)
        bin_mask = np.zeros(image_shape)
        bin_mask[np.where(input_mask > 0)] = 1
        edge_1 = np.zeros(image_shape)
        edge_2 = np.zeros(image_shape)
        ks = 3
        kernel = np.ones((ks,ks,ks))
        my_help = ndimage.generic_filter(bin_mask, threed_conv_sum, footprint=kernel)
        edge_1[np.where((bin_mask==1) & (my_help<=(ks*3-1)))]=1
        edge_2[np.where((bin_mask==0) & (my_help>=1))]=1
        return [edge_1,edge_2]

    # [ie,oe] = get_edges(tm_bin)
    # ni_img = nib.Nifti1Image(ie, img.affine, img.header)
    # nib.save(ni_img,'ie.nii')
    # ni_img = nib.Nifti1Image(oe, img.affine, img.header)
    # nib.save(ni_img,'oe.nii')


    # -------------------------------------------------------------------------------

    wm_start = np.zeros(img_shape)
    wm_start[np.where((tm_bin>0) & (t1>0) & (t1<t1_wm_max))]=1
    ni_img = nib.Nifti1Image(wm_start, img.affine, img.header)
    nib.save(ni_img,'wm_start.nii')

    tm_median = statistics.median(t1[np.where(wm_start==1)])
    tm_std = statistics.stdev(t1[np.where(wm_start==1)])

    mask2_min = np.zeros(img_shape)
    mask2_min[np.where(t1>(tm_median + 4 * tm_std))]=1
    mask2_min = binary_fill_holes(mask2_min)
    # mask2_min = remove_solitary_pixels(mask2_min)
    ni_img = nib.Nifti1Image(mask2_min, img.affine, img.header)
    nib.save(ni_img,'mask2_min.nii')

    mask2_max = np.zeros(img_shape)
    mask2_max[np.where(t1>(tm_median + 2 * tm_std))]=1
    mask2_max = binary_fill_holes(mask2_max)
    # mask2_max = remove_solitary_pixels(mask2_max)
    ni_img = nib.Nifti1Image(mask2_max, img.affine, img.header)
    nib.save(ni_img,'mask2_max.nii')

    # --------------------------with dilation--------------------------------------

    # temp = nib.load('mask2_min.nii')
    # temp = temp.get_fdata()

    # footprint = np.ones((3,3,3)) 

    # dilated_img = dilation(temp,footprint)
    # # dilated_img = dilation(gmc_temp)
    # found_pixels = np.zeros(img_shape)
    # found_pixels[np.where((dilated_img==1) & (mask2_max==1))] = 1

    # ni_img = nib.Nifti1Image(found_pixels, img.affine, img.header)
    # nib.save(ni_img,'found_pixels.nii') 

    # def threed_conv_sum(values):
    #      return np.sum(values)                   

    # kernel = np.ones((3,3,3))

    # my_help = ndimage.generic_filter(found_pixels, threed_conv_sum, footprint=kernel)

    # my_help[np.where(my_help==1)]=0
    # my_help[np.where(my_help>1)]=1

    # to_be_added = found_pixels*my_help

    # ni_img = nib.Nifti1Image(to_be_added, img.affine, img.header)
    # nib.save(ni_img,'to_be_added.nii') 

    # my_num = np.sum(to_be_added)

    # while my_num>0:
    #     temp[np.where(to_be_added==1)]=1
    #     mask2_max[np.where(found_pixels==1)]=0
    #     dilated_img = dilation(temp,footprint)
    #     found_pixels = np.zeros(img_shape)
    #     found_pixels[np.where((dilated_img==1) & (mask2_max==1))] = 1

    #     my_help = ndimage.generic_filter(found_pixels, threed_conv_sum, footprint=kernel)

    #     my_help[np.where(my_help==1)]=0
    #     my_help[np.where(my_help>1)]=1
    #     to_be_added = my_help
    #     my_num = np.sum(to_be_added)
        
    # ni_img = nib.Nifti1Image(temp, img.affine, img.header)
    # nib.save(ni_img,'mask2_gd_corrected.nii') 


    # ----------------------------- with get_edges----------------------------------------


    temp = nib.load('mask2_min.nii')
    temp = temp.get_fdata()

    still_available = np.zeros(img_shape)
    still_available[np.where((mask2_max>0) & (mask2_min==0))]=1

    [myhelp,outer_edge] = get_edges(temp)

    found_pixels = np.zeros(img_shape)
    found_pixels[np.where((outer_edge==1) & (still_available==1))] = 1

    to_be_added=np.zeros(img_shape)
    to_be_added[np.where(found_pixels==1)] = 1

    my_num = np.sum(to_be_added)

    while my_num>0:
        temp[np.where(to_be_added==1)]=1
        mask2_max[np.where(found_pixels==1)]=0
        
        [myhelp,outer_edge] = get_edges(temp)

        found_pixels = np.zeros(img_shape)
        found_pixels[np.where((outer_edge==1) & (still_available==1))] = 1
        
        to_be_added=np.zeros(img_shape)
        to_be_added[np.where(found_pixels==1)] = 1
        
        my_num = np.sum(to_be_added)
        
    ni_img = nib.Nifti1Image(temp, img.affine, img.header)
    nib.save(ni_img,'mask2_rlf.nii') 


    # -------------------------------------------------------------------------------

    # temp = nib.load('mask2_min.nii')
    # temp = temp.get_fdata()

    # mask2_max = nib.load('mask2_max.nii')
    # mask2_max = mask2_max.get_fdata()

    # still_available = np.zeros(img_shape)
    # still_available[np.where((mask2_max>0) & (mask2_min==0))]=1

    # # [myhelp,outer_edge] = get_edges(temp)

    # footprint = np.ones((3,3,3)) 

    # outer_edge = dilation(temp,footprint)

    # found_pixels = np.zeros(img_shape)
    # found_pixels[np.where((outer_edge==1) & (still_available==1))] = 1

    # to_be_added=np.zeros(img_shape)
    # to_be_added[np.where(found_pixels==1)] = 1

    # my_num = np.sum(to_be_added)

    # while my_num>0:
    #     temp[np.where(to_be_added==1)]=1
    #     mask2_max[np.where(found_pixels==1)]=0
        
    #     outer_edge = dilation(temp,footprint)
      
    #     found_pixels = np.zeros(img_shape)
    #     found_pixels[np.where((outer_edge==1) & (still_available==1))] = 1
        
    #     to_be_added=np.zeros(img_shape)
    #     to_be_added[np.where(found_pixels==1)] = 1
        
    #     my_num = np.sum(to_be_added)
        
    # ni_img = nib.Nifti1Image(temp, img.affine, img.header)
    # nib.save(ni_img,'mask2_gd_corrected.nii') 


    # ------------------------------------------------------------------------------

    gmc = nib.load('mask2_rlf.nii')
    gmc = gmc.get_fdata()

    # wm = nib.load('wm_start.nii')
    # wm = wm.get_fdata()

    wm = np.zeros(img_shape)

    csf = np.zeros(img_shape)

    combined_tm = np.zeros(img_shape)

    wm[np.where((tm_bin==1) & (gmc==1))] = 0
    wm[np.where((tm_bin==1) & (gmc==0))] = 1


    csf[np.where((gmc==1) & (t1>1.5*t1_gm_max))] = 1
    gmc[np.where((gmc==1) & (t1>1.5*t1_gm_max))] = 0


    combined_tm[np.where(gmc==1)] = 1
    combined_tm[np.where(csf==1)] = 3
    combined_tm[np.where(wm==1)] = 2

    for z in range(1,img_shape[2]-1):
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                px = tm_bin[x, y, z]
                if tm_bin[x, y, z] != 0:
                    neigh = find_neigh(tm_bin, x, y, z)
                    if 0 in neigh:
                        csf[x,y,z] = 1

    ni_img = nib.Nifti1Image(gmc, img.affine, img.header)
    nib.save(ni_img,'gmc.nii')   

    ni_img = nib.Nifti1Image(csf, img.affine, img.header)
    nib.save(ni_img,'csf.nii')  

    ni_img = nib.Nifti1Image(wm, img.affine, img.header)
    nib.save(ni_img,'wm.nii')   

    ni_img = nib.Nifti1Image(combined_tm, img.affine, img.header)
    nib.save(ni_img,'combined_tm.nii')  

    # -------------------------------------------------------------------------------

    img = nib.load('t1.nii')
    img01 = img.get_fdata()

    img_shape = np.shape(img01)

    gradient_map_3D = np.zeros(img_shape)
    gradient_x_map_3D = np.zeros(img_shape)
    gradient_y_map_3D = np.zeros(img_shape)
    gradient_z_map_3D = np.zeros(img_shape)
    orientation_map_3D = np.zeros(img_shape)
    rgb_map_3D = np.zeros(img_shape)
    rgb_map = np.stack((rgb_map_3D,)*3,axis=-1)

    for z in range(1,img_shape[2]-1):
        z_front = z + 1
        z_back = z - 1
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                if img01[x,y,z]!=0:
                    if img01[x+1,y,z]!=0 and img01[x-1,y,z]!=0:
                        g_pos_x = img01[x+1,y,z]-img01[x,y,z]
                        g_neg_x = img01[x,y,z]-img01[x-1,y,z]
                        g_x = np.mean([g_pos_x,g_neg_x])  
                    # elif img01[x+1,y,z]==0 or img01[x-1,y,z]==0:
                    #     g_x = 0
                    #     g_pos_x = img01[x+1,y,z]-img01[x,y,z]
                    #     g_x = g_pos_x
                    # elif img01[x+1,y,z]!=0 and img01[x-1,y,z]==0:
                    #     g_neg_x = img01[x,y,z]-img01[x-1,y,z]
                    #     g_x = g_neg_x
                    else:
                        g_x = 0
                
                    
                    if img01[x,y+1,z]!=0 and img01[x,y-1,z]!=0:
                        g_pos_y = img01[x,y+1,z]-img01[x,y,z]
                        g_neg_y = img01[x,y,z]-img01[x,y-1,z] 
                        g_y = np.mean([g_pos_y,g_neg_y])
                    # elif img01[x,y+1,z]==0 or img01[x,y-1,z]==0:
                    #     g_y = 0
                    #     g_pos_y = img01[x,y+1,z]-img01[x,y,z]
                    #     g_y = g_pos_y
                    # elif img01[x,y+1,z]==0 and img01[x,y-1,z]!=0:
                    #     g_neg_y = img01[x,y,z]-img01[x,y-1,z]
                    #     g_y = g_neg_y
                    else:
                        g_y = 0
             
                    if img01[x,y,z_front]!=0 and img01[x,y,z_back]!=0:
                        g_pos_z = img01[x,y,z_front]-img01[x,y,z]
                        g_neg_z = img01[x,y,z]-img01[x,y,z_back]
                        g_z = np.mean([g_pos_z,g_neg_z])
                    # elif img01[x,y,z_front]==0 or img01[x,y,z_back]==0:
                    #     g_z = 0
                    #     g_pos_z = img01[x,y,z_front]-img01[x,y,z]
                    #     g_z = g_pos_z
                    # elif img01[x,y,z_front]==0 and img01[x,y,z_back]!=0:
                    #     g_neg_z = img01[x,y,z]-img01[x,y,z_back]
                    #     g_z = g_neg_z
                    else:
                        g_z = 0
                  

                    if g_x != 0 and g_x != 0 and g_x != 0:
                        g_hyp = math.sqrt(g_x**2+g_y**2+g_z**2)
                            # if g_y!=0 or g_x!=0:
                            #     theta = math.atan(g_y/g_x)
                            #     orientation_map_3D[x,y,z]=theta
                            # else:
                            #     orientation_map_3D[x,y,z]=0
                        gradient_map_3D[x,y,z]=g_hyp
                        gradient_x_map_3D[x,y,z]=g_x
                        gradient_y_map_3D[x,y,z]=g_y
                        gradient_z_map_3D[x,y,z]=g_z
                
                # to RGB:
                # red = g_hyp*math.sin(theta)
                # green = g_hyp*math.cos(theta)
                # blue = 0
                # rgb_map[x,y,z,0] = red
                # rgb_map[x,y,z,1] = green
                # rgb_map[x,y,z,2] = blue
                

    # gradient_map_mag = np.abs(gradient_map)
    # gradient_map_mag = np.abs(gradient_x_map)
            
    # plt.figure()
    # plt.imshow(gradient_x_map_3D[:,350,:])
    # plt.imshow(gradient_y_map_3D[:,350,:])
    # plt.imshow(gradient_map_3D[:,350,:],cmap = 'gray')
    # plt.imshow(gradient_map_3D[:,:,85], cmap = 'gray')
    # # plt.imshow(gradient_map_mag)

    ni_img = nib.Nifti1Image(gradient_map_3D, img.affine, img.header)
    nib.save(ni_img,'g_3D.nii')

    # ni_img = nib.Nifti1Image(orientation_map_3D, img.affine, img.header)
    # nib.save(ni_img,'o_3D.nii')
    # ni_img = nib.Nifti1Image(rgb_map, img.affine, img.header)
    # nib.save(ni_img,'rgb_3D.nii')

    def avg_func(values):
          return values.mean()

    kernel = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]])


    gradient_map_conv = ndimage.generic_filter(gradient_map_3D, avg_func, footprint=kernel)
    gx_conv = ndimage.generic_filter(gradient_x_map_3D, avg_func, footprint=kernel)
    gy_conv = ndimage.generic_filter(gradient_y_map_3D, avg_func, footprint=kernel)
    gz_conv = ndimage.generic_filter(gradient_z_map_3D, avg_func, footprint=kernel)

    gradient_map_3D_conv_first = np.zeros(img_shape)

    for z in range(1,img_shape[2]-1):
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                g_hyp = np.hypot(math.sqrt(gx_conv[x,y,z]**2+gy_conv[x,y,z]**2),gz_conv[x,y,z])
                gradient_map_3D_conv_first[x,y,z]=g_hyp

    # gradient_map_mag_conv = ndimage.generic_filter(gradient_map_mag, avg_func, footprint=kernel)


    ep_map = 1-(gradient_map_3D_conv_first/gradient_map_conv)

    where_are_NaNs = isnan(ep_map)
    ep_map[where_are_NaNs] = 0

    ni_img = nib.Nifti1Image(ep_map, img.affine, img.header)
    nib.save(ni_img,'ep_map.nii')



    # apply thresholds to EP_map
    def ep_thr_func(values):
        min_val = values.min()
        max_val = values.max()
        avg_val = (min_val+max_val)/2
        return avg_val
        
    ep_map_conv = ndimage.generic_filter(ep_map, ep_thr_func, footprint=kernel) 

    ep_map_thr = np.zeros(img_shape)
       
    for z in range(1,img_shape[2]-1):
        for x in range(img_shape[0]-1):
            for y in range(img_shape[1]-1):
                if ep_map[x,y,z]>0.25:
                    if ep_map[x,y,z]>ep_map_conv[x,y,z]:
                        ep_map_thr[x,y,z] = ep_map[x,y,z]
                        
    # ep_map_thr[np.where(ep_map_thr<0.2)] = 0
    # where_are_NaNs = isnan(ep_map_thr)
    # ep_map_thr[where_are_NaNs] = 0
    # plt.imshow(ep_map_thr[:,350,:])
    ni_img = nib.Nifti1Image(ep_map_thr, img.affine, img.header)
    nib.save(ni_img,'ep_map_thr.nii')

    #-------------------------Linear fit R1 = p0 + p1*EP----------------------- ##

    t1_img = nib.load('t1.nii')
    t1_img = t1_img.get_fdata()

    lg_results = []

    count = 0

    for k in range(1,img_shape[2]-1):
        for i in range(1,img_shape[0]-1):
             for ii in range(1,img_shape[1]-1):
                 if ep_map[i, ii, k]>0 or t1_img[i, ii, k]>0:
                     r1_values = find_window(t1_img, i, ii, k)
                     ep_values = find_window(ep_map, i, ii, k)
                     x_ = np.array(ep_values)
                     y_ = np.array(r1_values)
                     x = x_.flatten().reshape((-1,1))
                     y = y_.flatten()
                     model = LinearRegression().fit(x,y)
                     p0 = model.intercept_
                     p1 = model.coef_
                     #cc = model.score(x, y_.flatten().reshape((-1,1)))
                     # cc = math.sqrt(cc_)
                     cc = np.corrcoef(x_.flatten(),y_.flatten())
                     V = ep_map_thr[i, ii, k]*abs(p1)
                     local_average = np.mean(x_)
                     results = [[i, ii, k],local_average,p0,p1,cc,V]
                     lg_results.append(results)


    csf_candidates = []
    wm_candidates =  []
    all_cand = []


    for item in range(len(lg_results)):
         item_ = lg_results[item]
         #cc = item_[4]
         cc = item_[4][0][1]
         # ccs.append(cc)
         V = item_[5][0]
         # vs.append(v)
         p1 = item_[3][0]
         # p1s.append(p1)
         all_cand.append(item_[0])
         if p1 < 0 and cc < 0 and V > 100:
             wm_candidates.append(item_[0])
         elif p1 > 0 and cc > 0 and V > 100:
             csf_candidates.append(item_[0])
             

    gmc_ = nib.load('gmc.nii')
    gmc = gmc_.get_fdata()

    gmc_t1_mask = gmc*t1_img


    csf_cand = np.zeros(img_shape)
    for i in range(len(csf_candidates)):
        x = csf_candidates[i][0]
        y = csf_candidates[i][1]
        z = csf_candidates[i][2]
        csf_cand[x,y,z] = 1


    wm_cand = np.zeros(img_shape)
    for i in range(len(wm_candidates)):
        x = wm_candidates[i][0]
        y = wm_candidates[i][1]
        z = wm_candidates[i][2]
        wm_cand[x,y,z] = 1
        
        
    csf_cand[np.where(ep_map_thr==0)]=0
    wm_cand[np.where(ep_map_thr==0)]=0
        
    ni_img = nib.Nifti1Image(csf_cand, img.affine, img.header)
    nib.save(ni_img,'csf_candidates.nii')   

    ni_img = nib.Nifti1Image(wm_cand, img.affine, img.header)
    nib.save(ni_img,'wm_candidates.nii')   


    #get avarage local values for T1 GM (excluding csf_cand and wm_candidates)

    gmc_t1_mask = gmc*t1_img
    gmc_t1_mask[np.where((csf_cand==1) & (wm_cand==1))]=0

    kernel5 = np.ones((5,5,5))

    gmc_t1_conv = ndimage.generic_filter(gmc_t1_mask, avg_func, footprint=kernel5)

    ni_img = nib.Nifti1Image(gmc_t1_conv, img.affine, img.header)
    nib.save(ni_img,'gmc_t1_conv.nii') 

    csf_cand_2 = np.zeros(img_shape)
    wm_cand_2 = np.zeros(img_shape)

    csf_cand_2[np.where((csf_cand==1) & (t1>1.1*gmc_t1_conv) & (t1>0))] = 1
    wm_cand_2[np.where((wm_cand==1) & (t1<0.9*gmc_t1_conv) & (t1>0))] = 1

    ni_img = nib.Nifti1Image(csf_cand_2, img.affine, img.header)
    nib.save(ni_img,'csf_cand_2.nii') 

    ni_img = nib.Nifti1Image(wm_cand_2, img.affine, img.header)
    nib.save(ni_img,'wm_cand_2.nii') 


    # csf_cand = np.zeros(img_shape)
    # for i in range(len(csf_candidates)):
    #     x = csf_candidates[i][0]
    #     y = csf_candidates[i][1]
    #     z = csf_candidates[i][2]
    #     px = t1_img[x,y,z]
    #     adj_t1_pixels = find_window(gmc_t1_mask,x,y,z)
    #     local_average_gm = np.mean(adj_t1_pixels)
    #     thres = local_average_gm * 0.1
    #     px_up = local_average_gm+thres
    #     if px>px_up: 
    #         csf_cand[x,y,z] = 1
            


    # wm_cand = np.zeros(img_shape)
    # for i in range(len(wm_candidates)):
    #     x = wm_candidates[i][0]
    #     y = wm_candidates[i][1]
    #     z = wm_candidates[i][2]
    #     px = t1_img[x,y,z]
    #     adj_t1_pixels = find_window(gmc_t1_mask,x,y,z)
    #     local_average_gm = np.mean(adj_t1_pixels)
    #     thres = local_average_gm * 0.1
    #     px_low = local_average_gm-thres
    #     if px<px_low: 
    #         wm_cand[x,y,z] = 1
            
        
    # ni_img = nib.Nifti1Image(wm_cand, img.affine, img.header)
    # nib.save(ni_img,'wm_candidates.nii')    

    # ------------------------------------------------------------------------------  
              
    temp = nib.load('csf.nii')
    temp = temp.get_fdata()

    still_available = np.zeros(img_shape)
    still_available[np.where((csf_cand_2>0) & (csf==0))]=1

    [myhelp,outer_edge] = get_edges(temp)

    found_pixels = np.zeros(img_shape)
    found_pixels[np.where((outer_edge==1) & (still_available==1))] = 1

    to_be_added=np.zeros(img_shape)
    to_be_added[np.where(found_pixels==1)] = 1

    my_num = np.sum(to_be_added)

    while my_num>0:
        temp[np.where(to_be_added==1)]=1
        csf_cand_2[np.where(found_pixels==1)]=0
        
        [myhelp,outer_edge] = get_edges(temp)

        found_pixels = np.zeros(img_shape)
        found_pixels[np.where((outer_edge==1) & (still_available==1))] = 1
        
        to_be_added=np.zeros(img_shape)
        to_be_added[np.where(found_pixels==1)] = 1
        
        my_num = np.sum(to_be_added)
        
    ni_img = nib.Nifti1Image(temp, img.affine, img.header)
    nib.save(ni_img,'csf_updated_2.nii') 

    # ------------------------------------------------------------------------------

    temp = nib.load('wm.nii')
    temp = temp.get_fdata()

    still_available = np.zeros(img_shape)
    still_available[np.where((wm_cand_2>0) & (wm==0))]=1

    [myhelp,outer_edge] = get_edges(temp)

    found_pixels = np.zeros(img_shape)
    found_pixels[np.where((outer_edge==1) & (still_available==1))] = 1

    to_be_added=np.zeros(img_shape)
    to_be_added[np.where(found_pixels==1)] = 1

    my_num = np.sum(to_be_added)

    while my_num>0:
        temp[np.where(to_be_added==1)]=1
        wm_cand_2[np.where(found_pixels==1)]=0
        
        [myhelp,outer_edge] = get_edges(temp)

        found_pixels = np.zeros(img_shape)
        found_pixels[np.where((outer_edge==1) & (still_available==1))] = 1
        
        to_be_added=np.zeros(img_shape)
        to_be_added[np.where(found_pixels==1)] = 1
        
        my_num = np.sum(to_be_added)
        
    ni_img = nib.Nifti1Image(temp, img.affine, img.header)
    nib.save(ni_img,'wm_updated_2.nii') 

    # --------------------------------------------------------------------------

    combined_tm_updated = np.zeros(img_shape)
    gmc_updated = np.zeros(img_shape)

    csf_updated = nib.load('csf_updated_2.nii')
    csf_updated = csf_updated.get_fdata()

    wm_updated = nib.load('wm_updated_2.nii')
    wm_updated = wm_updated.get_fdata()

    gmc_updated[np.where( (tm_bin==1) & (csf_updated!=1) & (wm_updated!=1))] = 1
    combined_tm_updated[np.where( (tm_bin==1) & (csf_updated!=1) & (wm_updated!=1))] = 1
    combined_tm_updated[np.where(csf_updated==1)] = 3
    combined_tm_updated[np.where(wm_updated==1)] = 2


    ni_img = nib.Nifti1Image(gmc_updated, img.affine, img.header)
    nib.save(ni_img,'gmc_updated_2.nii')   
      
    ni_img = nib.Nifti1Image(combined_tm_updated, img.affine, img.header)
    nib.save(ni_img,'combined_tm_updated_2.nii')  

    # -------------------------------------------------------------------------------
    os.chdir(cur_dir)