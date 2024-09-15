# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:08:26 2022

Here is just to find angles, is the latest Methieu version after he corrected the angle measurement.
this is part of the "python_project" code

@author: ague
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import os
import _pickle as pkl
import matplotlib.dates as mdates
from datetime import datetime
import csv
from scipy import stats


##Part 1##

##input paths##
disk_location =r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\Measurements\root_stem_results\interekt_74_e_3_3200-3225.h5" #is the h5 file from SLEAP tracker

##output paths##
box_plant = disk_location.split("\\")[-1].split(".")[0]
box = box_plant.split('_')[0]
#box_plant= disk_location[-24:-12]
#box= disk_location[-24:-20]
output_path= r"C:\Users\user\Desktop\ALPHA\2023"
file_means_box = output_path + f'\{box}.csv'#{box}file mean from all the plants you worked with find_angle function
# angles_box= r'C:\Users\user\Desktop\ALPHA\exp3\angles\randi\means_matches\box_5.csv'
file_right = output_path + f'\\right_{box_plant}.csv' #file wuth angles dor right leaf
file_left = output_path + f'\\left_{box_plant}.csv' #file wuth angles dor left leaf
file_mean_plant = output_path + f'\\mean_{box_plant}.csv' #file average plant movement (left+right/2)

#%%

##Part 2##
def find_angle (h5_file, file_right, file_left, file_mean_plant, file_means_box, file_treatment):
    """
    Parameters
    ----------
    h5_file : file from SLEAP tracker with the locations
    file_strings : file with the picture names-pictire dates

    file_right : where you will save the right angles list
    file_left : where you will save the left angles list

    Returns
    -------
    plant angle : array with the average movement from both leaves

    """
    with h5py.File(h5_file, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

    list_theta_r_raw = []
    list_theta_r = []
    list_theta_l_raw = []
    list_theta_l = []

    for i in range(0, np.shape(locations)[0]):

    # Getting coordinates for all tracked points:
        center_point = np.array(locations[i,0,:,0])
        top_leaf_R = np.array(locations[i,1,:,0])
        low_leaf_R = np.array(locations[i,2,:,0])
        top_leaf_L = np.array(locations[i,3,:,0])
        low_leaf_L = np.array(locations[i,4,:,0])
        # NB: center_point[0] = x ; center_point[1] = -y (the sign is because of the 'image convention')
        # Computing the angles between the direction defined by the leaf and the downward vertical:

        ## Leaf of the right:
        dx_R = low_leaf_R[0] - top_leaf_R[0]
        dy_R = -(low_leaf_R[1] - top_leaf_R[1]) # (the sign is because of the 'image convention')

        theta_R_raw = np.arctan2(dy_R, dx_R) # angle between the horizontal and the leaf
        if theta_R_raw <=0:
            theta_R = 90+np.rad2deg(theta_R_raw) # angle between the (downward) vertical and the leaf
        else:
            theta_R = 90+np.rad2deg(theta_R_raw) # angle between the (downward) vertical and the leaf


        ## Leaf on the left:
        dx_L = low_leaf_L[0] - top_leaf_L[0]
        dy_L = -(low_leaf_L[1] - top_leaf_L[1]) # (the sign is because of the 'image convention')

        theta_L_raw = np.arctan2(dy_L, -dx_L) # angle between the horizontal and the leaf
        # NB: Here we take -dx_L instead of dx_L because the leaf points to the left
        # It's like taking the mirror image of the left leaf
        # Doing so, we have the same angle convention for theta_R and theta_R
        if theta_L_raw <=0:
            theta_L = 90+np.rad2deg(theta_L_raw) # angle between the (downward) vertical and the leaf
        else:
            theta_L = 90+np.rad2deg(theta_L_raw) # angle between the (downward) vertical and the leaf

        list_theta_r_raw.append(theta_R_raw)
        list_theta_r.append(theta_R)
        list_theta_l_raw.append(theta_L_raw)
        list_theta_l.append(theta_L)

    plant_angle= np.nanmean(np.array([list_theta_l, list_theta_r]), axis= 0) ##Creates a plant mean using both leaves

    #saves
    np.savetxt(file_left, list_theta_l, delimiter = ",")  #save list in CSV file for (maybe) future analysis
    np.savetxt(file_right, list_theta_r, delimiter = ",")  #save list in CSV file for (maybe) future analysis
    np.savetxt(file_mean_plant, plant_angle, delimiter = ",", header= box_plant) #save plant mean
    # next four lines save and append mean angles, use it to save means from the same traetment in single file
    save_mean_box= open(file_means_box, 'a')
    writer = csv.writer(save_mean_box, delimiter=',')
    writer.writerow(plant_angle)
    save_mean_box.close()

    save_mean_treatment= open(file_treatment, 'a')
    writer = csv.writer(save_mean_treatment, delimiter=',')
    writer.writerow(plant_angle)
    save_mean_treatment.close()

    #plant_angle.to_csv('file_means_box', mode='a', index=False, header=box_plant)

    return plant_angle

#angles= (find_angle(disk_location, file_right, file_left))
#print(angles)
#print(len(angles))

#%%%
folder_disk_location = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\Measurements"



lst_of_files = os.listdir(folder_disk_location)


for file in lst_of_files:

    print(f'{5*"###"}\n{file}\n{5*"###"}')
    plant_box = file.split(".")[0]
    box = plant_box.split('_')[0]

    output_path= r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\Measurements"
    disk_location = folder_disk_location +"\\"+ file

    ## creating output paths
    file_means_box = output_path + f'\\mean_box\\{box}.csv'#{box}file mean from all the plants you worked with find_angle function
    file_treatment = output_path + f'\\treatment.csv' #file with all plants from the treatment
    file_right = output_path + f'\\right\\right_{plant_box}.csv' #file wuth angles dor right leaf
    file_left = output_path + f'\\left\\left_{plant_box}.csv' #file wuth angles dor left leaf
    file_mean_plant = output_path + f'\\mean_plant\\mean_{plant_box}.csv' #file average plant movement (left+right/2)

    find_angle(disk_location, file_right, file_left, file_mean_plant, file_means_box, file_treatment)




