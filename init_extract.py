# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 13:55:22 2022

multiple file data extract and transfer to excel


1. open folder with new init image and initial contact image

2. get from images:
support length(pix)
y coordinate in eq of support bot edge
reference length for pix2cm conversion
contact distance from stem tip to contact with support

save to excel

@author: Amir
"""

#%% imports
import numpy as np
import pandas as pd
from glob import glob# for listing files in folder
import os

# custom functions
import sys
sys.path.append(r"C:\Users\Amir\Documents\PHD\Python\Data_Extraction")

sys.path.append('..')
import useful_functions
from useful_functions import get_ypix,distance_2d,arr_to_excel,\
    exp_details,live_line, append_to_list

#%% initialize
filenames = []
exp = []
temp = []
view_t = []
exp_num = []
event_num = []
view = []
support_length = []
dist_straw_from_hinge = []
side_eq_ypos_sup_bot = []
pix2cm = [] #  length(cm) / length (pix)
contact_distance = []
minsc_photo_path = r'C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\minsc_photos'
initial_path = r'\initial\new'
contact_path = r'\contact\new'
init_files = glob(os.path.join(minsc_photo_path+initial_path,'*.JPG'))
contact_files = glob(os.path.join(minsc_photo_path+contact_path,'*.JPG'))
N = len(init_files)
M = len(contact_files)
exp_dict = {} # save data of each exp and use per event
event_dict = {} # data per event
#%% init details: dictionary
for i in range(N):
    exp_details(init_files[i], exp_num, event_num, view) #add to temp for this exp

    line = []
    if view[-1] == 'side': # if side view
        live_line(init_files[i],'straw length',line) # get length from img in pix, q to finish, d to delete line
        support_length.append(distance_2d(line[-1])) #append to var
        live_line(init_files[i],'distance of support from hinge',line)
        dist_straw_from_hinge.append(distance_2d(line[-1]))
        side_eq_ypos_sup_bot.append(get_ypix(init_files[i],'bottom of support')) # get y pos of bottom of support

    # pix2cm for either side or top .view
    line = []
    live_line(init_files[i],'reference length',line) # ref length for pix2cm conversion
    real_dist = float(input()) # get actual distance of ref obj
    sumdist = 0
    for l in line: # if using multiple lines
        sumdist += distance_2d(l) # sum len of lines for reference
    pix2cm.append(real_dist/sumdist) # pix distance from image, input real distance by hand
    # create exp dictionary
    if view[i] == 'side':
        add2dict = [dist_straw_from_hinge[i],support_length[i],side_eq_ypos_sup_bot[i],pix2cm[i]] # add parameters to new exp
        exp_dict[exp_num[i]] = add2dict
    elif view[i] == 'top': # add top view pix2cm
        append_to_list([support_length,dist_straw_from_hinge,side_eq_ypos_sup_bot],-1,1) #
        add2dict.append(pix2cm[i])
        exp_dict[exp_num[i]] = add2dict

for j in range(M): # 2M rows, side+top per contact img
    # exp_details(init_files[i], exp_num, event_num, view)-> temp =  [exp_num, event_num, view]
    exp_details(contact_files[j],temp,temp,temp) # get contact pic data
    # extract contact distance
    line = []
    live_line(contact_files[j],'contact distance',line) # press q to finish
    sumlen = 0
    for l in line: # sum all lines drawn
        sumlen += distance_2d(l) # sum over all lines
    contact_distance.append(sumlen) # save contact distance

    # add same contact distance to both views
    add2dict = exp_dict[temp[-3]].copy()
    event_dict[(temp[-3], temp[-2], 'side')] = add2dict
    event_dict[(temp[-3], temp[-2], 'side')].append(contact_distance[j])
    event_dict[(temp[-3], temp[-2], 'top')] = event_dict[(temp[-3], temp[-2], 'side')] # copy to top

#%% convert unified data to excel
#set column names
column = ['support dist from hinge(pix)','support length(pix)',
            'y coordinate in eq of support bot edge',
            'pix2cm_side','pix2cm_top','contact distance from stem tip'] #,'exp number','event','view']

df = pd.DataFrame.from_dict(event_dict,orient='index',columns=column)
# df = df.T

xl_path = minsc_photo_path + '\\{:d}-{:d}.xlsx'.format(int(exp_num[0]),int(exp_num[-1]))
writer = pd.ExcelWriter(xl_path, engine = 'xlsxwriter')
df.to_excel(writer)
writer.close()

#%% measure distance on specific img
filename = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\Calibrations\fluctuations_1sec_int\DSC_8159.JPG"
line = []
live_line(filename,'contact distance',line) # press q to finish
sumlen = 0
for l in line: # sum all lines drawn
    sumlen += distance_2d(l) # sum over all lines
print(sumlen)
#%%