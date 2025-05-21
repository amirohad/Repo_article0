# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:04:42 2023

mark the 2 points near the stem close to the contact position with the support

calculate the angle between the points

save to events variable

track 2 points - 1 from either side of the support
once the stem has crossed to it's other side, or 2 points
close to each other and the support from the 'far side' of the support.

use resample to lower noise

set threshold angle to determine twining (50 deg?)

notes:
* how to get rid of tracking artifacts, early high angles?
* !!! smoothing !!! resampling !!!
* start from later?

@author: Amir
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import h5py
import re
import os,glob # for listing files in folder
import math as m
import matplotlib.dates as mdates
from datetime import datetime
import csv
from scipy import stats
import scipy.interpolate
from scipy.signal import savgol_filter,resample
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import shutil

import importlib
import useful_functions as uf # my functions

importlib.reload(uf)
# do 9 1 and 2
#%% functions
# calculate angle from 2 xy points
def two_point_vs_horizontal_angle(p1,p2):
    '''get 2 points (x1,y1) and (x2,y2) and return their angle relative to the
    x axis'''
    try:
        a = np.degrees(m.atan((p2[1]-p1[1])/(p2[0]-p1[0])))
        return a
    except:
        return np.nan

# get tracked coordinates
def get_track_stem_near_support(filename):
    '''get filename, extract coordinates of 2 points
        close to the support from side view'''
    with open(filename,"r") as data:
        lines = data.readlines()
        N = round(np.size(lines,0)/2)
        times = [60*x for x in range(N)]
        sup_point = [[]]*N
        far_point = [[]]*N
        i = 0
        for line in lines:
            currentline = line.split(",")
            obj_ind = int(currentline[-2])
            xtl=float(currentline[0]) # xtl- x top left
            ytl=float(currentline[1])
            w=float(currentline[2])
            h=float(currentline[3])
            xcntr=xtl+w/2 # calculate x coordinate of box center
            ycntr=ytl-h/2
            if obj_ind == 0:
                sup_point[i] = [xcntr,ycntr]
            elif obj_ind == 1:
                far_point[i] = [xcntr,ycntr]
                i += 1
        return times,sup_point,far_point

# Function to safely convert a string to a float
def safe_float_convert(value):
    try:
        return float(value)
    except ValueError:
        return None  # or handle the error as you see fit

# Function to calculate the average from a cell
def calculate_average(cell_value):
    # Check if the cell value is a string
    if isinstance(cell_value, str):
        # Split the numbers by ',' and attempt to convert each to float
        numbers = [safe_float_convert(num) for num in cell_value.split(',')]
        # Filter out any failed conversions (None values)
        numbers = [num for num in numbers if num is not None]
        # Check if numbers list is empty after filtering
        if not numbers:
            return None  # or handle the empty list as you see fit
        return sum(numbers) / len(numbers)
    else:
        # Attempt to convert non-string values directly
        return safe_float_convert(cell_value)
    
# fitting
def logfunc(x, A, c):  return A * (np.log(x)) + c

#%% get tracking files
# get misc data for experiments
misc_path = r'C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2c_motor_modified_CN\motor mod_misx_new.xlsx'
motor_pandas = pd.read_excel(misc_path,sheet_name='meta-data motor_mod')
# motor speed is + if it is C-CW (with CN) and negative if it is CW (anti-CN)

# copy to
# log_folder_path = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\twine_init_logs"
# log_folder_path = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2c_motor_modified_CN\twine_init_logs"

# data
# path = r"\\132.66.44.238\AmirOhad\PhD\Experiments\Force Measurements\Exp2c_motor_modified_CN\measurements"
path = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2c_motor_modified_CN\twine_init_logs"
CN_path = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2c_motor_modified_CN\analyze_CN"
# pendulum files
# path = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\Measurements\angle_extract2\88_2_side"
# filename = r"\\132.66.44.238\AmirOhad\PhD\Experiments\Force Measurements\Exp2_Pendulum\Measurements\21\side\CSRT_021_1_near_contact\Rois_1_\CSRT_021_side_1_near_contact.txt"

# get all files in folder "path" to file_list
file_list = [os.path.join(path,file) for file in os.listdir(path) if file.endswith('motorized.txt')]
print(file_list)

# # walk through files
# for root,dirs,files in os.walk(path):
#     for file in files:
#         if file.endswith('motorized.txt'):
#             print(root,file)
#             filepath = os.path.join(root,file)
#             file_list.append(filepath)
#             end = filepath.split("\\")[-1]
#             shutil.copy2(filepath,os.path.join(log_folder_path,end)) # save copy

#%% calculate angle of line connecting these points relative to horizontal
for j in range(0,len(file_list)):
    try:
        # start_i = int(motor_pandas['contact_frame'][j])
        times,sup_point,far_point = get_track_stem_near_support(file_list[j])
        alpha = []
        l = len(far_point)-1
        for i in range(l):
                alpha.append(two_point_vs_horizontal_angle(far_point[i],sup_point[i]))
                # if i>start_i:
                # if abs(alpha[i])>0:
                # if j!=4: continue
                # plt.plot(times[i]/(60),alpha[i],'bx')
                # else: alpha[i]= 0
                # else: plt.plot(times[i]/(60),alpha[i],'cx')

        target_length = int(l/2)
        alpha_resample,times_resample = resample(alpha[:-1],target_length,t=times[:-1])
        if j==l-2:
            plt.figure()
            plt.plot(times[i]/(60),alpha[i],'bx')
            plt.plot(times_resample/60,alpha_resample,'rx')
            plt.title(file_list[j][-17:-14])

    except Exception as e:
        print(f"An error occurred: {e}") # Print the error message


# twine_time = []
# plt.figure()
for j in range(len(file_list)):
    a = []
    # start_i = int(motor_pandas['contact_frame'][j])
    # get coordinates of 2 points and times
    times,sup_point,far_point = get_track_stem_near_support(file_list[j])
    exp_num = re.findall('CSRT_\d+',file_list[j])[0][-2:]
    event_num = re.findall('CSRT_\d+_\d+',file_list[j])[0][-1]
    twine = 0
    row = motor_pandas[motor_pandas['exp_num']==int(exp_num)][motor_pandas['event_num']==int(event_num)]
    for i in range(len(far_point)):
        a.append(two_point_vs_horizontal_angle(far_point[i],sup_point[i]))
        # check if angle is near horizontal
        # if i>start_i: start = 1 # if using start_frame
        
        # check if threshold angle has been crossed
        # if start==1:
        if a[i-1] > 50 and a[i] > 50:
            # save twine time in minutes
            # motor_pandas.at[j,'twine_time(min)'] = times[i-start_i]/60
            row['twine_time(min)'] = times[i]/60
            twine = 1
            break
        if twine==0:
            row['no_twine_duration(min)'] = times[-1]/60
    # replace row in motor_pandas
    motor_pandas.loc[motor_pandas['exp_num']==int(exp_num)][motor_pandas['event_num']==int(event_num)] = row
#%% plots for motor modifications

plt.figure()
plt.plot(motor_pandas['effective_CN_rate(rph)'],motor_pandas['twine_time(min)'],'bx',label='twine')
plt.plot(motor_pandas['effective_CN_rate(rph)'],motor_pandas['no_twine_duration(min)'],'rx',label='no twine')
plt.xlabel('effective_CN_rate(rph)')
plt.ylabel('time(min)')
plt.legend()

# calculate (f_CN0+F_motor)/f_CN0 to ploi?

# Apply the function to df to calculate average CN time
# motor_pandas['CN_avg(min)'] = motor_pandas['CN_periods(min)'].apply(calculate_average)

# use the average CN time from CN-box-track-analysis file:
CN_list = [os.path.join(CN_path,file) for file in os.listdir(CN_path) if file.endswith('CN.xlsx')]


# get avg CN time from the CN_list acorrding to the exp number
# the avg CN time of each exp is in the 'CN_total_avg' column of each file in the CN-box list
# match for each event the CN time from the CN-box file
for i in range(len(CN_list)):
    exp_num = int(re.findall('\d+_analyze',CN_list[i])[0][:2])
    CN_box_i = pd.read_excel(CN_list[i])
    motor_pandas.loc[motor_pandas['exp_num']==exp_num,'CN_avg(min)'] = CN_box_i['CN_total_avg'][0]
    print(motor_pandas.loc[motor_pandas['exp_num']==exp_num,'CN_avg(min)'])

# Calculate the rate in rph
motor_pandas['CN_avg_rate(rph)'] = 60 / motor_pandas['CN_avg(min)']
motor_pandas['normalized_effective_CN_rate'] = \
        (motor_pandas['CN_avg_rate(rph)']-motor_pandas['stage_rotation(rph)'])#\
        # /motor_pandas['CN_avg(min)']

plt.figure()
non_zero_indices = motor_pandas['twine_time(min)'] >1
plt.plot(motor_pandas['normalized_effective_CN_rate'][non_zero_indices],
         motor_pandas['twine_time(min)'][non_zero_indices],'bx')
# plt.plot(motor_pandas['normalized_effective_CN_rate'],
#          motor_pandas['twine_time(min)'],'bx')
plt.xlabel(r'$f_{CN}+f_{motor}(rph)$',fontsize = 20)
plt.ylabel('twine time(min)',fontsize = 16)
# if motor is faster then CN rate then no twining is possible.
# contact will be broken.
plt.vlines(0, 0, 200,'r',label = 'motor faster then CN')
plt.legend()
#%% save to xl
backup_path = r'C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2c_motor_modified_CN\motor mod_misc_results.xlsx'
# uf.pdDF2xl(backup_path ,motor_pandas)
motor_pandas.to_excel(backup_path)
#%% linear fitting (manual twine time estimate)
# import data from xl
motor_pandas = pd.read_excel(misc_path,sheet_name='meta-data motor_mod')
# get (no)twine times and remove nans
data = pd.DataFrame({'x':motor_pandas['effective_CN_rate(rph)'],
                    'y1':motor_pandas['twine_time_estimate(min)'],
                    'y2':motor_pandas['no twine time(min)']}).dropna()

# plot raw data
y1_raw = data[data['y1']>0]['y1']
x1_raw = data[data['y1']>0]['x']
fig,ax = plt.subplots(3,1)
fs = 16
# make the figure taller
fig.set_figheight(15)
ax[0].plot(x1_raw,y1_raw,'bo',label='time to twine(min)')
# ax[0].set_title('time to twine(min) vs effective CN rate(rph)')
ax[0].set_xlabel('Effective CN rate(rph)',fontsize = fs)
ax[0].set_ylabel('time to twine(min)',fontsize = fs)
uf.fit_w_err(ax[0],x1_raw,5/60,y1_raw,2)
# plot semi-log
ax[1].plot(np.log(x1_raw),y1_raw,'bo',label='time to twine(min)')
# ax[1].set_title('semi-log plot')
uf.fit_w_err(ax[1],np.log(x1_raw),np.log(5/60),y1_raw,2)
ax[1].set_xlabel('log(Effective CN rate(rph))',fontsize = fs)
ax[1].set_ylabel('time to twine(min)',fontsize = fs)
# plot log-log
ax[2].plot(np.log(x1_raw),np.log(y1_raw),'bo',label='time to twine(min)')
# ax[2].set_title('log-log plot')
uf.fit_w_err(ax[2],np.log(x1_raw),np.log(5/60),
                    np.log(y1_raw),np.log(2))
ax[2].set_xlabel('log(Effective CN rate(rph))',fontsize = fs)
ax[2].set_ylabel('log(time to twine(min))',fontsize = fs)

#%% func fitting (manual twine time estimate)
# choose from: 'linfunc','logfunc','expfunc','polyfunc'
# import data from xl
misc_path = r'C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2c_motor_modified_CN\motor mod_misx_new.xlsx'
motor_pandas = pd.read_excel(misc_path,sheet_name='meta-data motor_mod')
# get (no)twine times and remove nans
data = pd.DataFrame({'x':motor_pandas['effective_CN_rate(rph)'],
                    'y1':motor_pandas['twine_time_estimate(min)'],
                    'y2':motor_pandas['no twine time(min)']}).dropna().sort_values('x') 

# plot raw data
y1_raw = data[data['y1']>0]['y1']
x1_raw = data[data['y1']>0]['x']
fig,ax = plt.subplots(3,1)
fs = 16
# make the figure taller
fig.set_figheight(15)
ax[0].plot(x1_raw,y1_raw,'bo')
# ax[0].set_title('time to twine(min) vs effective CN rate(rph)')
ax[0].set_xlabel('Effective CN rate(rph)',fontsize = fs)
ax[0].set_ylabel('time to twine(min)',fontsize = fs)
uf.fit_w_err(ax[0],x1_raw,0.05,y1_raw,5,fit_func=uf.expfunc)
# plot semi-log
ax[1].plot(np.log(x1_raw),y1_raw,'bo')
# ax[1].set_title('semi-log plot')
uf.fit_w_err(ax[1],np.log(x1_raw),0.05,y1_raw,5)
ax[1].set_xlabel('log(Effective CN rate(rph))',fontsize = fs)
ax[1].set_ylabel('time to twine(min)',fontsize = fs)
# plot log-log
ax[2].plot(np.log(x1_raw),np.log(y1_raw),'bo')
# ax[2].set_title('log-log plot')
uf.fit_w_err(ax[2],np.log(x1_raw),0.05,np.log(y1_raw),0.1)
ax[2].set_xlabel('log(Effective CN rate(rph))',fontsize = fs)
ax[2].set_ylabel('log(time to twine(min))',fontsize = fs)

#%% Calculate the average y values for const jumps in x values
x_values = np.arange(data['x'].min(), data['x'].max(), 0.1)
y1_avg = []
y2_avg = []
gap = 0.1 
for x in x_values:
    y1_values = data[data['x'].between(x, x+gap)]['y1']
    y2_values = data[data['x'].between(x, x+gap)]['y2']
    
    # Remove zeros from y1_values and y2_values
    y1_values = y1_values[y1_values > 0]
    y2_values = y2_values[y2_values > 0]
    
    y1_avg.append(y1_values.mean())
    y2_avg.append(y2_values.mean())

# Plot the averages
plt.plot(x_values, y1_avg, 'bo', label='average time to twine(min)')
plt.plot(x_values, y2_avg, 'ro', label='average time of no twining(min)')
# ax[0].errorbar(all_r_dict.mean().keys(),all_r_dict.mean().values,
                # xerr=5/np.sqrt(12),yerr=all_r_dict.std(),fmt='o',)
plt.ylim([0, 250])
plt.xlabel('Effective CN rate(rph)')
plt.ylabel('Average time(min)')
# plt.title('Average of y for each 0.1 jump in x')
plt.legend()
plt.show()
# %%
