# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:38:27 2022

choose how many sections to measure
get pix2cm length and input actual length
measure distance of each segment from stem tip
in each frame: for each segment mark line(or connect a few lines)
    between top of each dot to the bottom of the next dot above it.

@author: Amir
"""
#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from glob import glob# for listing files in folder
import os
import re
import sys
sys.path.append('..')
from useful_functions import get_ypix,arr_to_excel,\
    exp_details,live_line, append_to_list,multiline_dist,distance_2d
import time
from stat import ST_MTIME


def get_time(file_path):
    if not os.path.isfile(file_path):
        return("path time not found")
    STAT = os.stat(file_path)
    # return time.strftime('%d-%m %H:%M', time.localtime(STAT[ST_MTIME]))
    return time.strftime('%d-%m %H:%M:%S', time.localtime(STAT[ST_MTIME]))
#%% initialize
base_path = r'C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\Measurements'
exp = 114
exp_growth = '\\'+ str(exp)+r'\Side\growth'
files = glob(os.path.join(base_path+exp_growth,'*.JPG'))

# get time series in minutes- first frame is time=0 and time differences
frames =[int(re.findall('DSC_\d{4,5}',file)[0][4:]) for file in files if re.findall('DSC_\d{4,5}',file)]
mnts = np.divide(frames,2)
times = [int(x-min(mnts)) for x in mnts] # add time in minutes
# times.sort() # sort times
dt = [y-x for x,y in zip(times[:-1],times[1:])]

n = 3 # number of distances to track
N = len(files) # number of time points
segments =[]

for file in files: # get pix2cm
    if re.findall('pix2cm',file):
        pix_length = multiline_dist(file,'pix2cm') # get pix2cm
        real_dist = float(input()) # get actual distance of ref obj
        pix2cm = real_dist/pix_length # conversion rate

for i in range(0,n): # for each segment, open img and measure line
    segments.append((multiline_dist(files[0], 'distance of section '+str(i+1)+' from tip,q-exit,d-delete')))
segment_names = ['{:.2f} cm from tip'.format(x*pix2cm) for x in segments]


df_length_pix = pd.DataFrame(columns=segment_names)
df_growth = pd.DataFrame(columns=segment_names)
df_gvel = pd.DataFrame(columns=segment_names)

for file in files:
    # length =[]
    # if re.findall('pix2cm',file):
    #     pix_length = multiline_dist(file,'pix2cm') # get pix2cm
    #     real_dist = float(input()) # get actual distance of ref obj
    #     continue
    # else:
    if not re.findall('pix2cm',file):
        frame = re.findall('DSC_\d{4,5}',file)[0][4:]
        seg_dist = []
        for i in range(0,n): # for each segment, open img and measure line
            seg_dist.append((multiline_dist(file,'img '+frame+ ', section '+str(i+1)+' q-exit,d-delete')))
        df_length_pix.loc[frame] = seg_dist

# convert pix to length and calculate growth, growth velocity and growth rate
segments = np.multiply(segments,pix2cm) # distance from tip in cm
df_length = df_length_pix.multiply(pix2cm) # lengths in cm

df_growth = abs(df_length.diff(axis=0)) # growth for each time span
df_growth = df_growth.iloc[1: , :] # drop first row (nan)
df_gvel = df_growth.div(dt,axis=0) # divide growth by delta_t to get velocity
# df_grate = df_gvel.div(df_length[:, df_length.columns != 'time']) # ??
df_grate =  df_gvel.div(df_length.iloc[1:,:])*100 # divide by length of previous section to get % g_rate

df_length['time']=times[:] # add time to df
df_growth['time']=times[1:]
df_gvel['time']= times[1:]
df_grate['time']= times[1:]


# sort by time stamp
df_length = df_length.sort_values(by=['time'])
df_growth = df_growth.sort_values(by=['time'])
df_gvel = df_gvel.sort_values(by=['time'])
df_grate = df_grate.sort_values(by=['time'])
#%% plots
plt1 = df_gvel.plot('time')
plt1.set_xlabel(r'time(min)',fontsize=16)
plt1.set_ylabel('growth velocity(cm/min)',fontsize=16)
plt1.legend(loc='upper left')

plt2 = df_length.plot('time')
plt2.set_xlabel(r'time(min)',fontsize=16)      #'\alpha > \beta$')
plt2.set_ylabel('segment size(cm)',fontsize=16)
plt2.legend(loc='upper left')

avggrate = np.mean(df_grate)
plt.figure()
plt.plot(segments,[avggrate[0],avggrate[1],avggrate[2]],'x')
plt.xlabel('distance from tip(cm)',fontsize=16)
plt.ylabel('average growth rate(%)',fontsize=16)

plt4 = df_grate.plot('time')
plt4.set_xlabel(r'time(min)',fontsize=16)
plt4.set_ylabel('growth rate(%)',fontsize=16)
plt4.legend(loc='upper left')
#%% to excel
xl_path = base_path+exp_growth

with pd.ExcelWriter(xl_path+r'\growth data.xlsx') as writer:
    df_length.to_excel(writer, sheet_name='segment lengths')
    df_growth.to_excel(writer, sheet_name='growth lengths')
    df_gvel.to_excel(writer, sheet_name='growth velocity')
    df_grate.to_excel(writer, sheet_name='growth rate')
    # writer.close()

#%%
l = float(input('input actual length: '))
#%%