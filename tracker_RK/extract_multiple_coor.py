'''
from a tracking file, get the coordinates of multiple objects

Written by Amir Ohad 1/9/2024
'''
#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
#%%

def funcget_tracked_data(filename,obj=0,view=[],camera='nikon',contact=[]):
    with open(filename,"r") as datafile:
        lines= datafile.readlines()
        # del lines[0] # remove first line to avoid 2 zero times
        N=np.size(lines,0) # number of lines
        xtl=[[]]*N # x top left
        ytl=[[]]*N # y top left
        w=[[]]*N # box width
        h=[[]]*N # box height
        index=[[]]*N # tracked object index
        xcntr=[[]]*N # box x center
        ycntr=[[]]*N # box y center
        # dist=[[]]*N # distance from equilibrium position
        timer=[[]]*N # time marks
        timer[0]=0 # time starts at zero
        # timer_epoch1=[[]]*N
        i=0 # count rows
# x,y,w,h ; start at upper left corner
        for line in lines:
            if line==[]: break
            currentline = line.split(",") # split by ','
            index[i]=int(currentline[-2]) # get index of tracked object
            if index[i] in obj:    
                # print(i)    # if current line belongs to the requested tracked object
                xtl[i]=float(currentline[0]) # xtl- x top left
                ytl[i]=float(currentline[1])
                w[i]=float(currentline[2])
                h[i]=float(currentline[3])
                xcntr[i]=xtl[i]+w[i]/2 # calculate x coordinate of box center
                ycntr[i]=ytl[i]-h[i]/2
                # if view=='top':
                #     dist[i]=np.sqrt((xcntr[i]-xcntr[0])**2+(ycntr[i]-ycntr[0])**2)
                # else:
                #     dist[i]=abs(xcntr[i]-xcntr[0])
                
                timer[i] = 30*i*()
            else:
                print('skipped non-selected tracked object')
                print(timer[i],type(timer[i]),i,currentline)
            i+=1
            if i%50 ==0: 
                print(i)
                print(currentline)

        # x = np.subtract(xcntr,xcntr[0]) # return x relative to start point
        # y = np.subtract(ycntr,ycntr[0]) # return y relative to start point
        timer = np.zeros(N) # initialize timer array
        round_count = 0 # initialize round count
        C = max(index) # max number of tracked objects
        T = int(N/(C-1)) # number of time points per tracked object
        for i in range(N):
            if i % (C-1) == 0: # check if a new round starts
                round_count += 30 # increment round count
            timer[i] = round_count # assign timer value for each index
        return xcntr,ycntr,timer,index,N,C,T
#%% read data into dataframe
all_points_df = pd.DataFrame()
path = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2f_CN_stem_dynamics\1\1_full_Tracked\Rois_1_cropped_1\CN_stem_dynamics_1_CSRT.txt"
xcntr,ycntr,timer,index,N,C,T = funcget_tracked_data(path,obj=range(1,C),camera='nikon')
all_points_df['xcntr'] = xcntr
all_points_df['ycntr'] = ycntr
all_points_df['timer'] = timer
all_points_df['index'] = index
# Remove lines where index is 0
all_points_df = all_points_df[all_points_df['index']!=0]

#%% plot x data
plt.figure()
# plot x,y of 1 tracked object over time
for i in range(1,C):
    plt.plot(all_points_df['xcntr'][all_points_df['index']==i],
        label='object '+str(i))
plt.legend(fontsize=8)

# plot data
plt.figure()
# plot x,y of 1 tracked object over time
for i in range(1,C):
    plt.plot(all_points_df['ycntr'][all_points_df['index']==i],
        label='object '+str(i))
plt.legend(fontsize=8)
#%% plot x,y data of multiple objects over time by color code for time
plt.figure()
# plot x,y of 1 tracked object over time
for i in range(1,C):
    plt.scatter(all_points_df['xcntr'][all_points_df['index']==i],
        all_points_df['ycntr'][all_points_df['index']==i],
        c=all_points_df['timer'][all_points_df['index']==i],
        label='object '+str(i))
plt.legend(fontsize=8)
plt.colorbar()
plt.show()
#%% plot distance of each neighboring objects over time
# define distance calculation function
def calc_xy_dist_df(df, pa, pb):
    return np.array((df[pa] - arrX1[pb]))**2 + \
           (np.array(arrY1[pa]) - np.array(arrY1[pb]))**2
# calculate distance between first and second index for each point in time
plt.figure()
for pa in range(1,C):
    for pb in range(pa+1,C):
        dist = np.sqrt(calc_xy_dist_df(pa,pb))
        plt.plot(all_points_df['timer'][all_points_df['index']==pa], dist, label=f'{pa}-{pb}')
        

# plt.plot(all_points_df['timer'][all_points_df['index']==pa], dist)
plt.xlabel('Time')
plt.ylabel('Distance')
plt.title('Distance between first and second index')
plt.show()

#%%
#  return np.array((all_points_df['xcntr'][all_points_df['index']==pa]) - 
                # np.array(all_points_df['xcntr'][all_points_df['index']==pb]))**2 + \
                # (np.array(all_points_df['ycntr'][all_points_df['index']==pa]) -
                # np.array(all_points_df['ycntr'][all_points_df['index']==pb]))**2
#%%