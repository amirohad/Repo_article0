# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:38:45 2023

cleaned script for data analysis and figure plotting for twinig initiation paper

1. import libraries
2. import main excel file
3. import tracking data (support position, contact position,2 stem points near support)
4. import results from root_stem_extractor
5. remove problematic events from raw data
6. populate event-class variables
7. populate jump times form CN delay exp and motorized CN exp
8. get AmirP simulation data
9. figures:
    1. Force trajectory of 1 twine and 1 slip
    2. Radius and Youngs modulus fits
    3. F fit to sine and compare to simulation
    4. correlation(s?) of fit amplitude
    5. fit_period-normalized F(t) vs CN-normalized F(t)
    6. CN delay data: expected vs measured
    7. twine vs slip distributions:
        a. l_tip, b. T_dec/T_CN, c. eps_max, d. F_int (or?) F_max
    8. motorized CN data?

@author: Amir Ohad
"""
#%% import libraries
# generic
import sys
import numpy as np
import os,glob # for listing files in folder
import re # regular expressions
import pandas as pd
import scipy
import seaborn
import math as m
from scipy.signal import savgol_filter
from scipy.spatial import distance as sci_distance
from scipy.stats import kruskal

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.lines as mlines
import random
import itertools
# from progress_bar import InitBar
from tqdm import tqdm
import pickle as pkl
import datetime
now = datetime.datetime.now()

# custom
import exp2funcs_clean
sys.path.append('..')
import useful_functions as uf 

# import error_estimates

#general parameters
cmap = plt.get_cmap('viridis')
fs = 16 # standard font size for plots

# unit conversion for paper figures
mg2mN = 1/100
mg2uN = 1*10

#%% import data
root_path = r'C:\Users\Amir\Documents\PHD\Experiments\Force Measurements'
basepath = root_path+r'\Exp2_Pendulum' # pendulum exp folder
figure_folder = r'C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Figures'
excel= r'\Exp2_supplementary_measurements_events.csv'

# jump_path = r'\jump_position\jump_pos_w-diameter_pix.csv'
jump_path2 = r'\jump_position\cn_delay.csv'
# straw exp data
data_panda = pd.read_csv(basepath+excel)


# import tracked data, initiate variables
track_sup_path = r'\track_logs'
track_folder_path = basepath+track_sup_path

track_contact_path = r'\contact_track_logs'
contact_folder_path = basepath+track_contact_path

track_near_sup_path = r'\twine_init_logs'
track_near_sup_folder_path = basepath+track_near_sup_path

h5_path = r'\Measurements\root_stem_results'
h5_folder_path = basepath+h5_path

E_path = r'\Young_moduli'
E_folder_path = basepath+E_path

N_track = len(os.listdir(track_folder_path)) # get number of files in track folder
N_contact=len(os.listdir(contact_folder_path)) # get number of files in contact folder
N_tot = len(data_panda) # get number of lines in excel
N_E = len(os.listdir(E_folder_path))

jump_root_path = r"\\132.66.44.238\AmirOhad\PhD\Experiments\Force Measurements\Exp2b_CN_Delay\Measurements"
# CN delay data
CN_delay_path = jump_root_path + r'\first_last_plusXmin\img_angles_merged.xlsx'
jump_panda = pd.read_excel(CN_delay_path)

#%% remove problematic events from raw data
delete_rows = [] # save rows to delete
problem_exp = [] # exp_num of problem events

for i in range(N_tot): # remove problem events and non-Helda events
    if data_panda.at[i,'problem']!='na' or data_panda.at[i,'Bean_Strain']!='Helda':
        delete_rows.append(i)
        problem_exp.append(data_panda.at[i,'Exp_num'])
N = N_tot - len(delete_rows) # modify num of rows

data_panda = data_panda.drop(data_panda.index[delete_rows]) # remove prob. events
data_panda = data_panda.reset_index() # redo index
#%% get misc. file lists

# get track files for support bottom coordinates
remove_chars = re.compile('[,_\.!?]') # what to remove from strings
track_dict = {} # save support track in dictionary by exp and events
i=0 # start with first track file
for file in glob.glob(os.path.join(track_folder_path, '*.txt')): # for each event:
    exp = int(re.findall('_\d{3,4}_',file)[0].replace('_','')) # find exp number (3-4 digits)
    event = int(re.findall('[0-9]',re.findall('_\d{1}\D',file)[0].replace('_',''))[0]) # find event number
    viewt = re.findall('(side{1}|top{1})',file)[0] #.replace('_','')
    track_dict[(exp,event,viewt)] = [file] # add new exp
    i+=1

# get track files for stem-support contact coordinates
contact_dict = {} # save support contact track in dictionary by exp and events
i=0 # start with first contact file
for file in glob.glob(os.path.join(contact_folder_path, '*.txt')): # for each event:
    exp = int(re.findall('_\d{3,4}_',file)[0].replace('_','')) # find exp number (3-4 digits)
    event = int(re.findall('_[0-9]_',file)[0].replace('_','')) # find event number
    contact_dict[(exp,event)] = [file] # add new exp
    i+=1

# get track files for 2 stem positions on either side of support
near_sup_track_dict = {}
i=0 # start with first stem_near_sup file
for file in glob.glob(os.path.join(track_near_sup_folder_path, '*.txt')): # for each event:
    exp = int(re.findall('_\d{3,4}_',file)[0].replace('_','')) # find exp number (3-4 digits)
    event = int(re.findall('_[0-9]_',file)[0].replace('_','')) # find event number
    near_sup_track_dict[(exp,event)] = [file] # add new exp
    i+=1

# get h5 files for stem near support
h5_dict = {}
i=0 # start with first h5 file
for file in glob.glob(os.path.join(h5_folder_path, '*.h5')): # for each event:
    exp = int(re.findall('interekt_\d{2,3}_',file)[0].split('_')[1]) # find exp number (3-4 digits)
    event = int(re.findall('e_[0-9]_',file)[0].split('_')[1]) # find event number
    start_frame = int(re.findall('_\d{2,5}-\d{2,5}',file)[0].replace('_','').split('-')[0]) # find start frame
    h5_dict[(exp,event,start_frame)] = [file] # add new exp
    i+=1

# get Young modulus files
E_dict = {}
for file in glob.glob(os.path.join(E_folder_path, '*.csv')):
    exp = int(re.findall('\d{2,3}',file)[0])
    E_dict[exp]=pd.read_csv(file,header=None)

#%% clear plants and events
plants = []
events = []
#%% populate plant and event instances
i = 0
N = len(data_panda)
for i in tqdm(range(N)): # N
    exp = int(re.findall('\d{3,4}',data_panda.at[i,'Exp_num'])[0]) # exp num
    view = data_panda.at[i,'View']  # side of top view
    if view == 'top':
        plants[-1].pix2cm_t = float(data_panda.at[i,'Top_pix2cm'])

    if i==0 or exp!=plants[-1].exp_num: # append new plant with data from pandas
        #basic data
        plants.append(exp2funcs_clean.Plant(data_panda,basepath,i,exp))
        # view dependent data
        plants[-1].view_data(data_panda,i)
        # circumnutation data
        plants[-1].cn_data(data_panda,i)
        #Youngs modulus by segment with avg
        plants[-1].getE(E_dict)


    #event
    event =  int(re.findall('_[0-9]',data_panda.at[i,
       'Exp_num'])[0].replace('_','')) # find event number

    # if exp!=25 or event!=0: continue   # work specific event

    # if it is the 1st event or the previous event_num different from current,
    # add new event to list
    if len(events)==0 or events[-1].event_num != event or \
        events[-1].p.exp_num != exp:
        events.append(exp2funcs_clean.Event(plants[-1],data_panda,i))
    events[-1].event_num = event

    # view dependent data
    events[-1].view_data(data_panda,i,view)

    # stop for debugging
    # if i==125:
    #     print('check')

    # get automated extraction of twine(decision) time
    events[-1].get_twine_time(exp,event,view,
                      h5_dict,near_sup_track_dict,50,track_dict,to_plot=0)

    # print for debugging
    # try: print(f' auto-dec index = {events[-1].auto_dec_index}')
    # except: print(f'auto twine_time method: {events[-1].auto_twinetime_method}')

    # get track data, select decision period data, pix2cm,
    events[-1].event_base_calcs(view,track_dict,contact_dict)
    # calc
    events[-1].event_calc_variables(view)
    # events[-1].jump_angle(view,jump_panda)
    print(f'i={i}, exp number {exp}, {event}')

#%% figure - fit Amirp simulation data + residual (supplementary 2)
# define parameters

TIMESCALE = 1.0
eg_per_cycle = 0.1
kappa_cn=0.1
growth_timestep = 30 #sec
T_cyc = 360
cycle_time= T_cyc*growth_timestep
N_cycles= 1 #will run for 1 full cycle
T=N_cycles*T_cyc
CN_omega=2*np.pi/cycle_time

growth_stretch=1.0+eg_per_cycle*growth_timestep/cycle_time #not really growing
base_length = 20 #mm

base_radius = 1 #mm
density = 10000*((1e-3)**3) #not actual mass! for computation
nu = 0.1#0.5
E = 5e6*1e-3 #kilo Pa
poisson_ratio = 0.5
growth_zone=base_length #mm

# CN=kappa_cn*base_radius*CN_omega/growth_rate #CN sensitivity

cylinder_height=np.array([25.0])
cylinder_radius=np.array([1.0])
cylinder_density=np.array([100000.0])

# import raw data
sim_file = 'twining_sims.pckl'
sim_path = os.path.join(basepath[:-14],
                    'Sim & models Amirp','simulation_data',sim_file)
with open(sim_path, 'rb') as f:
    data = pkl.load(f, encoding='latin1')

# take one trajectory and normalize it by maximum and Tcn
k = 8 # 8 default
i0=40 # 15 default
i1=140 # 140 default
sim_f = data['f'][k][i0:i1]-data['f'][k][i0]
sim_t = np.subtract(data['t'][8][i0+1:i1+1],data['t'][8][i0+1])
Tcn = 180
sim_f_norm = np.divide(sim_f,max(sim_f))
sim_t_norm = np.divide(sim_t,Tcn)
# should i add all simulation plots with std?

# fit raw sim to sine
res_sim = uf.fit_sin(sim_t, sim_f,1/180)
fitted = res_sim["fitfunc"]
offset = res_sim["offset"] # from fit
Amplitude = res_sim["amp"]
period = res_sim["period"]

#simulation normalized by fit to sine
sim_t_sine_norm = np.divide(sim_t,period)
sim_f_sine_norm = np.divide(sim_f,Amplitude)
cutoff = uf.closest(sim_t_sine_norm,0.4)

# plot check
fig,ax = plt.subplots(1,2)
x_range = np.linspace(0,max(sim_t),1000)
ax[0].plot(sim_t,sim_f,'b',label='sim data')
ax[0].plot(x_range,uf.sinfunc(x_range,
            res_sim["amp"],res_sim["omega"],res_sim["offset"]),'r',label='sim fit')
ax[0].set_xlabel(r'$t$',fontsize=fs)
ax[0].set_ylabel(r'$F$',fontsize=fs)
ax[0].legend()

x_range = np.linspace(0,max(sim_t_norm),1000)
ax[1].plot(sim_t_norm,sim_f_norm,'b',label='normalized by input CN-period')
ax[1].plot(sim_t_sine_norm,sim_f_sine_norm,'r',label='normalized by fitted sine period')
ax[1].set_xlabel(r'$t/T_{norm}$',fontsize=fs)
ax[1].set_ylabel(r'$F/A_{norm}$',fontsize=fs)
ax[1].legend()
plt.tight_layout()

plt.figure()
plt.plot(sim_t,sim_f-uf.sinfunc(sim_t,
            res_sim["amp"],res_sim["omega"],res_sim["offset"]))
savep=0
if savep==1:
    plt.savefig(basepath+
            r"\figures\Paper\fit2sine\simulation fits and normalizations.svg",
            dpi=200)
#%% figure - fitting exp data to sine
# F fit to sine and filter events with r^2<0.85(?)
N= len(events)
time = [[]]*N
allf_interp = []
savep = 0

plt.figure()
count = []
tmax = 0

for i in range(N):
    try:
        # filters:
        # if events[i].twine_state==0: continue # remove slip/twine events
        # f = uf.remove_outliers(events[i].F_bean,0.75)
        f = uf.remove_outliers_loop(events[i].F_bean,cutoff=0.75,loop=50) # reduce outlying data points
        events[i].f_clean = f
        l = len(f)
        if l<5: continue # remove super short events
        if f[0]>0.5: continue # remove large non-zero start
        t = events[i].timer
        events[i].n_Tcn = uf.closest(t,events[i].p.avgT*60*0.5) # cutoff fits till half of T_CN
        time[i] = np.divide(t[:events[i].n_Tcn],60) # time in minutes

        #  fit to sine:
        res = uf.fit_sin(time[i], f[:events[i].n_Tcn],1/(2*events[i].p.avgT))
        count.append(i)
        events[i].sine_fit = res
        events[i].fit_offset = res["offset"] # from fit
        events[i].fit_A = res["amp"]
        events[i].fit_period = res["period"]
        events[i].r_squared = res["r_squared"]
        events[i].A_err = np.sqrt(np.diag(res['rawres'][1]))[0][0]
        if events[i].twine_state==1:
            e_color='blue'
            e_alpha=0.5
        elif events[i].twine_state==0:
            e_color='green'
            e_alpha=0.5

        # plot individual event forces normalized by sine fit:
        # plt.plot(time[i]/(events[i].fit_period),
        #          (f[:events[i].n_Tcn]-events[i].fit_offset)/(events[i].fit_A),'-',
        #          color=e_color,alpha=e_alpha,markersize=0.2)

        # interpolate normalized forces:
        interp_f = scipy.interpolate.UnivariateSpline(time[i]/(events[i].fit_period),
                   (f[:events[i].n_Tcn]-events[i].fit_offset)/(events[i].fit_A),
                   k=2,ext='zeros') # interpolate
        normtime = np.arange(time[i][0]/(events[i].fit_period),
                     time[i][-1]/(events[i].fit_period),0.001) # get normalized time stamps
        tmax = max(tmax,max(normtime))
        allf_interp.append(interp_f(normtime)-interp_f(normtime)[0])
        plt.plot(normtime,interp_f(normtime)-interp_f(normtime)[0],
                 color=e_color,alpha=e_alpha,zorder=2)
        # continue
    except Exception as e:
        print(f'error: {e}, {i=}')
        continue
print(f'plotted {len(allf_interp)=} curves')

# calc mean,median, std from interpolated forces
allf_interp = pd.DataFrame(allf_interp)
# Count the non-NaN values in each column
non_nan_counts = allf_interp.apply(lambda col: col.count(), axis=0)
# Find columns with less than k non-NaN values
k=15
columns_below_threshold = non_nan_counts[non_nan_counts < k].index.tolist()
# remove columns after that which has less then k non-NaN values
allf_interp = allf_interp.iloc[:, :min(columns_below_threshold)]

avgf = allf_interp.mean()
l_avg = len(avgf)
# avgf = scipy.signal.savgol_filter(avgf,int(0.4*l_avg)+int(0.4*l_avg)%2+1, 2)
medf = allf_interp.median()
stdf = allf_interp.std()

# plot sine func, mean normalized data, stderr
x_range = np.linspace(0, 1,1000)
allnormtime = np.arange(0,tmax+0.1,0.001)
plt.fill_between(allnormtime[:len(avgf)],avgf-stdf, avgf+stdf,
                 alpha=0.7,zorder=0,color='lightblue')
plt.plot(x_range[:len(avgf)],avgf,'b-',
         linewidth=2,label=r'$F_{mean}$',zorder=1)
res_avgf = uf.fit_sin(x_range[:len(avgf)], avgf,1/(np.pi))
plt.plot(x_range, uf.sinfunc(x_range,res_avgf["amp"],
             res_avgf["omega"],res_avgf["offset"]),'--r',
             label=r'$sin(\omega_{fit2mean}t)$',zorder=1) #sine fit to $F_{avg}$
# add all simulation plots with std?
plt.plot(sim_t_sine_norm,sim_f_sine_norm,'k',
          label=r'$F_{sim}$',zorder=1) #$sin(\omega_{fit2sim}t)$

# labels
# plt.title(r'normalized by fit to $f(t)=Asin(wt)+c$')
t1 = r'fit to $F_{avg}: $'
t2 = f'RMSE={res_avgf["rmse"]:.2e}, '
t3 = f'S={res_avgf["S"]:.2e}, '+ r'$R^2$='+f'{res_avgf["r_squared"]:.3f}'
# plt.title(t1+t2+t3)
plt.xlabel(r'$\frac{t}{T_{fit}}$',fontsize=fs)
plt.ylabel(r'$\frac{F-c_{fit}}{A_{fit}}$',fontsize=fs)
plt.xlim([-0.01,0.5])
plt.ylim([-0.1,1.2])
plt.legend(fontsize=fs)
savep=1
if savep==1:
    plt.savefig(figure_folder+
            r"\Paper\fit2sine\raw,f_avg,fits and simulation 2024_01_05.svg",
            dpi=200)

# r^2 scatter plot of fitted trajectories
rsq_dist = []
events_filter = []
plt.figure()
for i in range(N):
    if 'r_squared' in dir(events[i]):
        if m.isinf(events[i].r_squared):continue
        elif events[i].r_squared<0:continue
        else:
            rsq_dist.append(events[i].r_squared)
            plt.plot(i,events[i].r_squared,'x')
            if events[i].r_squared>0.9 and abs(events[i].r_squared)<1:
                events_filter.append(events[i])
            else:
                continue
                # events[i].r_squared = 0


# r^2 distribution of fitted trajectories
plt.figure()
plt.hist(rsq_dist,bins=50,range=[0,1],color=cmap(0.2))
plt.xlabel(r'$R^2$',fontsize=fs)
plt.ylabel(r'Frequency',fontsize=fs)
tmean = r'$R^{2}_{mean}$'
tmedian = r'$R^{2}_{median}$'
plt.title(tmean+f'={np.mean(rsq_dist):.2f}'+
          tmedian+f'={np.median(rsq_dist):.2f}',fontsize=14)
#%% figure - Force trajectory of 1 twine and 1 slip
#  change force to mN?
savep = 0
i_slip = 18
i_twine = 43
plt.figure()
plt.plot(events[i_slip].timer/60,np.multiply(events[i_slip].F_bean,mg2mN),
         color=cmap(0.2))
plt.plot(events[i_twine].timer/60,np.multiply(events[i_twine].F_bean,mg2mN),
         color=cmap(0.8))
plt.xlabel('time(min)',fontsize=fs)
plt.ylabel('Force(mN)',fontsize=fs)
# plt.legend(['twine event','slip event'])
plt.vlines(events[i_slip].timer[-1]/60, 0,
           events[i_slip].F_bean[-1]*mg2mN, color=cmap(0.2),
           linestyle='dashed',label='slip event')
plt.vlines(events[i_twine].timer[-1]/60, 0,
           events[i_twine].F_bean[-1]*mg2mN, color=cmap(0.8),
           linestyle='dashed',label='twine event')
plt.legend()

print("twine:"+ str(events[i_twine].twine_state==1))
print("slip:"+ str( events[i_slip].twine_state==0))

file_end = '\\jump vs expected.svg'
if savep == 1:
    plt.savefig(basepath + r"\figures\Paper\F4"+file_end)
#%% figure - plot median/mean of youngs modulus & or radius

all_E_dict = pd.DataFrame()
all_r_dict = pd.DataFrame()
for i in range(len(plants)):
    # try:
    e_dict = plants[i].avgE_sections
    Es = [x for x in e_dict.values()]

    sections = [x for x in e_dict.keys()]
    df_dict = pd.DataFrame(e_dict,index=[0])
    all_E_dict = pd.concat([all_E_dict, df_dict], ignore_index=True) # not ignore?

    r_dict = plants[i].r_sections
    Rs = [x for x in r_dict.values()]
    r_dict = pd.DataFrame(r_dict,index=[0])
    all_r_dict = pd.concat([all_r_dict, r_dict], ignore_index=True)

n = all_E_dict.count().values
fontscale = 2
fig,ax = plt.subplots(2,1) # ax[0]- linear fit to radius,
                           # ax[1]- parabiolic fit to Young's modulus
ax[0].plot(all_r_dict.mean().keys(),all_r_dict.mean().values,'xk')
ax[0].set_xlabel('$L-s(cm)$',fontsize=1.1*fs)
ax[0].set_ylabel(r'$\bar{R}(cm)$',fontsize=1.1*fs)
ax[0].errorbar(all_r_dict.mean().keys(),all_r_dict.mean().values,
                xerr=5/np.sqrt(12),yerr=all_r_dict.std(),fmt='o',)
ax[0].tick_params(axis='x',labelsize=14)
ax[0].tick_params(axis='y',labelsize=14)


ax[1].plot(all_E_dict.mean(),'xr')
ax[1].set_xlabel('$L-s(cm)$',fontsize=1.1*fs)
ax[1].set_ylabel(r'$\bar{E}(MPa)$',fontsize=1.1*fs)
ax[1].errorbar(all_E_dict.mean().keys(),all_E_dict.mean().values,
                xerr=5/np.sqrt(12),yerr=all_E_dict.std(),fmt='o',)
ax[1].tick_params(axis='x',labelsize=14)
ax[1].tick_params(axis='y',labelsize=14)

# divide errors by counts? /all_E_dict.count()

# all_E_dict.plot.box(widths=np.divide(n,sum(n)))

E_mean = all_E_dict.mean(skipna=True,numeric_only=True)
r_mean = all_r_dict.mean()

x_sections = [x for x in E_mean.index] # take mid point of sections
x_continuous = np.arange(x_sections[0],x_sections[-1],0.05) # 20 per cm
x_25cm = x_continuous[x_continuous<25]

# interp_E_cubic = scipy.interpolate.UnivariateSpline(x_sections, E_mean.values,k=3)
# interp_E_cubic_25cm = interp_E_cubic(x_25cm)
# ax[1].plot(x_25cm,interp_E_cubic_25cm,label='cubic spline')
# fig = plt.figure()
# plt.plot(x_continuous,interp_E_cubic(x_continuous),label = 'cubic interpolation')
# plt.xlabel('L-s(cm)',fontsize = fs)
# plt.ylabel('E(MPa)',fontsize = fs)

interp_r_mean = scipy.interpolate.UnivariateSpline(x_sections[1:], r_mean.values,k=1)
interp_r_continuous = interp_r_mean(x_continuous)
interp_r_mean_25cm = interp_r_mean(x_25cm)
a,b = interp_r_mean.get_coeffs()
ax[0].plot(x_continuous,interp_r_continuous,label=f'$y={b:.3f}(L-s)+{a:.3f}$')
ax[0].legend(loc='upper left',fontsize=14)


# get r^2 for r fit to linear
residuals = r_mean.values- interp_r_mean(x_sections[1:])
ss_res = np.sum(residuals**2)
# get total sum of squares (ss_tot):
ss_tot = np.sum((r_mean.values-np.mean(r_mean.values))**2)
# get r_squared-value:
r_squared = 1 - (ss_res / ss_tot)
print(f'{r_squared=} linear')


def para_func(x,a,b): #
    return a*x**2+b #parabula

popt,pcov = scipy.optimize.curve_fit(para_func,x_sections,
                   E_mean.values,bounds=([0,0],[10,1]))
interp_E_exp_25cm = para_func(x_25cm,*popt)
interp_E_exp_cont = para_func(x_continuous,*popt)
fit_std = np.sqrt(np.diag(pcov))

# get r^2 for fit to parabola
residuals = E_mean.values- para_func(np.array(x_sections),*popt)
ss_res = np.sum(residuals**2)
# get total sum of squares (ss_tot):
ss_tot = np.sum((E_mean.values-np.mean(E_mean.values))**2)
# get r_squared-value:
r_squared = 1 - (ss_res / ss_tot)
print(f'{r_squared=} parabola.')

# save for each plant the youngs modulus along s
for i in range(len(events)):
    events[i].E_s = para_func(events[i].p.s,*popt)
    # events[i].max_strain_t('top')

a,b = popt.round(3)
ax[1].plot(x_continuous,interp_E_exp_cont,label=fr'$y={a}(L-s)^{2}+{b}$')
ax[1].legend(loc='upper left',fontsize=14)
#%% figure - correlations of fitted amplitude

# add r^2 to plots
# check the high amplitude outliers?

N = len(events_filter)
mass = [0.2,0.65,0.86,3.1]

# initiate variables
file_names = ['m20cm','L_base','E_contact','T_cn','avgE','m_base','L_tip',
              'L_base.omega','L_base.m_base.omega','m_tip','Omega_cn',
              'L_base.m_base','L_arm','Density','twine_time']

x_name = [r'$m_{20cm}(gr)$',r'$L_{base}(cm)$',r'$E_{contact}(MPa)$',
        r'$T_{cn}(min)$', r'$\hat{E}(MPa)$',r'$m_{base}(gr)$',
        r'$L_{tip}(cm)$',
        r'$L_{base}*\omega_{CN}(\frac{cm}{sec})$',
        r'$L_{base}*m_{base}*\omega_{CN}(\frac{cm*gr}{sec})$',
        r'$m_{tip}(gr)$',r'$\omega_{CN}$',
        r'$L_{base}*m_{base}(cm*gr)$',r'$L_{arm}(cm)$',
        r'$Density(\frac{gr}{cm^3})$',r'$T_{twine}(min)$'] #r'$$'
y_name = r'$A_{fit}(mgf)$'

# x_range = [[0,3e2],[0,1e-1],[0,2.5e5],[0,3],[0,100],[0,8]]
# y_range = [0,100]

# choose a parameter to correlate with Fmax
# 0: m20cm
# 1: L_base
# 2: estimated E at contact point
# 3: Tcn
# 4: E_avg
# 5: m_base
# 6: L_tip
# 7: L_base * Omega_cn
# 8: L_base* m_base * Omega_cn
# 9: m_tip
# 10: Omega_cn
# 11: L_base * m_base
# 12: L_arm
# 13: Density
# 14: m_tot
# 15: twine_time
k0 = 14 # to plot vs only 1 parameter
savep = 0

for k in range(len(x_name)):
    if k!=k0: continue # plot only for parameter k0
    # if k not in [1,2,5,6,7,10]: continue # group of plots
    x = [[]]*N
    y = [[]]*N
    y_err = [[]]*N
    outliers = []
    for i in range(N):
        try:
            # data cutoffs:
            if events_filter[i].twine_state==0: continue
            # if events_filter[i].dec_time/events_filter[i].p.avgT<0.25: continue # 0.75*T_cn
            # if events_filter[i].dec_time/events_filter[i].p.avgT>1: continue # ~2*T_cn
            # if events_filter[i].L_contact2stemtip_cm>0.75*events_filter[i].p.arm_cm: continue #contact at far part of arm
            # if events_filter[i].fit_A>40:
            # if events_filter[i].fit_A<1: continue # should I remove these?
            # if events_filter[i].p.avgT>120: continue
            # if events_filter[i].L_contact2stemtip_cm<0.1: continue # very short
            # if events_filter[i].dec_time>events_filter[i].p.avgT*2: continue # very long
            # setup sets- # plot only for given masses
            # if events[i].p.m_sup not in [mass[2],mass[3]]: continue

            a,b = events_filter[i].p.ab_r
            l_all = events_filter[i].p.L0
            l_tip = events_filter[i].L_contact2stemtip_cm
            l_arm = events_filter[i].p.arm_cm
            vol_20cm = events_filter[i].p.vol20
            density = events_filter[i].p.density
            vol_arm = l_arm*np.pi*((1/3)*(a**2)*(l_arm**2)+(a*b*l_arm)+b**2)
            vol_tip = l_tip*np.pi*((1/3)*(a**2)*(l_tip**2)+(a*b*l_tip)+b**2)
            vol_base = l_all*np.pi*((1/3)*(a**2)*(l_all**2)+(a*b*l_all)+b**2) - vol_tip
            m_base = vol_base*density
            m_tip = vol_tip*density
            m_arm = vol_arm*density
            i_contact = uf.closest(events_filter[i].L_contact2stemtip_cm,events_filter[i].p.L_s)
            L_s_contact = events_filter[i].p.L_s[i_contact]
            E_L_s_contact = interp_E_exp_cont[uf.closest(x_continuous,L_s_contact)]
            L_base = events_filter[i].p.L0-events[i].L_contact2stemtip_cm
            omega = 2*np.pi/events_filter[i].p.avgT


            #  save outliers to different vector (not included in fit)
            if events_filter[i].fit_A>45 or events_filter[i].fit_A<1 or \
            events_filter[i].dec_time>events_filter[i].p.avgT*2 or \
            events_filter[i].dec_time<events_filter[i].p.avgT/2:
                outliers.append([E_L_s_contact,
                         events_filter[i].fit_A-events_filter[i].fit_offset,
                         events_filter[i].A_err])# force outliers
                x[i] = np.nan
                y[i] = np.nan
                y_err[i] = np.nan
                continue

            # y = fit amplitude
            y[i] = (events_filter[i].fit_A)#-events_filter[i].fit_offset) #* L_base/events_filter[i].p.m_sup # for torque?
            # y[i] = events[i].dec_time
            y_err[i] = events_filter[i].A_err#/(events[i].dec_time/60) # give smaller error for longer trajectory
            # y_err[i] = 4


            if k==0: x[i] = events_filter[i].p.m20cm
            elif k==1: x[i] = L_base
            elif k==2: x[i] = E_L_s_contact
            elif k==3: x[i] = events_filter[i].p.avgT
            elif k==4: x[i] = np.mean(list(events_filter[i].p.avgE_sections.values()))
            elif k==5: x[i] = m_base
            elif k==6: x[i] = l_tip
            elif k==7: x[i] = L_base*omega
            elif k==8: x[i] = L_base*m_base*omega
            elif k==9: x[i] = m_tip
            elif k==10: x[i] = omega
            elif k==11: x[i] = L_base * m_base
            elif k==12: x[i] = l_arm # or m_arm
            elif k==13: x[i] = density
            elif k==14: x[i] = events_filter[i].dec_time
            # print(i,x[i],y[i])



        except Exception as e:
            x[i] = np.nan
            y[i] = np.nan
            y_err[i] = np.nan
            print(f'error: {e}')
            continue

    # remove [] & NaNs
    x1, y1, y1_err = zip(*[(a, b, c) for a, b,c  in zip(x, y,y_err)
          if not (np.isnan(a) or (isinstance(a, list) and not np.array(a).size>0) or a == 0
              or np.isnan(b) or (isinstance(b, list) and not np.array(b).size>0) or b == 0)])
    # np.array(a).size , len(a)

    # into dataframe
    xy_df = pd.DataFrame({'x':x1,'y':y1,'y_err':y1_err})#.dropna()
    xy_df = xy_df.sort_values(by='x') # sort

    # Extract x and y values as arrays
    x = xy_df['x'].values
    y = xy_df['y'].values
    y_err = xy_df['y_err'].values

    # initiate plot Figure
    fig,ax = plt.subplots(figsize=(8, 6))
    # xticks = np.arange(min(x1), max(x1) * 1.2, max(x1) / 4)
    xticks = np.linspace(min(x1),max(x1),5)
    ax.set_xticks(xticks)
    import matplotlib.ticker as ticker
    formatter = ticker.FormatStrFormatter('%.4g')
    ax.xaxis.set_major_formatter(formatter)
    plt.xlabel(x_name[k],fontsize=fs)
    plt.ylabel(y_name,fontsize=fs)

    # plot values (and outliers) with errors
    ax.errorbar(x, y, yerr=y_err, fmt='+',color='b', label='Data')
    ax.errorbar([v[0] for v in outliers],[v[1] for v in outliers],
                [v[2] for v in outliers],
                fmt='+',color='grey', label='Outliers')

    # fit with errors via curve fit
    def f_lin(x,a,b): return a*x + b
    errx = np.array([0/20 for i in np.arange(len(x))])
    erry = np.array(y_err)
    popt,pcov = scipy.optimize.curve_fit(f_lin, x, y,
                   p0=None, sigma=np.sqrt(errx**2+erry**2), absolute_sigma=True)
    # r squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    line_of_best_fit = popt[0] * x + popt[1]
    residuals = y - line_of_best_fit
    ss_residual = np.sum(residuals**2)
    chi_squared = np.sum(residuals**2 / y_err**2)/(len(x)-3)
    chi_squared_std = np.sqrt(2/(len(x)-3))
    r_squared = 1 - (ss_residual / ss_total)
    plt.plot(x,popt[0]*x+popt[1],'r-',
             label=rf'$\chi^{2}={chi_squared:.3f}\pm{chi_squared_std:.3f}, R^{2}: {r_squared:.3f}$')
    upper =(popt[0]+np.diag(pcov)[0])*x+popt[1]+np.diag(pcov)[1]
    lower = (popt[0]-np.diag(pcov)[0])*x+popt[1]-np.diag(pcov)[1]
    # plt.fill_between(np.arange(min(x),max(x),(max(x)-min(x))/len(x)),lower,upper)

    plt.title(rf'$ax+b = ({popt[0]:.4f}\pm {np.diag(pcov)[0]:.4f}) x + ({popt[1]:.4f}\pm {np.diag(pcov)[1]:.4f})$')
    plt.legend(fontsize=fs,loc='upper left')
 
    plt.show()
    file_end = '\\'+file_names[k]+'.svg'
    if savep == 1:
        plt.savefig(basepath+r"\figures\Paper\Amplitude"+file_end)
#%% figure - discrepency: fit_period-normalized F(t) vs CN-normalized F(t)
# experimental force trajectories normalized by experimental Tcn
# simulation force trajectory normalized by Tcn parameter of simulation
savep = 0
fig,ax = plt.subplots(figsize=(8, 6))#,dpi=150)
allf = []
allt = []
tmax = 0
allf_interp_Tcn = [] # all forces interpolated with Tcn normalized times
for i in range(N):
    try:
        # interp with Tcn
        f = uf.remove_outliers(events[i].F_bean,0.75)
        l = len(f)
        if len(f)<5: continue
        t = events[i].timer
        time[i] = np.divide(t[:events[i].n_Tcn],60) # minutes
        interp_f = scipy.interpolate.UnivariateSpline( time[i]/(events[i].p.avgT),
                       (f[:events[i].n_Tcn]-f[0])/(events[i].fit_A),k=2) # interpolate
        # print(f"{i} interpolated")
        normtime = np.arange(time[i][0]/(events[i].p.avgT),
                             time[i][-1]/(events[i].p.avgT),0.001) # get normalized time stamps
        tmax = max(tmax,max(normtime))
        allf_interp_Tcn.append(interp_f(normtime))
        print(i)
        allf.append(np.array(f[:events[i].n_Tcn]-f[0]))
        # plt.plot(normtime,interp_f(normtime))
        allt.append(np.array(time[i]/(events[i].p.avgT)))
        # plt.plot(time[i]/(events[i].p.avgT),f[:events[i].n_Tcn]-f[0],'b',alpha=0.2)
    except Exception as e:
        print(f'error: {e}')
allf_interp_Tcn = pd.DataFrame(allf_interp_Tcn)
avgf = allf_interp_Tcn.mean()
stdf = allf_interp_Tcn.std()
stdf.iloc[-1] = stdf.iloc[-2]
l = len(avgf)
allnormtime = np.arange(0,tmax+0.1,0.001)
minus_std = scipy.signal.savgol_filter(avgf-stdf,
                                           int(0.2*l)+int(0.2*l)%2+1, 3)
plus_std = scipy.signal.savgol_filter(avgf+stdf,
                                           int(0.2*l)+int(0.2*l)%2+1, 3)
avgf = scipy.signal.savgol_filter(avgf,int(0.2*l)+int(0.2*l)%2+1, 3)

plt.plot(allnormtime[:len(avgf)],avgf,'b',label='experiments')
plt.fill_between(allnormtime[:len(avgf)],
             minus_std, plus_std,alpha=0.5,zorder=3)

plt.xlabel(r'$t/T_{CN}$',fontsize=fs)
plt.ylabel(r'$F/A_{fit}$',fontsize=fs)

# plot simulation and pure sine - redo!
sim_t_sine_norm = np.divide(sim_t,period)
sim_f_sine_norm = np.divide(sim_f,Amplitude)
cutoff = uf.closest(sim_t_sine_norm,0.8)
plt.plot(sim_t_sine_norm[:cutoff],sim_f_sine_norm[:cutoff],
         '-',color='red',label='simulation')


plt.xticks(ticks=np.arange(0,0.75,0.25),fontsize=fs-5)
plt.yticks(ticks=np.arange(0,1.25,0.25),fontsize=fs-5)
plt.legend()

file_end = '\\cn rate delay.svg'
if savep == 1:
    plt.savefig(basepath + r"\figures\Paper\F4"+file_end,dpi=200)
#%% figure - experimental&simulation force trajectories normalized by Tcn
savep = 0
fig,ax = plt.subplots(figsize=(8, 6))#,dpi=150)
allf = []
allt = []
tmax = 0
allf_interp_Tcn = [] # all forces interpolated with Tcn normalized times
for i in range(N):
    try:
        # interp with Tcn
        f = uf.remove_outliers(events[i].F_bean,1)
        l = len(f)
        if len(f)<5: continue
        t = events[i].timer
        time[i] = np.divide(t[:events[i].n_Tcn],60) # minutes
        interp_f = scipy.interpolate.UnivariateSpline( time[i]/(events[i].p.avgT),
                       (f[:events[i].n_Tcn]-f[0])/(events[i].fit_A),k=2) # interpolate
        # print(f"{i} interpolated")
        normtime = np.arange(time[i][0]/(events[i].p.avgT),
                             time[i][-1]/(events[i].p.avgT),0.001) # get normalized time stamps
        tmax = max(tmax,max(normtime))
        allf_interp_Tcn.append(interp_f(normtime))
        print(i)
        allf.append(np.array(f[:events[i].n_Tcn]-f[0]))
        # plt.plot(normtime,interp_f(normtime))
        allt.append(np.array(time[i]/(events[i].p.avgT)))
        # plt.plot(time[i]/(events[i].p.avgT),f[:events[i].n_Tcn]-f[0],'b',alpha=0.2)
    except:
        print(f'problem at {i}')
allf_interp_Tcn = pd.DataFrame(allf_interp_Tcn)
avgf = allf_interp_Tcn.mean()
stdf = allf_interp_Tcn.std()

allnormtime = np.arange(0,tmax+0.1,0.001)
plt.plot(allnormtime[:len(avgf)],avgf,'b',label='$F_{avg}$')
plt.fill_between(allnormtime[:len(avgf)],
             avgf-stdf, avgf+stdf,alpha=0.5,zorder=3)
plt.xlabel(r'$t/T_{CN}$',fontsize=fs)
plt.ylabel(r'$F/A_{fit}$',fontsize=fs)

# plot simulation and pure sine
plt.plot(sim_t_sine_norm[:cutoff],sim_f_sine_norm[:cutoff],
         color='red',label='simulation')
plt.plot(x_range[:cutoff], uf.sinfunc(x_range, 1,2*np.pi,0)[:cutoff],
         '--k',label=r'$sin(2\pi t)$')

plt.xticks(ticks=np.arange(0,1.25,0.25),fontsize=fs-7)

plt.legend()

file_end = '\\cn rate delay.svg'
if savep == 1:
    plt.savefig(basepath + r"\figures\Paper\F4"+file_end,dpi=200)
#%% figure - twine vs slip distributions
savep = 0 # = 1 to savecv figure

var_slip = []
var_twine = []
var_name = [r'$\int\epsilon_{max}dt$ (sec)',r'$\epsilon_{max}$',
            r'$\int Fdt (mgf*sec)$', r'$T_{dec}/T_{cn}$','W(Fmg*cm)',
            r'$l_{tip} (cm)$','$F_{max}$ (mgf)',r'$\tau_{max}(mgf*cm)$',
            r'$\int \tau dt(mgf*cm*s)$']
x_range = [[0,2e2],[0,0.5e-1],[0,2.5e5],[0,2.5],
           [0,40],[0,8],[0,60],[0,4e2],[0,3e6]]
value_name = 'frequency' # density
n_tot = len(events)
n_tests = int(5e5)
n_bins = 20
# choose a comparison
# 0: integrated strain
# 1: max strain
# 2: integrated force
# 3: normalized decision time
# 4: work
# 5: contact distance from stem tip
# 6: max force until dec
# 7: max torque
# 8: integrated torque
k = 7
# compare: integ_f, alpha, F_bean, L_contact2stemtip_cm,
# F1stmax , L_base,dec_time, p.avgT, int_eps

# save values to 2 lists
for i in range(n_tot):
    # cutoffs
    try:
        if k==0: var = events[i].int_eps
        elif k==1: var = max(events[i].eps_t)
        elif k==2: var = events[i].integ_f
        elif k==3: var = events[i].dec_time/events[i].p.avgT
        elif k==4: var = events[i].work
        elif k==5: var = events[i].L_contact2stemtip_cm
        elif k==6: var = max(events[i].F_bean)
        elif k==7: var = max(events[i].torque)
        elif k==8: var = events[i].integ_torque

    except:
        continue

    if events[i].twine_state == 1:
        var_twine.append(var)# set variable to plot
    else:
        var_slip.append(var)

# create pandas df
df1 = pd.DataFrame(var_twine,columns=['twine'])
df2 = pd.DataFrame(var_slip,columns=['slip'])
df = pd.concat([df1,df2], axis=1)

# compare two distributions
# permutation test

# perm_pval = uf.permutation_test(var_twine, var_slip, n_tests,x_range[k],
#                             'permutation differences for '+var_name[k])
# ttest_pval,ks_pval,mises_pval,manwhiteney_pval = uf.non_parametric(df1,df2)

# pandas histogram
hist_twine = np.histogram(df1,bins=n_bins,range=x_range[k])
hist_slip = np.histogram(df2,bins=n_bins,range=x_range[k])

df.plot.hist(stacked=False, bins=n_bins, figsize=(6, 6),
             grid=False,range=x_range[k],density=False,
             color = [cmap(0.8),cmap(0.2)],alpha = 0.65)
plt.xlabel(var_name[k], fontsize=fs)
plt.ylabel('Frequency', fontsize=fs)
wasserstein = scipy.stats.wasserstein_distance(var_slip, var_twine,
       u_weights=None, v_weights=None)/(np.mean(var_twine)+np.mean(var_slip))
JS = sci_distance.jensenshannon(hist_twine[0], hist_slip[0]) ** 2

# print('wasserstein/sum(means)): ' + var_name[k]+ ' ' +str(wasserstein))
plt.title(f'a {wasserstein=:.3f} and {JS=:.3f}')
# plt.title(f"permutation pval ={perm_pval:.2e} ,ks pval= {ks_pval:.2e},\
# von-mises = {mises_pval:.2e},man whiteney pval = {manwhiteney_pval:.2e}")

# plt.legend(label)
    # df.plot.density()
# df_norm.plot.hist(stacked=False,bins=100,range=x_range[k],density=False,alpha=0.8)

file_end = '\\xxx.svg'
if savep == 1:
    plt.savefig(basepath + r"\figures\Paper\S2"+file_end)

# boxplot
fig,ax = plt.subplots(1,1)
colors = ['pink', 'lightblue']
dfboxplot = ax.boxplot(df.dropna(),notch=True,patch_artist=True,labels=['twine','slip'])
for patch, color in zip(dfboxplot['boxes'], colors):
    patch.set_facecolor(color)
KS = kruskal(var_twine,var_slip)
plt.title(f'Kruskall-Wallis = {KS[1]:.2e}')
#%% figure - plot T_cn distribution
plt.figure(figsize=(8, 6))
a = []
nbin = 15
for i in range(len(events)):
    a.append(events[i].p.avgT)
    print(events[i].p.avgT)
    # a.append(events[i].p.avgT/events[i].p.start_height**0.5)
bin_range = [70,130]
# bin_range=None

n = plt.hist(a,bins=nbin,range=bin_range,color = cmap(0.2),alpha=0.7)
plt.xlabel(r'$\hat{T}_{CN}(min)$',fontsize=fs)
plt.ylabel(r'$Frequency$',fontsize=fs)
plt.vlines(np.mean(a), 0, max(a),'k')
plt.ylim([0,25])
# plt.savefig(basepath + r"\figures\Paper\CN_dist.svg",dpi=200)

#%% figure 9.1 - CN delay at jump frame
CN_delay_path = jump_root_path + r'\first_last_plusXmin\img_angles_merged.xlsx'
jump_panda = pd.read_excel(CN_delay_path)

# clean non-valued entries
# 't_contact(sec)' is contact duration
# 'delay(sec)' is elapsed time after the jump moment
# df_cleaned = jump_panda.dropna(subset=['t_contact+delay(sec)','t_contact(sec)'])
# new_row['delay(sec)']
# filter by values in specific columns
# df[(df['exp'] == int(exp)) & (df['event'] == int(event))]
# Adding a new column which is the sum of 'X' and 'Y'
# df['Sum_XY'] = df['X'] + df['Y']
# use groupby and apply
# total time after contact initiation
jump_panda['dec_time+delay(sec)'] = jump_panda['t_contact(sec)']+jump_panda['delay(sec)']


# error estimate: atan(d/l) where d is error of point selection for point in image
# and l is the distance between the 2 points making the line.
# get image resolutions and save to the jump_panda


# Specify the path to the folder containing images
# folder_path = root_path + r''
# extract data from all files in folder
# image_data_df = uf.extract_image_resolution(folder_path)

# add column to jump_panda according to the exp event t_contact delay parameters


# difference of measured vs expected as function of delay/t_contact(not including jump)
fig,ax = plt.subplots(3,1,sharex=True)
y_range = [-10,270]
no_jump = jump_panda[jump_panda['delay(sec)']>0]
y_label = r'$\Delta\phi(rad)$'
x_label = r'$\frac{t_{contact}}{T_{CN}}$'
plot_title = 'difference from expected angle'

# difference of measured vs expected as function of t_contact(only jump)
only_jump = jump_panda[jump_panda['delay(sec)']==0]

# ax[0].plot(only_jump['t_contact(sec)']/only_jump['avg_cn_period(sec)'],
#           np.rad2deg(only_jump['expected_angle(rad)']-only_jump['base2tip(rad)']),'xb')
uf.plot_definitions(fig=fig,ax=ax[0],xlabel=x_label,sharex=True,suptitle=plot_title,
                    ylabel=y_label+r'$_{base2tip}$',yrange=y_range )
# ax[1].plot(only_jump['t_contact(sec)']/only_jump['avg_cn_period(sec)'],
#           np.rad2deg(only_jump['expected_angle(rad)']-only_jump['tip_lin(rad)']),'xb')
uf.plot_definitions(fig=fig,ax=ax[1],xlabel=x_label,sharex=True,
                    ylabel=y_label+r'$_{tip-lin}$',yrange=y_range )
# ax[2].plot(only_jump['t_contact(sec)']/only_jump['avg_cn_period(sec)'],
#           np.rad2deg(only_jump['expected_angle(rad)']-only_jump['base_lin(rad)']),'xb')
uf.plot_definitions(fig=fig,ax=ax[2],xlabel=x_label,sharex=True,
                    ylabel=y_label+r'$_{base-lin}$',yrange=y_range )

# linear fits and overlayed plots
x = only_jump['t_contact(sec)']/only_jump['avg_cn_period(sec)']


# base2tip
y = np.rad2deg(only_jump['expected_angle(rad)']-only_jump['base2tip(rad)'])
errx = np.array([0.01 for i in np.arange(len(y[y>0]))])
erry = np.array([5  for i in np.arange(len(y[y>0]))])
fit1 = uf.fit_w_err(ax[0],x[y>0],errx,y[y>0],erry,legend_loc='upper left')
# tip-linear
y = np.rad2deg(only_jump['expected_angle(rad)']-only_jump['tip_lin(rad)'])
errx = np.array([0.01 for i in np.arange(len(y[y>0]))])
erry = np.array([5 for i in np.arange(len(y[y>0]))])
fit2 = uf.fit_w_err(ax[1],x[y>0],errx,y[y>0],erry,legend_loc='upper left')
# base-linear
y = np.rad2deg(only_jump['expected_angle(rad)']-only_jump['base_lin(rad)'])
errx = np.array([0.01 for i in np.arange(len(y[y>0]))])
erry = np.array([1 for i in np.arange(len(y[y>0]))])
fit3 = uf.fit_w_err(ax[2],x[y>0],errx,y[y>0],erry,legend_loc='upper left')

print('yeh')


#%% figure 10.1 - CN delay decay after different contact duration
#!!! remember- the error for the contact duration will be the delta_t of pictures
# to identify when the jump occured exactly, and the angle will depend on the
# natural CN rate times that delta_t

# total time after contact initiation
jump_panda['t_contact+delay(sec)'] = jump_panda['t_contact(sec)']+jump_panda['delay(sec)']

# difference of measured vs expected as function of delay/t_contact(not including jump)
fig,ax = plt.subplots(3,1,sharex=True)
y_range = [-10,270]
no_jump = jump_panda[jump_panda['delay(sec)']>0]
x_label = r'$\frac{t_{delay}}{t_{contact}}$'
y_label = r'$\Delta\phi(rad)$'
plot_title = 'difference from expected angle'
#base2tip
ax[0].plot(no_jump['delay(sec)']/no_jump['t_contact(sec)'],
         np.rad2deg(no_jump['expected_angle(rad)']-no_jump['base2tip(rad)']),'xb')
uf.plot_definitions(fig=fig,ax=ax[0],xlabel=x_label,sharex=True,
                    ylabel=y_label+r'$_{base2tip}$',suptitle=plot_title,yrange=y_range)
#tip-linear
ax[1].plot(no_jump['delay(sec)']/no_jump['t_contact(sec)'],
         np.rad2deg(no_jump['expected_angle(rad)']-no_jump['tip_lin(rad)']),'xb')
uf.plot_definitions(fig=fig,ax=ax[1],xlabel=x_label,sharex=True,
                    ylabel=y_label+r'$_{tip-lin}$',yrange=y_range )
# base-linear
ax[2].plot(no_jump['delay(sec)']/no_jump['t_contact(sec)'],
         np.rad2deg(no_jump['expected_angle(rad)']-no_jump['base_lin(rad)']),'xb')
uf.plot_definitions(fig=fig,ax=ax[2],xlabel=x_label,sharex=True,
                    ylabel=y_label+r'$_{base-lin}$',yrange=y_range )

# plot 5 'delay' trajectories from 1 plant
expa = jump_panda[jump_panda['exp']==11]
eventa_s = expa[expa['event']==8]
eventa_s1 = expa[expa['event']==7]
eventa_s2 = expa[expa['event']==6]
eventa_s3 = expa[expa['event']==5]
eventa_s4 = expa[expa['event']==4]
ev_a = [eventa_s,eventa_s1,eventa_s2,eventa_s3,eventa_s4]

# plot all trajectories from 1 plant with 1 angle
# select angle
ang = r'base2tip(rad)'
# select plant
p = 17
exp_a = jump_panda[jump_panda['exp']==p]
color = iter(cm.rainbow(np.linspace(0, 1, max(exp_a['event']))))
for i in range(1,max(exp_a['event'])+1):
    c = next(color)
    ev_a = exp_a[exp_a['event']==i]
    ax[2].plot(ev_a['delay(sec)']/ev_a['t_contact(sec)'],
        np.rad2deg(ev_a['expected_angle(rad)']-ev_a[ang]),
        '-x',alpha=0.5,color=c,
        label=rf'{ev_a["t_contact(sec)"].iloc[0]}sec contact')
uf.plot_definitions(fig=fig,ax=ax[2],xlabel=x_label,sharex=True,
                    ylabel=y_label+r'$_{tip-lin}$',legend=True)

# plot all trajectories for each plant, color per plant, for given angle.
color = iter(cmap(np.linspace(0, 1, max(jump_panda['exp']))))
# fig,ax = plt.subplots(1,1)
uf.plot_definitions(xlabel=x_label,ylabel=y_label,title=f'{ang}')
for i in range(1,max(jump_panda['exp'])):
    c = next(color)
    if i != 7: exp_i = jump_panda[jump_panda['exp']==i]
    for k in range(1,max(exp_i['event'])+1):
        ev_i_k = exp_i[exp_i['event']==k]
        # plot raw angles
        # plt.plot(ev_i_k['delay(sec)']/ev_i_k['t_contact(sec)'],
        #     np.rad2deg(ev_i_k['expected_angle(rad)']-ev_i_k[ang]),
        #     '-x',alpha=0.3,color=c)
        # plot angles starting at zero
        y = 1*np.rad2deg(ev_i_k['expected_angle(rad)']-ev_i_k[ang]) # (ev_i_k['t_contact(sec)'])
        if len(y)>1:
            plt.plot(ev_i_k['delay(sec)'], # to normalize by dec time: x/ev_i_k['t_contact(sec)']
                 y-y.dropna().iloc[0],'-x',alpha=0.3,color=c)

# plot all trajectories, color according to t after jump, for given angle.
# set plot parameters
ang = r'tip_lin(rad)'
x_label = r'$\frac{t}{T_{CN}}$'
# y_label = r'$\frac{\phi_{expected}}{\phi_{measured}}$'
y_label = r'$\Delta\phi(rad)$'
uf.plot_definitions(xlabel=x_label,ylabel=y_label,title=f'{ang}')

# normalize color range
norm_color = plt.Normalize(jump_panda['t_contact(sec)'].min(),
                           jump_panda['t_contact(sec)'].max())

# colorbar to show the mapping
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
sm.set_array([])
plt.colorbar(sm, label='time after jump(sec)')

# loop over experiments and events per experiment
for i in range(1,max(jump_panda['exp'])):
    if i != 7: exp_i = jump_panda[jump_panda['exp']==i]
    for k in range(1,max(exp_i['event'])+1):
        ev_i_k = exp_i[exp_i['event']==k] # select event
        c = cmap(norm_color(ev_i_k['t_contact(sec)'].iloc[0])) # select color by contact time
        # plot angles starting at zero
        # y = 1*np.rad2deg(ev_i_k['expected_angle(rad)']-ev_i_k[ang])

        # plot angles ratio or difference
        y = (ev_i_k['expected_angle(rad)']-ev_i_k[ang])
        if len(y)>1:
            plt.plot(ev_i_k['delay(sec)']/ev_i_k['avg_cn_period(sec)'],
                     y,'-x',alpha=0.5,color=c)

# try plot with pivot_table ?
# jump_panda.pivot_table(values='expected_angle(rad)',index='delay(sec)',
#                        columns='exp').plot()

#%% ploting functions

# if condition-> replace measured with 2pi-measured.
def twopi_condition(df,meas,expt):
    '''define condition for subtracting angle from 2pi'''
    con =  abs(df[meas]/df[expt])
    # condition_met = con > np.pi
    # df.loc[condition_met,meas] = 2*np.pi-df.loc[condition_met,meas]
    # Print the condition for debugging
    # print("Condition array:\n", con)

    condition_met = con > 2
    print("Rows where condition is met (only meas and expt columns):\n", df.loc[condition_met, [meas, expt]])

    # Update only the rows where the condition is met
    df.loc[condition_met, meas] = 2 * np.pi - df.loc[condition_met, meas]

    # Print DataFrame after modification for verification (only meas and expt columns)
    print("DataFrame after modification (only meas and expt columns):\n", df.loc[condition_met, [meas, expt]])

# twopi_condition(jump_panda,'base2tip(rad)','expected_angle(rad)')
# twopi_condition(jump_panda,'tip_lin(rad)','expected_angle(rad)')
# twopi_condition(jump_panda,'base_lin(rad)','expected_angle(rad)')

def fit_w_err(ax,x,dx,y,dy,fit_func = linfunc, data_color = 'blue',
              fit_color = 'red',add_legend = False, legend_loc='best'):
    '''Fit data to the provided function and plot with error bars.

    Parameters:
    ax : matplotlib axis object
    x, dx : array-like, data and errors in x
    y, dy : array-like, data and errors in y
    fit_func : callable, function to fit the data
    data_color : str, color for the data points
    fit_color : str, color for the fitted curve
    legend_loc : str, location for the legend
    to_print : bool, whether to print function parameters on the plot

    Returns:
    popt : array, optimal values for the parameters
    pcov : 2D array, the estimated covariance of popt
    goodness of fit: R^2 and chi^2_red
    '''
    # Plot the data with error bars
    if add_legend: label_data='Experiements'
    else: label_data=None
    ax.errorbar(x, y, xerr=abs(dx), yerr=abs(dy), fmt='+',
                color=data_color,label=label_data)
    # fit with errors via curve fit
    popt,pcov = scipy.optimize.curve_fit(fit_func, x, y,
            p0=None, sigma=np.sqrt(dx**2+dy**2), absolute_sigma=True)
    # ax.plot(x,(fit_func(x,*popt)),color=fit_color,
    #         label='fit')#fit_func(x,*popt))

    # goodness of fit
    ss_total = np.sum((y - np.mean(y)) ** 2)
    line_of_best_fit = popt[0] * x + popt[1] # fit_func(x, *popt)
    residuals = y - line_of_best_fit
    ss_residual = np.sum(residuals**2)
    chi_squared_red = np.sum(residuals**2 / (dy**2+dx**2))/(len(x)-3)
    chi_sr_std = np.sqrt(2/(len(x)-3))
    r_squared = 1 - (ss_residual / ss_total)
    ax.plot(x,popt[0]*x+popt[1],'-',color=fit_color,
             label=rf'y = {popt[0]:.2f}x + {popt[1]:.2f},\
             $\chi^{2}={chi_squared_red:.2f}\pm {chi_sr_std:.2f}, R^{2}: {r_squared:.2f}$')
    if add_legend: ax.legend(loc=legend_loc)

    # plot residuals
    # plt.figure()
    # plt.errorbar(x, y-fit_func(x,*popt), yerr=dy, fmt='x',color='black')

    return popt,pcov#,r_squared,chi_squared_red
    
# Function for plotting and fitting
def plot_and_fit(ax, df, angle_column,y_label_suffix, sharex=True, data_color='blue'):
    x = df['t_contact(sec)'] / df['avg_cn_period(sec)']
    # y = np.rad2deg(df['expected_angle(rad)'] - df[angle_column])
    y = df['expected_angle(rad)'] - df[angle_column]
    valid_indices = y > 0
    errx = np.ones(len(y[valid_indices]))*0.01
    erry = np.ones(len(y[valid_indices])) * 0.15  # adjust error as needed

    uf.plot_definitions(fig=fig, ax=ax, xlabel=x_label,sharex=sharex,
                        ylabel=y_label + y_label_suffix,yrange=y_range)
    return fit_w_err(
                    ax, x[valid_indices], errx, y[valid_indices], erry,
                    data_color = data_color,fit_color = 'red',add_legend=True)
#%% figure 9.2 - CN delay at jump frame
CN_delay_path = r"\\132.66.44.238\AmirOhad\PhD\Experiments\Force Measurements\Exp2b_CN_Delay\Measurements\first_last_plusXmin\img_angles_merged.xlsx"
jump_panda = pd.read_excel(CN_delay_path)

# Prepare data for plots
only_jump = jump_panda[jump_panda['delay(sec)'] == 0]
# y_range = [-10, 270]
y_range = [-0.5, 5]
y_label = r'$\Delta\theta$'
x_label = r'$\frac{t_{contact}}{T_{CN}}$'
plot_title = 'Difference from Expected Angle'

fig, ax = plt.subplots(3, 1, sharex=True)
fig.suptitle(plot_title)

# Plotting and fitting
fit1 = plot_and_fit(ax[0], only_jump, 'base2tip(rad)', r'$_{base2tip}$')
fit2 = plot_and_fit(ax[1], only_jump, 'tip_lin(rad)', r'$_{tip-lin}$')
fit3 = plot_and_fit(ax[2], only_jump, 'base_lin(rad)', r'$_{base-lin}$')
# add exp 26 - 31
new = only_jump[only_jump['exp']>18]
fit1 = plot_and_fit(ax[0], new, 'base2tip(rad)', r'$_{base2tip}$',data_color='black')
fit2 = plot_and_fit(ax[1], new, 'tip_lin(rad)', r'$_{tip-lin}$',data_color='black')
fit3 = plot_and_fit(ax[2], new, 'base_lin(rad)', r'$_{base-lin}$',data_color='black')
#%% only base2tip angle
fig,ax = plt.subplots(1,1)
fig.suptitle('Base to Tip angle difference after slip')
fit1 = plot_and_fit(ax, only_jump, 'base2tip(rad)', r'$_{base2tip}$ (rad)',sharex=False)


#%% 