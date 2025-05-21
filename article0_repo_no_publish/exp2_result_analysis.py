"""
Created on Thuesday June 18 14:38:45 2024

different plots for exp 2

@author: Amir Ohad
"""
#%% import libraries
import numpy as np
import matplotlib.pyplot as plt

# import importlib
# import useful_functions as uf # my functions

importlib.reload(uf)
#%% import data
# sdfdas
#%%
# sdf

#%% figure 10.2 - with groupby (chat-gpt)

#angle fixes: if CN_rate*t > 5*angle: subtract 2 pi?
# if CN_rate < angle/5: add 2 pi?

# try for each angle type:

def choose_xy(group):
    """
    Select x and y data for plotting from the group.

    Args:
    group (DataFrame): Grouped DataFrame for a single trajectory.

    Returns:
    tuple: x and y data for plotting.
    """
    # choose angle
    ang = ['base_lin(rad)','base2tip(rad)','tip_lin(rad)'] # angle options
    angle = ang[1] # choose which angle type to use
    c = 3# choose plot type
    x_label = r'$\frac{t}{T_{CN}}$'


    # difference of measured from expected vs t/contact_time(w/o jump angle)
    if c == 1:
        x = group['delay(sec)'] / group['avg_cn_period(sec)']
        # if group[angle].iloc[0]>2 *m.pi:
        #     group[angle] = group[angle] - 2*m.pi
        y = group['expected_angle(rad)'] - group[angle]
        y_label = r'$\Delta\phi(rad)$'
    # as 1, but start from angle 0
    elif c == 2:
        x = group['delay(sec)'] / group['avg_cn_period(sec)']
        y = group['expected_angle(rad)'] - group[angle]
        # Drop NaN values from y and get the indices of the remaining rows
        non_nan_indices = y.dropna().index

        # Use these indices to filter both x and y
        x = x.loc[non_nan_indices]
        y = y.loc[non_nan_indices]

        # Subtract the first value of y from all values in y
        if not y.empty:
            y = y - y.iloc[0]
        y_label = r'$\Delta\phi(rad)-\phi_0$'
    # difference of measured from expected normalized by CN_period
    elif c == 3:
        x = group['delay(sec)'] / group['avg_cn_period(sec)']
        y = (group['expected_angle(rad)'] - group[angle])/group['avg_cn_period(sec)']
        y_label = r'$\frac{\Delta\phi(rad)}{T_{CN}}$'
    return x, y, x_label, y_label, angle

def plot_trajectory(x, y, color_param, cmap, norm_color):
    """
    Plot a single trajectory.

    Args:
    x (Series): x data for plotting.
    y (Series): y data for plotting.
    color_param (float): Parameter to determine color.
    cmap (Colormap): Matplotlib Colormap for coloring the trajectories.
    norm_color (Normalize): Matplotlib Normalize instance for scaling colors.
    """
    color = cmap(norm_color(color_param))
    if len(y) > 1:
        plt.plot(x, y, '-x', alpha=0.5, color=color)

def plot_all_trajectories(jump_panda, cmap):
    """
    Plot all trajectories using groupby and apply method.

    Args:
    jump_panda (DataFrame): Pandas DataFrame containing the trajectory data.
    cmap (Colormap): Matplotlib Colormap for coloring the trajectories.
    # """
    # x_label = r'$\frac{t}{T_{CN}}$'
    # y_label = r'$\Delta\phi(rad)$'
    # angle_label = 'tip_lin(rad)'
    # uf.plot_definitions(xlabel=x_label, ylabel=y_label, title=angle_label)
    # Tcn = jump_panda['avg_cn_period(sec)']
    norm_color = plt.Normalize(jump_panda['t_contact(sec)'].min(),
                               jump_panda['t_contact(sec)'].max())

    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    scalar_map.set_array([])
    plt.colorbar(scalar_map, label='contact time(sec)')

    filtered_data = jump_panda[jump_panda['exp'] != 7]
    grouped = filtered_data.groupby(['exp', 'event'])

    for ind, group in grouped:
        x, y, x_label, y_label, angle_label = choose_xy(group)
        plot_trajectory(x, y, group.iloc[0]['t_contact(sec)'], cmap, norm_color)

    ax = plt.gca()
    uf.plot_definitions(ax = ax, xlabel=x_label, ylabel=y_label, title=angle_label) # test
# Example usage
# plot_all_trajectories(jump_panda, cmap)
plt.figure()
plot_all_trajectories(jump_panda,cmap)
#%% CN_Delay 3D dot-line plot:
# plot delta Phi along a t_contact/T_cn axis and a t_delay/T_cn axis.

# choose which angle type to use
ang = ['base_lin(rad)','base2tip(rad)','tip_lin(rad)'] # angle options
angle = ang[2]

# Assuming jump_panda is your DataFrame
# Replace 'exp_column' and 'event_column' with the actual column names for experiment and event

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Normalize t_contact for colormap
norm_color = plt.Normalize(jump_panda['t_contact(sec)'].min(), jump_panda['t_contact(sec)'].max())
# norm_color = plt.Normalize(jump_panda['exp'].min(), jump_panda['exp'].max())

cmap = cm.viridis # magma, plasma,

# Iterate through each combination of experiment and event
for (exp, event), group in jump_panda.groupby(['exp', 'event']):
    X = group['t_contact(sec)'] / group['avg_cn_period(sec)']
    # print(group['exp'],group['t_contact(sec)'])
    Y = group['delay(sec)'] / group['avg_cn_period(sec)']
    Z = group['expected_angle(rad)'] - group[angle]
    if len(Z.dropna())>1:
        Z = Z - Z.dropna().iloc[0]
    # Z = Z-Z.iloc[0]
    # Get color from the colormap
    color = cmap(norm_color(group['t_contact(sec)'].mean()))
    # color = cmap(norm_color(group['exp'].mean()))


    # Scatter plot for the points
    ax.scatter(X, Y, Z, color=color)
    # Line plot to connect the points
    ax.plot(X, Y, Z, label=f'Exp: {exp}, Event: {event}',color=color)

ax.set_xlabel(r'$\frac{T_{contact}}{T_{CN}}$')
ax.set_ylabel(r'$\frac{T_{delay}}{T_{CN}}$')
ax.set_zlabel(r'$\Delta\phi (rad)$')

ax.set_xlim([0,1])
ax.set_ylim([0,0.3])
ax.set_zlim([0,3])

# Color bar settings for Z values
scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
scalar_map.set_array([])

cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Contact duration(sec)')

# plt.title('3D Scatter Plot with Connected Points')
# plt.legend()
plt.show()

#%% 3D mesh surface
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import scipy.interpolate
import pandas as pd

# Function to apply a simple moving average for smoothing
def smooth_data_mesh(data, window_size=3):
    return data.rolling(window=window_size, min_periods=1).mean()

# Angle options and selection
ang = ['base_lin(rad)', 'base2tip(rad)', 'tip_lin(rad)']
angle = ang[0]  # Selecting the second angle type

# Assuming jump_panda is your DataFrame

# Creating a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize lists for surface plot data
all_X = []
all_Y = []
all_Z = []

# Axis limits based on data ranges
x_lim = 5
y_lim = 5
z_lim = -0.05  # Z axis lower limit

# Loop through each group in DataFrame, only including groups with exp > 15
for (exp, event), group in jump_panda.groupby(['exp', 'event']):
    if exp <= 15:
        continue  # Skip groups with exp <= 15

    group = group[group['expected_angle(rad)'] - group[angle] > z_lim]

    X = group['t_contact(sec)'] / group['avg_cn_period(sec)']
    Y = group['delay(sec)'] / group['avg_cn_period(sec)']
    Z = group['expected_angle(rad)'] - group[angle]

    # Apply smoothing to Z values
    Z_smooth = smooth_data_mesh(Z,window_size = 201)

    # Filter data within specified limits
    valid_data = (X <= x_lim) & (Y <= y_lim) & (Z<=2)
    all_X.extend(X[valid_data])
    all_Y.extend(Y[valid_data])
    all_Z.extend(Z_smooth[valid_data])

# Normalize Z for colormap
norm = plt.Normalize(min(all_Z), max(all_Z))
cmap = cm.viridis

# Grid creation and interpolation
xi, yi = np.linspace(0, x_lim, 200), np.linspace(0, y_lim, 200)
xi, yi = np.meshgrid(xi, yi)
zi = scipy.interpolate.griddata((all_X, all_Y), all_Z, (xi, yi), method='cubic')

# Plotting the surface, colored by Z values
surf = ax.plot_surface(xi, yi, zi, cmap=cmap, facecolors=cmap(norm(zi)), edgecolor='none')

# Color bar settings for Z values
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Z values')

# Setting axis labels
ax.set_xlabel(r'$\frac{T_{contact}}{T_{CN}}$')
ax.set_ylabel(r'$\frac{T_{delay}}{T_{CN}}$')
ax.set_zlabel(r'$\Delta\phi (rad)$')

ax.set_xlim([-0.05,0.5])
ax.set_ylim([0.05,0.25])
ax.set_zlim([-0.5,2])
plt.show()

#%%figure 11? - twine dist. for different CN rates (motor modified)
print('sup')

pass
#%% figure 1? - correlations of twine/slip time
# do i want these? is it interesting?
#%% S1- all twine/slip events
from matplotlib.lines import Line2D
savep = 0

legend_elements = [Line2D([0], [0], color=cmap(0.2), lw=2, label='twine'),
                   Line2D([0], [0], color=cmap(0.8), lw=2, label='slip')]
# Create the figure
fig, ax = plt.subplots()
ax.legend(handles=legend_elements, loc='best')

for i in range(N):
    if events[i].twine_state==1:
        plt.plot(events[i].timer/60,np.multiply(events[i].F_bean,mg2mN),
                 color=cmap(0.2),alpha=0.4)
for i in range(N):
    if events[i].twine_state==0:
        plt.plot(events[i].timer[:-7]/60,np.multiply(events[i].F_bean[:-7],mg2mN),
                 color=cmap(0.8),alpha=0.8)

plt.xlabel('time(min)',fontsize=fs)
plt.ylabel('Force(mN)',fontsize=fs)

plt.show()

file_end = '\\  xxx .svg'
if savep == 1:
    plt.savefig(basepath + r"\figures\Paper\S1"+file_end)
#%% test plots
plt.figure()
for i in range(len(events)):
    if 'r_squared' not in dir(events[i]):
        # plt.plot(np.array(events_filter[i].F_bean)/events_filter[i].p.m_sup)
        print(str(events[i].p.exp_num)+' , '+str(events[i].event_num)
              + ' has no fit')
        print(events[i].twine_state)
        plt.plot(events[i].F_bean,'-b')
    elif events[i].r_squared==0:
        print(str(events[i].p.exp_num)+' , '+str(events[i].event_num)
              + r' R^2 = 0')
        print(events[i].twine_state)
        plt.plot(events[i].F_bean,'-b')
#
fig,ax = plt.subplots(2,1)
for i in range(len(events)):
    if events[i].twine_state==0:
        ax[0].plot(events[i].L_contact2stemtip_cm,events[i].dec_time,'x')
    else:
        ax[1].plot(events[i].L_contact2stemtip_cm,events[i].dec_time,'x')
#
for i in range(len(events)):
    if events[i].twine_state==0:
    # if 'fit_A' in dir(events[i]):
        if events[i].L_contact2stemtip_cm>5:
            print(i,events[i].p.exp_num,events[i].event_num,events[i].L_contact2stemtip_cm)
#%% test twine times
for i in range(len(events)):
    try:
        if events[i].twine_state==1 and events[i].auto_dec_time_min<60:
            plt.plot(f'{events[i].p.exp_num}, {events[i].event_num}',
                      events[i].auto_dec_time_min,'rx')
            print(f'{i=},{events[i].p.exp_num=}, {events[i].event_num=},\
                  {events[i].auto_twinetime_method=},{events[i].auto_dec_time_min=},\
                  {events[i].dec_time=}')
        # else:

        #     plt.plot(f'{events[i].p.exp_num}, {events[i].event_num}',
        #           events[i].dec_time,'bx')

    except:
        continue

#%% test angles
for i in range(len(events)):
    try:
        if (events[i].p.exp_num,events[i].event_num) in [(21,1),\
           (49,4),(61,1),(92,3),(111,1)]:
            plt.figure()
            plt.title(f'{i=}')
            plt.plot(events[i].near_sup_time,events[i].near_sup_angle,'x')
            print(events[i].p.exp_num,events[i].event_num)
    except Exception as e:
        print(f'{e}')
#%% debug

print ('what do you wish to debug my funky little friend?')
#%% find events
for i in range(len(events_filter)):
    try:
        i_contact = uf.closest(events_filter[i].L_contact2stemtip_cm,events_filter[i].p.L_s)
        L_s_contact = events_filter[i].p.L_s[i_contact]
        E_L_s_contact = interp_E_exp_cont[uf.closest(x_continuous,L_s_contact)]
        if abs(E_L_s_contact-14)<2:
            print(i,events[i].p.exp_num,events[i].event_num)
            print(events[i].fit_A)
    except Exception as e:
        print(f'{e}')
#%% test sine fits
for i in range(120,140):
    try:
        if events[i].twine_state==0: continue
        f = uf.remove_outliers_loop(events[i].F_bean,cutoff=0.75,loop=50)
        events[i].f_clean = f
        l = len(f)
        t = events[i].timer
        events[i].n_Tcn = uf.closest(t,events[i].p.avgT*60*0.5) #
        time[i] = np.divide(t[:events[i].n_Tcn],60) # time in minutes
        res = uf.fit_sin(time[i], f[:events[i].n_Tcn],1/(2*events[i].p.avgT))
        #plot data
        data_color='blue'
        data_alpha=0.4
        plt.figure()
        plt.plot(time[i],f[:events[i].n_Tcn],'-',
                  color=data_color,alpha=data_alpha,markersize=0.2)

        # plot fitted sine func
        x_range = np.linspace(0,time[i][-1],1000)
        plt.plot(x_range, uf.sinfunc(x_range,res["amp"],
                      res["omega"],res["offset"]),'--r')
        plt.title(f'{res["r_squared"]=}')
        plt.figure()
        plt.plot(time[i],f[:events[i].n_Tcn]-uf.sinfunc(time[i],res["amp"],
                      res["omega"],res["offset"]))
        plt.title('residuals')
    except Exception as e:
        print(f'{e}')
        continue
#%% linearize F (arcsine) and fit
plt.figure()
for i in range(len(events)):
    F_1 = m.asin(events[i].F_bean)
    plt.plot(F_1)
#%% twine time vs rotation time (motor plots with fit)

import importlib
import sys
import useful_functions as uf # my functions
sys.path.append(r'C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\twining_initiation\Data_Extraction')

importlib.reload(uf)

var1_twine = []
var2_twine = []
for i in range(n_tot):
    var1 = events[i].dec_time
    var2 = events[i].p.avgT
    # print(var1,var2)
    if events[i].twine_state==1: 
        if var1<50: 
            print(events[i].p.exp_num,events[i].event_num)
            continue
        var1_twine.append(var1)
        var2_twine.append(var2)
df1 = pd.DataFrame(var1_twine,columns=['twine time(min)'])
df2 = pd.DataFrame(var2_twine,columns=['avg CN(min)'])
# make df1 the first column and df2 the second column
df = pd.concat([df1,df2],axis=1).sort_values(by='avg CN(min)')
fig,ax = plt.subplots()
ax.errorbar(1/(df['avg CN(min)']/60),df['twine time(min)'],
        xerr=0.05,yerr=7,fmt='+',color='black',label='no motor twine')


# plot data from motor modified experiments
misc_path = r'C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2c_motor_modified_CN\motor mod_misx_new.xlsx'
motor_pandas = pd.read_excel(misc_path,sheet_name='meta-data motor_mod')
# get (no)twine times and remove nans
data = pd.DataFrame({'x':motor_pandas['effective_CN_rate(rph)'],
                    'y1':motor_pandas['twine_time_estimate(min)'],
                    'y2':motor_pandas['no twine time(min)']}).dropna().sort_values('x') 

# plot raw data
y1_raw = data[data['y1']>0]['y1']
x1_raw = data[data['y1']>0]['x']
fs = 16
ax.plot(x1_raw,y1_raw,'bo', label='motor twine')
uf.fit_w_err(ax,x1_raw,0.05,y1_raw,5,fit_func=uf.expfunc_core) # expfunc_core powerfunc

ax.set_xlabel('Rotation rate (rph)',fontsize = fs)
ax.set_ylabel('Twine Time (min)',fontsize = fs)
# ax.title('Twine Time vs Average CN Time')
ax.legend(fontsize = fs-3)
ax.set_xlim([0,2])
ax.set_ylim([0,400])
plt.show()

# do the same for average rotation times
# the 3 times below 50 are mistakes in the pendulum exp twine estimation
#%% support masses
mass = [0.2,0.65,0.86,3.1]
fig,ax = plt.subplots(1,4)
# make figure wider
fig.set_figwidth(13)
# set all plots to the same scale and labels
for i in range(len(mass)):
    ax[i].set_ylim([0,0.125])
    ax[i].set_xlim([0,12500])
    ax[i].set_title(f"Mass: {mass[i]}g",fontsize=fs)
    ax[i].set_xlabel("time(sec)",fontsize=fs)
ax[0].set_ylabel("alpha(rad)",fontsize=fs)

for i in range(len(events)):
    # if events[i].twine_state==1: continue
    # if events[i].p.m_sup not in [mass[2],mass[3]]: continue
    if events[i].p.m_sup in mass:
        index = mass.index(events[i].p.m_sup)
        ax[index].plot(events[i].timer, events[i].alpha,
                       'x',markersize=0.5,color=cmap(mass[index]/max(mass)))
        # / or * by contact length? (np.mean(events[i].L_contact2suptip))
    
#%% twine/slip probability graphs
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
n_bins = 15
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
k = 8
# compare: integ_f, alpha, F_bean, L_contact2stemtip_cm,
# F1stmax , L_base,dec_time, p.avgT, int_eps

# save values to 2 lists
for i in range(n_tot):
    if events[i].p.exp_num==69 and events[i].event_num==1 or\
        events[i].p.exp_num==98 and events[i].event_num==2: continue
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


# pandas histogram
hist_twine = np.histogram(df1,bins=n_bins,range=x_range[k])
hist_slip = np.histogram(df2,bins=n_bins,range=x_range[k])

df.plot.hist(stacked=False, bins=n_bins, figsize=(6, 6),
             grid=False,range=x_range[k],density=False,
             color = [cmap(0.8),cmap(0.2)],alpha = 0.65)

plt.xlabel(var_name[k], fontsize=fs)
plt.ylabel('Frequency', fontsize=fs)

# Calculate the ratio between slip and twine events for each bin
ratio = (hist_slip[0] - hist_twine[0] )/ (hist_slip[0] + hist_twine[0] )
ratio_mod = []
# for r in ratio:
#     if r == np.inf: ratio_mod.append(1)
#     elif np.isnan(r): ratio_mod.append(0)
#     else: ratio_mod.append(r)

    # ratio_mod.append(r if r != np.inf, 1 elif r == np.inf, else 0)
# Get the number of events in each bin for twine and slip
twine_counts, _ = np.histogram(df1, bins=n_bins, range=x_range[k])
slip_counts, _ = np.histogram(df2, bins=n_bins, range=x_range[k])

# Divide the ratio by the number of events in each bin
ratio /= (twine_counts + slip_counts)

# Plot the adjusted ratio
plt.figure()
plt.bar(hist_twine[1][:-1], ratio, width=np.diff(hist_twine[1]), color=cmap(0.2))
plt.xlabel(var_name[k], fontsize=fs)
plt.ylabel('Adjusted Slip/Twine Ratio', fontsize=fs)
plt.title('Adjusted Slip/Twine Ratio vs ' + var_name[k], fontsize=fs)
plt.show()

#%%