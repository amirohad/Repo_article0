# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:19:53 2023

functions for exp2 - Pendulum force

A. plant class
B. event class
0. get tracked data
1. calculate angle relative to vertical in side and top views
2. calc angle for time series
3. calculate force in grams of bean on support via moment equilibrium equation,
    in units of cgs.
4. calculate force for time series
5. h5 functions
6. get point coordinates near support-> get angle relative to horizontal

@author: Amir
"""
#%% imports
import math as m
import numpy as np
import re
import time
import seaborn
import scipy
import sys
import os
import h5py
from scipy.signal import savgol_filter,resample
import os,glob
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

sys.path.append('..')
import useful_functions as uf # my functions
#%% A - plant class
class Plant:
    """ plant class, insert all parameters from XL, Youngs modulus,
        top and side trajectories"""

    def __init__(self,df,basepath,i,exp):
        self.exp_num = exp
        self.plant_path = os.path.join(basepath,r'Measurements',
           str(self.exp_num),r'Side',r'cn')

        self.genus = df.at[i,'Bean_Strain']
        self.camera = df.at[i,'Camera']  # images from nikon or pi
        self.m_sup = float(df.at[i,'Straw_Weight(gr)']) # support mass
        self.arm_cm = float(df.at[i,'Exp_start_arm_length(cm)'])
        self.start_height = float(df.at[i,'Exp_start_arm_length(cm)'])
        self.m20cm = float(df.at[i,'Weight20cm(gr)'])
        self.L0 = float(df.at[i,'initial_length(cm)']) # initial arm+height
        self.mbean = float(df.at[i,'Final weight(g)'])
        # from physical micrometer measurements
        self.r_measure = {5:float(df.at[i,'Diameter_5(mm)'])/20,
             10:float(df.at[i,'Diameter_10(mm)'])/20,
             15:float(df.at[i,'Diameter_15(mm)'])/20,
             20:float(df.at[i,'Diameter_20(mm)'])/20,
             35:float(df.at[i,'Diameter_mid(mm)'])/20} # convert to r(cm)
        # self.m20cm = 0.173 # m_average over all - can change to individual
        self.L_s = np.arange(0,25,0.05) # L-s: from 0 till 25cm
        self.s = np.arange(0,self.L0,0.05) # s from 0 till L0

        # self.density = float(data_panda.at[i,'Density(g/mm3)'])
        # self.density = self.m20cm/(5*m.pi*sum(np.square(
        #     [t/10 for t in self.r_measure.values()]))) # avg shoot density in first 20cm (g/cm^3())

    def view_data(self,df_tot,i):
        df = df_tot.iloc[i]
        self.pix2cm_s = float(df.at['Side_pix2cm']) # side pixel to cm ratio
        self.pix2cm_t = float(df.at['Top_pix2cm']) # top pixel to cm ratio

        self.Lsup_pix = df.at['Dist_straw_from_hinge(pixels)'] \
            + df.at['Straw_length(pixels)']# support lenth in pix
        self.Lsup_cm = self.Lsup_pix*self.pix2cm_s # support lenth in cm
        # z position of bottom tip of support
        self.support_base_z_pos_pix = float(df.at['side_equil_ypos-bot_sup(pixels)'])
        self.support_base_z_pos_cm = self.support_base_z_pos_pix*self.pix2cm_s

    def getE(self,E_dict):
        self.avgE_sections = {}
        self.r_sections = {}
        self.avgE_sections[0]=0 # E is 0 at tip?

        if self.exp_num in E_dict.keys():
            for k in range(len(E_dict[self.exp_num][0])):
                if E_dict[self.exp_num][0][k]!=E_dict[self.exp_num][0][k]: continue
                if re.findall('\d{1,2}-',E_dict[self.exp_num][0][k]):
                    sect = float(re.findall('\d{1,2}-',
                          E_dict[self.exp_num][0][k])[0].replace('-',''))+5/2 # take mid point of section
                    r = float(E_dict[self.exp_num][1][k])
                if E_dict[self.exp_num][0][k]=='avg':
                    self.avgE_sections[sect]=float(E_dict[self.exp_num][1][k]) # youngs modulus
                    self.r_sections[sect]=float(r) # radius

            # spline to get continuous form for E(s), r(s)
            x_sections = [x for x in sorted(self.avgE_sections.keys())] # take mid point of sections
            r_sections_list = [x for x in sorted(self.r_sections.values())] # is in mm!

            if len(x_sections)>=2: # spline 3+ points
                interp_r = scipy.interpolate.UnivariateSpline(x_sections[1:],
                                                              r_sections_list,k=1)
            else:
                x_sections = [x for x in sorted(self.r_measure.keys())]
                r_sections_list = [x for x in sorted(self.r_measure.values())]
                interp_r = scipy.interpolate.UnivariateSpline(x_sections[1:],
                                                              r_sections_list,k=1)
        else: # if no image analysis, rely on physical measurements
            x_sections = [x for x in sorted(self.r_measure.keys())]
            r_sections_list = [x for x in sorted(self.r_measure.values())]
            interp_r = scipy.interpolate.UnivariateSpline(x_sections,
                                      r_sections_list,k=1) # is in mm!

        # interpolate r(s)
        # interp_r = scipy.stats.linregress(x_sections, r_sections_list) # ??
        self.interp_r = interp_r(self.L_s) # r in mm(L_s in cm)
        self.ab_r = interp_r.get_coeffs()
        a,b = self.ab_r

        # volume: integrate analytic np.pi*r(L-s)**2*r*dr*ds for L-s between 0 and 20
        self.vol20 = 20*np.pi*((1/3)*(a**2)*(20**2)+(a*b*20)+b**2) # in cm^3 # check!!
        self.density = self.m20cm/self.vol20 # in gr/cm^3
        # integrate numeric:
        # self.vol20 = np.pi*scipy.integrate.trapz(
        #     np.arange(0,20,0.05),self.interp_r**2)/self.m20cm

    def cn_data(self,df,i):
        '''df=data drame, i=index of data frame'''
        self.T,self.avgT = uf.get_Tcn(self.plant_path,df,i) # get Tcn from excel or folder
        self.omega0 = 2*m.pi/self.avgT # base rotation angular velocity

#%% B - Event class
class Event:
    """event class, plant as input"""
    def __init__(self,plant,df,i):
        self.p = plant

    def view_data(self,df,i,view):
        if view == 'side':
            self.frm0_side = int(df.at[i,'First_contact_frame']) # first contact frame (starts from 1)
            self.frm_dec_side = int(df.at[i,'Slip/Twine_frame']) # frame of twine_state/slip
        if view == 'top':
            self.frm0_top = int(df.at[i,'First_contact_frame']) # first contact frame (starts from 1)
            self.frm_dec_top = int(df.at[i,'Slip/Twine_frame']) # frame of twine_state/slip
            self.L_contact2stemtip_cm = self.p.pix2cm_t * float(df.at[i,
                'Contact_distance_from_stem_tip(pixels)']) # contact distance from stem tip
            self.twine_state = float(df.at[i,'Twine_status']) # twine_state/slip
            # self.L_base = float(df.at[i,'contact distance from base']) # in  cm, from db
            self.L_base = self.p.L0-self.L_contact2stemtip_cm
            self.start_arm = int(df.at[i,'Exp_start_arm_length(cm)']) # start arm length
    def get_twine_time(self,exp,event,view,h5_dict,near_sup_track_dict,
                       twine_threshold,track_dict,to_plot=0):
        '''if exp,event is in track_list- get twine start time from there
        else- get twine start time from h5_list
        else- get twine start time from manual- defer to original method'''
        if view=='side': return

        if self.twine_state==0: return

        if to_plot==1: figax = plt.subplots(2,2)
        else: figax = [[],[]]

        h5keys = [key[:-1] for key in h5_dict.keys()]
        if (exp,event) in h5keys: # use root-stem data as default
            self.near_sup_time,self.near_sup_angle = analyze_event(
                h5_dict,exp,event,axis=figax)
            self.auto_twinetime_method = 'h5'
            resamp=0
        elif (exp,event) in near_sup_track_dict.keys(): # if regular tracking exists
            self.auto_twinetime_method = 'track'
            self.near_sup_time,self.near_sup_angle = get_track_angle_near_support(
                    near_sup_track_dict,exp,event,axis=figax)
            resamp=1
        else:
            self.auto_twinetime_method = 'None'
            self.near_sup_time,self.near_sup_angle = ([0],[0])
            return

        x = twine_initiation(self.near_sup_time,self.near_sup_angle,
                             twine_threshold,resampling=resamp)
        if len(x)==3:
            self.auto_dec_angle,self.auto_dec_time_min,self.auto_dec_index = x

        else:# if no result from tracking (check why)
            self.auto_dec_angle,self.auto_dec_time_min,self.auto_dec_index = \
                np.nan,np.nan,np.nan
            return

        if self.auto_twinetime_method=='h5':
            # add to the time found in twine initiation(thrshold crossed) the
            # time from contact until the h5 analysis began
            compensate = h5_twine_initiation_compensation(
                                self,exp,event,h5_dict,track_dict)
            self.auto_dec_index += compensate
            self.auto_dec_time_min += compensate
        self.frm_dec_side = self.auto_dec_index
        self.frm_dec_top = self.auto_dec_index
        # print(f'{self.auto_dec_angle=}') print automated decision angle

    def event_base_calcs(self,view,track_dict,contact_dict):
        '''side view, get coordinates of: x,z track pix,
        x,z track cm, x,z contact pix, x,z contact cm,
        distance of track position to support tip,
        track timer, contact timer.
        top view, get coordinates of: x,y track pix,
        x,y track cm, '''
        if view == 'side':
            # x,z coordinates of side view
            self.x_track_side0,self.z_track_side0,time_s = funcget_tracked_data(
                track_dict[(self.p.exp_num,self.event_num,view)][0]
                ,[0,-1],view,self.p.camera)
            # (x,z) equilibrium coordinates
            self.x0_s,self.z0_s = self.x_track_side0[0],self.z_track_side0[0]
            # transform x,z to coordinates relative to x0,z0
            self.x_track_side,self.z_track_side = \
                np.subtract(self.x_track_side0,self.x0_s),\
                np.subtract(self.z_track_side0,self.z0_s)
            # convert to cm
            self.x_track_side_cm = np.multiply(self.x_track_side,self.p.pix2cm_s)
            self.z_track_side_cm = np.multiply(self.z_track_side,self.p.pix2cm_s)

            # get contact data
            self.x_cont,self.z_cont,self.contact_timer = funcget_tracked_data(
                contact_dict[(self.p.exp_num,self.event_num)][0],[0,-1],
                view,self.p.camera,1)
            # transform (x,z) to coordinates relative to x0,z0
            self.x_cont,self.z_cont = \
                np.subtract(self.x_cont,self.x0_s),np.subtract(self.z_cont,self.z0_s)
            # convert to cm
            self.x_cont_cm = np.multiply(self.x_cont,self.p.pix2cm_s)
            self.z_cont_cm = np.multiply(self.z_cont,self.p.pix2cm_s)

            # save to dict only coor within decision timeframe
            self.xz_contact = np.array([self.x_cont_cm[self.frm0_side:self.frm_dec_side],
                            self.z_cont_cm[self.frm0_side:self.frm_dec_side]])
            # get z dist. of tracked spot at equil. to bottom of support + convert to cm
            self.L_track2suptip = abs(self.z0_s-self.p.support_base_z_pos_pix)
            self.L_track2suptip_cm = self.L_track2suptip*self.p.pix2cm_s

        else:
            self.x_track_top0,self.y_track_top0,self.support_timer = funcget_tracked_data(
                track_dict[(self.p.exp_num,self.event_num,view)][0]
                ,[0,-1],view,self.p.camera)
            # (x,y) equilibrium coordinates
            self.x0_t,self.y0_t = self.x_track_top0[0],self.y_track_top0[0]
            # transform x,y to coordinates relative to x0,y0 in top view
            self.x_track_top,self.y_track_top = \
                np.subtract(self.x_track_top0,self.x0_t),np.subtract(self.y_track_top0,self.y0_t)
            # convert to cm
            self.x_track_top_cm = np.multiply(self.x_track_top,self.p.pix2cm_t)
            self.y_track_top_cm = np.multiply(self.y_track_top,self.p.pix2cm_t)

            # set length of events to dec period only
            self.dec_x_track_top = self.x_track_top_cm[self.frm0_top:self.frm_dec_top]
            self.dec_y_track_top = self.y_track_top_cm[self.frm0_top:self.frm_dec_top]
            self.dec_z_track_side = self.z_track_side_cm[self.frm0_side:self.frm_dec_side]
            # collect track xyz
            self.xyz = np.zeros((3,len(self.dec_x_track_top))) # all xyz of sup track
            self.xyz[0,::] = self.dec_x_track_top
            self.xyz[1,::] = self.dec_y_track_top
            self.dec_z_track_side,self.dec_y_track_top = \
                uf.adjust_len(self.dec_z_track_side,self.dec_y_track_top,
                  choose=self.dec_y_track_top)
            self.xyz[2,::] = self.dec_z_track_side
            # timer for decision period
            self.timer = np.subtract(self.support_timer[self.frm0_top:
                        self.frm_dec_top],self.support_timer[self.frm0_top])
            # self.xyz = np.array(self.xyz_supportrack) # in cm
            self.xyz0 = np.array([[self.xyz[0][0],self.xyz[1][0],
                self.xyz[2][0]]]*len(self.xyz[0])).T # initial xyz trk point


    def event_calc_variables(self,view):
        if view == 'side':
            return
        else:
            # decision time
            self.dec_time = (self.frm_dec_top-self.frm0_top)/2

            # calculate 3D distance and angle
            self.trk_dist = np.sqrt(sum((self.xyz-self.xyz0)**2)) # root of sum of squares
            self.alpha = [m.asin(d/(2*(self.p.Lsup_cm-
                            self.L_track2suptip_cm))) for d in self.trk_dist]
            if len(self.alpha)>3:
                self.omega = np.gradient(self.alpha)/30 # angular velocity
            # find z projection of contact position on support
            self.h = abs(np.subtract(self.xz_contact[1],self.xz_contact[1][0])) # z projection on vertical in cm
            self.h,self.alpha = uf.adjust_len(self.h,self.alpha,choose=self.alpha) # adjust length and trim to decision period
            self.L_contact2suptip = [b/(m.cos(a))+self.L_track2suptip_cm
                                   for a,b in zip(self.alpha,self.h)] # find hypotenus and add L_track2suptip dist. in cm
            # L_contact2suptip_pix[i] = np.multiply(x,pix2cm[i]) # distance of contact from sup. tip in cm
            # calculate force
            self.F_bean = F_of_t(self.L_contact2suptip,
                      self.p.Lsup_cm, self.alpha,self.p.m_sup)
            # calculate torque
            self.torque = np.multiply(self.F_bean,
                                  (self.start_arm-self.L_contact2stemtip_cm))

            # integrate force from initial contact till decision frame
            self.integ_f = scipy.integrate.trapz(self.F_bean,self.timer)

            # integrate torque
            self.integ_torque = scipy.integrate.trapz(self.torque,self.timer)

            # calculate work from contact till decision
            trk_dist_diff = list(np.diff(self.trk_dist))
            trk_dist_diff.append(0)
            self.work = sum(np.multiply(trk_dist_diff,self.F_bean))
            #need to multiply by dt(0.5min)
    def max_strain(self,E_Ls,t_i):

        """find maximal strain for given moment. input avgE(L-s),
        use individual interp_r"""
        s = np.arange(0,self.L_base,0.05) # s between 0 and L_base
        eps = []
        for i in range(len(s)): # for each position in s
            x_i = uf.closest(self.p.L_s,self.p.L0-s[i]) #find closest index in x to L-s
            eps.append((self.F_bean[t_i]*1e-5)*(self.L_base-s[i])/
                       (100*E_Ls[x_i]*self.p.interp_r[x_i]**3)) # force in Newton =~ F_mg*1e-5
        return max(eps)
    # the problem is cm and mm mixup, check E(!!) is in matching units:
    # megapascal is 100 N/cm^2, all lengths here are in cm

    def max_strain_t(self):
        '''get max strain per time point, integrate over time'''
        self.eps_t = []
        dec_len = len(self.timer[:uf.closest(self.timer, self.dec_time*60)])
        for i in range(dec_len):
            self.eps_t.append(self.max_strain(self.p.E_Ls,i))

        self.int_eps = scipy.integrate.trapz(
            self.eps_t[:dec_len],self.timer[:dec_len])

    def print_variables(self):
        # print variables using __dict__
        obj_dict = self.__dict__
        print("Using __dict__:", obj_dict)
#%% 0 get tracked data
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
            if index[i] in obj:        # if current line belongs to the requested tracked object
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
                if camera=='nikon':
                    timer = [30*x for x in range(N)]

            else:
                print('skipped non-selected tracked object')
                print(timer[i],type(timer[i]),i,currentline)
            i+=1

        # x = np.subtract(xcntr,xcntr[0]) # return x relative to start point
        # y = np.subtract(ycntr,ycntr[0]) # return y relative to start point
        return xcntr,ycntr,timer
#%% 1. calculate angle relative to vertical in side and top views
def calc_angle(lsup_pix,lsup_cm,dist_pix,pix2cm,view):
    if view=='side':
        alpha_deg=m.asin(dist_pix/lsup_pix) #pix calculation
    elif view=='top':
        alpha_deg=m.asin((dist_pix*pix2cm)/lsup_cm) #cm calculation
    return alpha_deg
# notice that if im in top view i dont have the lpix from the image, only the
# actual size from the side view/direct measurement
#%% 2. calc angle for time series
def alpha_of_t(lsup_pix,lsup_cm,dist_pix,pix2cm,view):
    # dlsup_pix,dlsup_cm,ddist_pix
    N=np.size(dist_pix) #size of distance vector
    angle=[[]]*N
    dangle=[[]]*N
    for i in range(N):
        angle[i]=calc_angle(lsup_pix,lsup_cm,dist_pix[i],pix2cm,view)
        # dangle[i]=calc_dangle(lsup_pix,dlsup_pix,lsup_cm,dlsup_cm,dist_pix[i],ddist_pix,pix2cm,view)
    return angle,dangle

#%% 3. calculate force in mg
def calc_F(d_contact,l_sup_cm,phi_t,m_sup):
    gcgs=980
    F_mg = 1000*m_sup * l_sup_cm * m.tan(phi_t)/(2*(l_sup_cm-d_contact))
# the force applied by bean stem in grams(!)
# to get the force in dyne-> *gcgs=980 cm/s^2
    return abs(F_mg)
#%% 4. calculate force for time series
def F_of_t(d_contact,l_sup_cm,phi,m_sup):
    #dl_sup_cm, dd_contact, dm_sup,dphi
    N=np.size(phi)
    Fvec=[[]]*N
    # dFvec=[[]]*N
    for i in range(N):
        Fvec[i]=calc_F(d_contact[i], l_sup_cm, phi[i], m_sup)
        # dFvec[i]=calc_dF(d_contact[i], dd_contact, l_sup_cm, dm_sup, phi[i], dphi[i], m_sup, dm_sup)
    return Fvec #dFvec
#%% 5. h5 functions
def angle_s(x,y):
    alpha_deg = (np.degrees([m.atan(np.gradient(y,x)[j])
                 for j in range(len(y))]))
    return alpha_deg

def analyze_event(h5_dict,target_exp,target_event,smooth_segment_length=50,axis=[]):
    '''find all event files, merge timestamps, get angle
        if plot=1: plot by period, save figure'''
    # get base path
    basepath = os.path.dirname(list(h5_dict.values())[0][0])
    # find all files for this event
    exp_event_list = [re.findall('interekt_'+str(target_exp)+
                     '_e_'+str(target_event)+'.*',file[0])
                      for file in h5_dict.values()]
    exp_event_list = [os.path.join(basepath,sublist[0])
                      for sublist in exp_event_list if sublist]

    if exp_event_list == []:
        print(f'no event {target_event} in exp {target_exp}')
        return

    # merge all timestamps - all_periods[i][j][k][l]:period,x|y,t,s
    all_periods = merge_event_data(exp_event_list)

    # plot by period
    event_theta = analyze_by_period(all_periods,smooth_segment_length,axis)

    # when plotting
    if len(axis[1])>0:
        # Set a common title for all subplots
        axis[0].suptitle(f'Exp #{target_exp}, event #{target_event}', fontsize=16)
        # Adjust layout
        plt.tight_layout()
        # save figure (and data?)
        # plt.savefig(os.path.join(base_path,
        #                  'results_'+target_exp+'_'+target_event+'.png'),dpi=200)
        print('done')

    times = [30*p for p in range(len(event_theta))]
    # add resampling in time here as well?
    return times, event_theta

def merge_event_data(event_file_list):
    all_periods = []
    # h5_file_list = [filename[0] for filename in event_file_list]
    # if not filenames: return []
    for filename in event_file_list:
        with h5py.File(filename) as f:
            sections = f["data"]
            section_names = list(sections.keys())
            # if 2 sections, take only the top one
            # print(filename,section_names)
            if len(sections)>1:
                a = sections[section_names[0]]
                b = sections[section_names[1]]
                # drop in y values is up in image, take first 100 points
                if np.average(a['yc'][0][:100])<np.average(b['yc'][0][:100]):
                    pass
                else: a=b
            else: a = sections[section_names[0]]

            # take only parts of stem with diameter < 80 (try)
            mask = np.array(a['diam'])<80
            yc = np.array(a['yc'])
            yc[~mask]=np.nan
            xc = np.array(a['xc'])
            xc[~mask]=np.nan
            # all_periods.append([a['xc'],a['yc']])
            all_periods.append([xc,yc])
    return all_periods

def analyze_by_period(periods,smooth_segment_length=50,axis=[]):
    '''input list of all event periods.
    if len(ax) is not positive- no plots.
    else: plots raw xy data, smoothed xy data for first l points,
    angle along l points,and average angle over time for l points.
    periods[i][j][k][l]:period,x|y,t,s'''
    # iterate over all period and all times within each period
    normalize = Normalize(vmin = 0, vmax = len(periods) - 1) # get normalized colormap
    t = 0
    colormap = plt.get_cmap('viridis')
    event_theta = []
    for j, period in enumerate(periods): # for every period, plot data
        section_color = colormap(normalize(j)) # set color for this period
        n = len(period[0])
        for i in range(n):
            # print(i,color)
            yc = period[1][i]
            xc = period[0][i]
            xc = xc[xc<10000]
            yc = abs(yc[yc<10000]-30000)

            # plot raw data in ax[0][0]
            if len(axis[1])>0:
                ax = axis[1]
                ax[0][0].plot(xc, yc,'o',color=section_color,
                                  markersize=1,alpha=max(0.25,i/n))

            # make odd interpolation window
            window = int(len(yc)/5)
            if window%2==0: window+=1
            else: continue
            # plot non-short segments
            if len(yc)>10:
                # shorten vectors and start them from the origin
                xc_cut = xc[:smooth_segment_length]
                xc_cut = -(xc_cut-xc_cut[0])
                yc_cut = savgol_filter(yc,window,2)[:smooth_segment_length]
                # yc_cut = yc[:smooth_segment_length]
                yc_cut = yc_cut-yc_cut[0]

                #resampling
                target_length = int(2*smooth_segment_length/3)
                xc_resample = np.linspace(xc_cut.min(), xc_cut.max(),target_length)[1:]
                yc_resample = resample(yc_cut, target_length)[1:]
                xc_resample = xc_resample-xc_resample[0]
                yc_resample = yc_resample-yc_resample[0]
                # resmooth after resampling
                # # Create a cubic spline interpolation
                # f = scipy.interpolate.interp1d(xc_resample, yc_resample, kind='cubic')
                # # Define the range of x values for the smoothed curve
                # x_smooth = np.linspace(min(xc_resample), max(xc_resample), 1000)
                # # Use the spline interpolation to get the smoothed y values
                # y_smooth = f(x_smooth)


                # calculate angle
                theta = angle_s(xc_resample,yc_resample)
                event_theta.append(np.average(theta)) # save average

                if len(axis[1])>0: # when plotting
                    # plot smoothed data in ax[0][1]
                    ax[0][1].plot(xc_resample,yc_resample,
                            color=section_color,alpha=max(0.25,i/n))
                    # plot angle over s for all times
                    ax[1][0].plot(theta,color=section_color,alpha=max(0.25,i/n))
                    # plot average angle over time for above data
                    ax[1][1].plot(t,np.average(theta),'o',
                                  color=section_color,alpha=max(0.25,i/n))
                    # set subplot titles
                    ax[0][0].set_title('raw xy')
                    ax[0][1].set_title('smoothed xy')
                    ax[1][0].set_title('angle for smoothed along shoot')
                    ax[1][1].set_title('average angle over time')
                    # set x-axis titles
                    ax[0][0].set_xlabel('x(pixels)')
                    ax[0][1].set_xlabel('x(pixels)')
                    ax[1][0].set_xlabel('s index') # small is close to contact i think
                    ax[1][1].set_xlabel(r'$t(sec)$') # normalize by individual Tcn!
                    # set y-axis titles
                    ax[0][0].set_ylabel('y(pixels)')
                    ax[0][1].set_xlabel('y(pixels)')
                    ax[1][0].set_ylabel(r'$\theta($'+chr(176)+')') # add degree symbol
                    ax[1][1].set_ylabel(r'$\theta_{avg}($'+chr(176)+')')
                    # set x&y axis limits
                    ax[0][1].set_xlim([0,30])
                    ax[0][1].set_ylim([-10,20])
                    ax[1][0].set_xlim([0,30])
                    ax[1][0].set_ylim([-30,90])
                    # ax[1][1].set_xlim([])
                    ax[1][1].set_ylim([-30,90])

            t += 30 # every step is 30 seconds
    return event_theta

def h5_twine_initiation_compensation(event,target_exp,target_event,
                                     h5_dict,track_dict):
    '''from track file get starting image number,
    (from events-class get index of contact index,)
    from h5 file name get number of first analyzed image,
    return difference between h5 start and first frame'''
    try:

        exp_event_list = [re.findall('interekt_'+str(target_exp)+'_e_'
                         +str(target_event)+'.*',file[0])
                          for file in h5_dict.values()] # h5 files for given event
        exp_event_list = [sublist[0] for sublist
                          in exp_event_list if sublist] # remove empty

        h5_first_image = int(re.findall('\d{3,5}-',
                            exp_event_list[0])[0].replace('-','')) # get 1st file
        all_track_files = list(track_dict.values())
        # get the track file of target exp-event
        track_file = [file[0] for file in all_track_files
                    if re.findall(f'0{target_exp}',file[0].replace('\\','_'))
                      and re.findall(f'_{target_event}',file[0].replace('\\','_'))
                      and re.findall('side',file[0])][0]
        lines = []
        # open track file and get number of first frame
        with open(track_file, 'r') as file:
            for line in file:
                lines.append(line)
        if len(re.findall('DSC_\d+',lines[0]))>0:
            zero_frame = int(re.findall('DSC_\d+',
                      lines[0])[0][4:])
        else:
            zero_frame = int(re.findall('\d+_CROPED',
                      lines[0])[0].replace('_CROPED',''))

        # dont actually need contact frame
        # from event class get # of frames till contact
        # event_contact_img_num = event.frm0_side

        frms_till_h5 = h5_first_image-zero_frame
        return frms_till_h5
    except Exception as e:
        print(f'error {e}')
        return 0

#%% 6. get point coordinates near support-> get angle relative to horizontal
''' get coordinates of 2 points close to the support on the left in side view,
    or close to support from each side'''
def get_track_stem_near_support(filename):
    with open(filename[0],"r") as data: # file is in list
        lines = data.readlines()
        N = round(np.size(lines,0)/2)
        times = [30*x for x in range(N)]
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

def two_point_vs_horizontal_angle(p1,p2):
    '''get 2 points (x1,y1) and (x2,y2) and return their angle relative to the
    x axis'''
    try:
        a = np.degrees(m.atan((p2[1]-p1[1])/(p2[0]-p1[0])))
        return a
    except:
        return np.nan

def get_track_angle_near_support(near_contact_file_list,exp,event,axis=[]):
    '''get list of files with 2 tracked points: points on either side
    of the support or both close to it on the far side. returns
    times and angles for these points'''
    try:
        times,sup_point,far_point = get_track_stem_near_support(
                                    near_contact_file_list[exp,event])
        alpha = []
        l = len(far_point)
        fig,ax = axis
        for i in range(1,l): # start from 2nd frame
                a = two_point_vs_horizontal_angle(far_point[i],sup_point[i])
                if abs(a)>0:
                    alpha.append(a)
                    if len(ax)>0:
                        ax[1][1].plot(times[i]/(60),a,'bo')
                else: alpha.append(0)
        return times[1:],alpha

    except Exception as e:
        print(f"An error occurred: {e}") # Print the error message
        return 0

#%% 7. get twine initiation time
def twine_initiation(times,angle,twine_threshold=50,resampling=0):
    '''get twine initiation angle,time in seconds
    return: angle, time in minutes, and index (need to double so since i half it
           in resampling) for given threshold'''
    try:
        def start_check(a,i):
            if abs(a[i])<20 and abs(a[i+1]<20):
                return True
            else: return False
        #resampling
        l = len(angle)-1
        target_length = int(l/2)
        if resampling == 1:
            angle,times = resample(angle[:-1],target_length,t=times[:-1])
        start = False
        for i in range(len(angle)-1):
            if not start:
                start = start_check(angle,i)
            else:
                if angle[i] > twine_threshold:
                    return angle[i],times[i]/60,2*i
        return []
    except Exception as e:
        print(f'error: {e}')
        return []
#%%