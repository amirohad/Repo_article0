# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:09:39 2020

error estimates file
resolution uncertainty = res/sqrt12
measure with pixels-> estimate based on movement with no movement
                      ,during movement and estimate how far off it is from target center?
dependent variable errors calculated with relative error formula

@author: Amir
"""


#%%
import math as m

dl_cm=0.1/m.sqrt(12) # ruler resolution
dl_pix= 2/m.sqrt(12) # initial estimate based on difficulty to find edge-1 each side
dm_sup=0.01/m.sqrt(12) # scale resolution
ddist_pix=2 # initial estimate
dg_cgs=0.1 # last digit for g=980.7
dl_sup_pix=2*dl_pix
dl_sup_cm=2*dl_cm
dx_cm=3 # range of pixels contact point could be at
