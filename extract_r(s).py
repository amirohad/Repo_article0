'''# -*- coding: utf-8 -*-
Created on Thur Nov 14 16:38:45 2024

script to extract radius of stem as function of length from tip
* go through the whole folder, folder by folder.
* get pix to cm ratio
* get outline
* how to extract radius from it? can use interect? 
* stitch 2 parts together? or keep just the known part?


@author: Amir Ohad
'''
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
#%% pix2cm ratio, open df
# Define the main folder path
main_folder_path = r'C:\Users\Amir\Documents\PHD\Experiments\Bean Growth\r(s)'

# ask whether to start new df
if input('Start new dataframe? (y/n)') == 'y':
    df = pd.DataFrame({
        'Folder': [],
        'Image': [],
        'Radius(cm)': [],
        'pix2cm (cm/pix)': []
    })
else:
    df = pd.read_excel(os.path.join(main_folder_path, 'r(s).xlsx'))

# Create a dataframe to store the results
df = pd.DataFrame({
    'Folder': [],
    'Image': [],
    'Radius(cm)': [],
    'pix2cm (cm/pix)': []
})
# Prompt the user to choose a starting folder
print("Subfolders in the main folder:")
subfolders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
for i, folder in enumerate(subfolders):
    print(f"{i}: {folder}")

start_index = int(input("Enter the number of the folder to start with: "))
subfolders = subfolders[start_index:]
# Iterate through each subfolder in the main folder
for subfolder in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        # Iterate through each image in the subfolder
        for image_file in glob.glob(os.path.join(subfolder_path, '*.jpg')):  # Assuming images are in .jpg format

            # Get the pix2cm ratio using the function from useful_functions
            pix_size = uf.multiline_dist(image_file,'select object of known length')
            real_size_cm = input('Enter the real size of the object in cm: ')
            pix2cm = float(real_size_cm) / pix_size # multiply pix by this to get cm

            # Append the results to the dataframe
            df = df.append({
                'Folder': subfolder,
                'Image': image_file,
                'Radius(cm)': None,
                'pix2cm (cm/pix)': pix2cm
            }, ignore_index=True)

# Save the dataframe to an excel file
output_file = os.path.join(subfolder_path, 'r(s).xlsx')

#%%