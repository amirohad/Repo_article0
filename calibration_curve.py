'''plot data
Created by Amir Ohad 2/9/2024'''
#%%
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%%
# Read the CSV file
filename = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\Calibrations\20200922 Force calibration 2 pulley+weights\force_calibration_curve.csv"
data = pd.read_csv(filename)

# Extract the columns you want to plot
# get column names and save to 2 variables
column_names = data.columns
x_name = column_names[0]
y_name = column_names[1]
x = data[x_name]
y = data[y_name]

# Plot the data
plt.plot(x, y,'x',label='Data')
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.title('Data Plot')
plt.show()

# fit a line to the data
coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(coefficients)
ys = polynomial(x)
plt.plot(x, y, 'x', x, ys, label='Fit')

# place the equation on the plot with r squared on the bottom right
r_squared = 1 - (np.sum((y - ys) ** 2) / np.sum((y - np.mean(y)) ** 2))
plt.text(0.5, 0.5, rf'{polynomial:.2f}\n $R^{2}={r_squared:.2f}$',
        horizontalalignment='right', verticalalignment='bottom',
        transform=plt.gca().transAxes)


#%%