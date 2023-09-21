#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:42:50 2023

@author: jing
"""

#mobile price categorization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

import os
abspath = os.path.abspath(__file__) #current file path
dname = os.path.dirname(abspath)#the directory of this file, dataset is also here
os.chdir(dname)

#%%
train = pd.read_csv('./mobileprice_dataset/train.csv')
test= pd.read_csv('./mobileprice_dataset/test.csv')

train.info()

#%%
#visualize data
visualize = train.copy()

visualize.plot(kind = 'scatter', x = 'battery_power',y = 'price_range', 
               grid = True, legend = True,
               figsize = (10,7),
               alpha = 1)
             #s = housing['population']/100, label = 'Population', 
             #c = 'median_house_value', cmap = 'jet', colorbar = True,
             #legend = True, sharex = False, figsize = (10,7))

plt.show()
#%%
#linear correlation
sns.color_palette("Set2")
#visualize.corr()
#plt.matshow(visualize.corr())
#plt.show()

#sns.heatmap(visualize.corr(), cbar=True, annot=None)

####
sns.relplot(visualize, x = 'px_height', y = 'px_width', hue='price_range', palette='pastel')
sns.relplot(visualize, x = 'battery_power', y = 'ram', hue='price_range', palette='pastel')
sns.relplot(visualize, x = 'clock_speed', y = 'ram', hue='price_range', palette='pastel')
temp = visualize.iloc[:,5:]
sns.pairplot(data=temp, hue='price_range', palette= 'pastel')
#%%
#with train.info() we can see there is no null item
#and all data type is in int or float
#-> not data manipulation needed
#%%
#do i want to normalize the data before running knn?
#lets try it without normalization first and see the result.
#we doo need normalization, self write or library?
#standard scalar is good because here doesn't seem to be an outlier!

train_scale = StandardScaler().fit_transform(train)
#%%











