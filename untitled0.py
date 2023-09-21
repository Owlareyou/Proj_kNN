#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:42:50 2023

@author: jing
"""

#mobile price categorization
import numpy as np
import pandas as pd

import os
abspath = os.path.abspath(__file__) #current file path
dname = os.path.dirname(abspath)#the directory of this file, dataset is also here
os.chdir(dname)

#%%
train_mobiledata = pd.read_csv('./mobileprice_dataset/train.csv')
test_mobiledata= pd.read_csv('./mobileprice_dataset/test.csv')



