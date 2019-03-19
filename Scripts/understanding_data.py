"""
Kaggle - santander

This .py document contains scripts dedicated to understanding the data
"""

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_set = pd.read_csv('../Data/train.csv')
test_set = pd.read_csv('../Data/test.csv')

x_train=train_set.iloc[:,2:202]
y_train=train_set.iloc[:,1]

x_test=test_set.iloc[:,1:201]

plt.subplot(200,1,1)
plt.scatter(x_train.iloc[:,0],x_train.iloc[:,1])
plt.subplot(200,1,2)
plt.scatter(x_train.iloc[:,0],x_train.iloc[:,2])
plt.show
'''
Evaluate autocorrelation in variables
'''

'''
Min max normalization
'''