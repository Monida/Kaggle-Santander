# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:59:55 2019

@author: Monica Daniela
"""
import numpy as np
import pandas as pd

def right_format (predictions,N,threshold):
    threshold = threshold
    predictions = np.where(predictions<threshold,0,1)
    number_of_ones = np.count_nonzero(predictions==1)
    
    idx=["ID_code"]
    idx[1:N]=["test_" + str(i) for i in range(N)]
    predicted_values = ["target"]
    predicted_values[1:N] = [predictions[i] for i in range(N)]
    predicted_values[0:5]
    results = [[idx[i],predicted_values[i]] for i in range(N+1)]
    results[0:5]
    predictions_df=pd.DataFrame(results[1:],columns=results[0])
    
    
    return (predictions_df,number_of_ones)