# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 04:53:02 2018

@author: iaaraya
"""
import pandas as pd
import statsmodels.tsa.seasonal
import statsmodels as st
#import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy
from data_processing import get_data



input_train, input_test ,output_train, output_test , min_speeds, max_speeds = get_data('', 'no_mvs_b08.csv',ts=1, lag=1)
    
input_test = input_test.flatten()

model2 = SARIMAX(input_test, order = (4,0,0), seasonal_order = (0,1,3,24))
res = model2.filter(np.array([ 0.85145337, -0.19835504,  0.05774695,  0.02914752, -0.77779017, -0.08159543, -0.04684304,  0.00701107]))
sarima_predictions = []
sarima_upper_bound = []
sarima_lower_bound = []
    
for i in range(100,len(input_test)):
    if i%100==0:
        print(i)
    prediction = res.get_prediction(start=i,end=i+23,dynamic=0)
    sarima_predictions.append(prediction.predicted_mean)
    confidence = prediction.conf_int()
    sarima_upper_bound.append(confidence[:,1])
    sarima_lower_bound.append(confidence[:,0])
np.savetxt('forecasting_results/sarima_predictions.txt', sarima_predictions)
np.savetxt('forecasting_results/sarima_uppers.txt', sarima_upper_bound)
np.savetxt('forecasting_results/sarima_lowers.txt', sarima_lower_bound)


