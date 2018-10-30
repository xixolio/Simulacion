# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:27:54 2018

@author: iaaraya
"""
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

def metrics_and_plots(wind_data,simulated_data):
    
    bins = np.linspace(0,1,30)
    n_simulations = simulated_data.shape[0]
    histogram_mse = np.zeros(n_simulations)
    acf_mse = np.zeros((n_simulations,101))
    pacf_mse = np.zeros((n_simulations,101))
    
    h1 = np.histogram(wind_data, bins)[0]/len(wind_data)
    acf1 = acf(wind_data, nlags=100)
    pacf1 = pacf(wind_data, nlags=100)

    
    for i in range(n_simulations):
        
        h2 = np.histogram(simulated_data[i,:], bins)[0]/len(simulated_data[i,:])
        histogram_mse[i] = np.mean((h1-h2)**2)
        
        acf2 = acf(simulated_data[i,:], nlags=100)
        pacf2 = pacf(simulated_data[i,:], nlags=100)
        acf_mse[i,:] = (acf1-acf2)**2
        pacf_mse[i,:] = (pacf1-pacf2)**2
        
    mean_hist_mse = np.array(np.mean(histogram_mse))
    mean_acf_mse = np.mean((acf_mse), axis=0)
    mean_pacf_mse = np.mean((pacf_mse), axis=0)

    
    return mean_hist_mse, mean_acf_mse, mean_pacf_mse
    
    
        
    
    