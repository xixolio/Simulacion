# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:25:55 2018

@author: iaaraya
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:15:03 2018

@author: iaaraya
"""
from keras.models import load_model
from keras.layers import SimpleRNN, LSTM, Input, Dense, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD, RMSprop, Nadam
from multiplicative_lstm import MultiplicativeLSTM
from itertools import product
from periodic_utils import *
from data_processing import get_data
from keras.utils import to_categorical
import numpy as np
from simulation_metrics import metrics_and_plots

states = [5,10,15,20,25]
lags = [1, 12, 24]
time_steps = [1, 10, 20, 30]
epochs = [10,20,30]

combinations = product(states, lags, time_steps,epochs)
parameters = [params for params in combinations]
models = []
data = []

for i,params in enumerate(parameters):
    
    print(str(i+1)+' out of '+str(len(parameters)))
    states = params[0]
    lag = params[1]
    time_steps = params[2]
    epochs = params[3]
    batches = 500
    
    wind_input, _ , wind_output, _ , min_speeds, max_speeds = get_data('', 'no_mvs_b08.csv',ts=time_steps, lag=lag)
    model_name = 'mu_sigma_'+str(lag)+'_'+str(time_steps)+'_'+str(states)+ \
    '_'+str(epochs)+'_'+str(batches)+'.h5'
    model = load_model('models/'+model_name,custom_objects={'likelihood': likelihood})
    
    wind_data, _ , _, _ , min_speeds, max_speeds = get_data('', 'no_mvs_b08.csv',ts=1, lag=1)
    wind_data = wind_data.reshape(-1)
    n_simulations = 100
    simulation_length = 2000
    simulated_data = simulate(n_simulations,simulation_length,wind_input[0],model,model_name)
    
    mean_hist, mean_acf, mean_pacf = metrics_and_plots(wind_data, simulated_data)
    
    results = [mean_hist]
    for element in mean_acf:
        results.append(element)
    for element in mean_pacf:
        results.append(element)
        
    np.savetxt('simulation_results/'+model_name+'.txt', np.array(results))
