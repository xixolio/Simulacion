# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:31:38 2018

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
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism=8, inter_op_parallelism_threads=8)))

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
    
    if states != 25 or lags == 1 or time_steps == 1 or epochs == 1:
        continue
    
    wind_input, _ , wind_output, _ , min_speeds, max_speeds = get_data('', 'no_mvs_b08.csv',ts=time_steps, lag=lag)
    model_name = 'periodic_mu_sigma_'+str(lag)+'_'+str(time_steps)+'_'+str(states)+ \
    '_'+str(epochs)+'_'+str(batches)+'.h5'
    model = load_model('models/'+model_name,custom_objects={'likelihood': likelihood})
    
    input_train, input_test ,_ , output_test , min_speeds, max_speeds = get_data('', 'no_mvs_b08.csv',ts=time_steps, lag=lag)
    
    #n_val = int(len(input_test)/2)
    val_input = input_test
    val_output = output_test
    
    n_simulations = 100
    horizon = 24

    period_offset = len(input_train) + lag + time_steps - 1
    periods = np.arange(period_offset,period_offset+len(val_input))%24
    
    #periods = periods[-len(val_input):]
    
    predictions, mse_per_horizon, upper_bounds, lower_bounds, CWC_per_horizon, CWC =\
    periodic_predict(n_simulations,horizon,val_input,periods,model,model_name,\
                     '',val_output)
    
    
    np.savetxt('forecasting_results/predictions_'+model_name+'.txt', np.array(predictions))
    np.savetxt('forecasting_results/upper_bounds_'+model_name+'.txt', np.array(upper_bounds))
    np.savetxt('forecasting_results/lower_bounds_'+model_name+'.txt', np.array(lower_bounds))
    np.savetxt('forecasting_results/CWC_horizons_'+model_name+'.txt', np.array(CWC_per_horizon))
    np.savetxt('forecasting_results/CWC_'+model_name+'.txt', np.array(CWC).reshape(1,1))
    np.savetxt('forecasting_results/mse_'+model_name+'.txt', mse_per_horizon)
