# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 04:09:59 2018

@author: iaaraya
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 01:25:19 2018

@author: iaaraya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:17:53 2018

@author: iaaraya
"""
import sys
from keras.layers import SimpleRNN, LSTM, Input, Dense, Lambda, Reshape, Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD, RMSprop, Nadam
from multiplicative_lstm import MultiplicativeLSTM
from itertools import product
from periodic_utils import *
from data_processing import get_data
from keras.utils import to_categorical
import numpy as np

states = [20,25]
lags = [24]
time_steps = [30]
epochs = [20,30]
hiddens = [5,10]

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
    hidden = params[4]
    #write_all = bool(sys.argv[6])
    
    
    prev_information = Input(shape=(time_steps,lag))
    x = Input(shape=(24,))
    #ones = Input(shape=(1,))
    
    lstm = LSTM(states, activation ='sigmoid')(prev_information)
    init_mask = input_mask(24,1,hidden, prev_activations = states,initial = True)
    out_mask = input_mask(23,hidden,2, prev_activations = 0,initial = False)
    
    
    concat = Concatenate()([lstm,x])
    dense = MaskedAE(hidden*23,mask=init_mask)(concat)
    dense = Activation('sigmoid')(dense)
    
    dense = MaskedAE(2*23,mask=out_mask)(dense)
    dense2 = Dense(2)(lstm)
    
    concat = Concatenate()([dense2,dense])
    reshaped = Reshape((24,2))(concat)
    outputs =Lambda(made_mu_sigma)(reshaped)
    
    model = Model(inputs=[prev_information, x], outputs=outputs)
    model.compile(loss=made_likelihood, optimizer = Adam())
    
    batches = 500
    wind_input, input_test , wind_output, output_test , min_speeds, max_speeds = get_data('', 'no_mvs_b08.csv',ts=time_steps, lag=lag, train_output=24)
    
    #periods = to_categorical(np.arange(0,len(wind_input))%24, num_classes=24)
    
    # Training
    
    model.fit([wind_input,wind_output],wind_output.reshape(-1,24,1), epochs=epochs,batch_size=batches)
    model_name = 'made_'+str(lag)+'_'+str(time_steps)+'_'+str(states)+ \
    '_'+str(epochs)+'_'+str(batches)+'_'+str(hidden)
    
    # Testing
    
    
    
    n_simulations = 100
    horizon = 24
    
    #period_offset = len(wind_input) + lag + time_steps - 1
    #periods = np.arange(period_offset,period_offset+len(input_test))%24
    
    predictions, mse_per_horizon, upper_bounds, lower_bounds, CWC_per_horizon, CWC =\
    made_predict(n_simulations,horizon,input_test,model,model_name,\
                     '',output_test)
    
    
    
    np.savetxt('forecasting_results/CWC_horizons_'+model_name+'.txt', np.array(CWC_per_horizon))
    np.savetxt('forecasting_results/CWC_'+model_name+'.txt', np.array(CWC).reshape(1,1))
    np.savetxt('forecasting_results/mse_'+model_name+'.txt', np.array(mse_per_horizon))
       
    np.savetxt('forecasting_results/predictions_'+model_name+'.txt', np.array(predictions))
    np.savetxt('forecasting_results/upper_bounds_'+model_name+'.txt', np.array(upper_bounds))
    np.savetxt('forecasting_results/lower_bounds_'+model_name+'.txt', np.array(lower_bounds))
    
    
    
    
    
    
    
