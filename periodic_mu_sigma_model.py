# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:17:53 2018

@author: iaaraya
"""

from keras.layers import SimpleRNN, LSTM, Input, Dense, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD, RMSprop, Nadam
from multiplicative_lstm import MultiplicativeLSTM
from itertools import product
from periodic_utils import *
from data_processing import get_data
from keras.utils import to_categorical
import numpy as np

states = [5,10,15,20,25]
lags = [1, 12, 24]
time_steps = [1, 10, 20, 30]
epochs = [10,20,30]

combinations = product(states, lags, time_steps,epochs)
parameters = [params for params in combinations]

for i,params in enumerate(parameters):

    states = params[0]
    lag = params[1]
    time_steps = params[2]
    epochs = params[3]
    
    inputs1 = Input(shape=(time_steps,lag)) #receives the actual time_series
    inputs2 = Input(shape=(24,)) # receives the one-hot-label vector
    lstm = LSTM(states, activation ='sigmoid')(inputs1)
    #srnn = Dense(states, activation ='relu')(inputs1)
    dense = Dense(2*24)(lstm) # 24 mu and sigmas, one for each period
    reshaped = Reshape((24,2))(dense)
    params = Lambda(get_periodic_params)([reshaped,inputs2]) # extracts the corresponding period activations
    outputs = Lambda(mu_sigma)(params)
    model = Model(inputs=[inputs1,inputs2], outputs=outputs)
    model.compile(loss=likelihood, optimizer = Adam())
    
    batches = 500
    wind_input, _ , wind_output, _ , min_speeds, max_speeds = get_data('', 'no_mvs_b08.csv',ts=time_steps, lag=lag)

    periods = to_categorical(np.arange(0,len(wind_input))%24, num_classes=24)
    
    model.fit([wind_input,periods],wind_output,epochs=epochs,batch_size=batches)
    model_name = 'periodic_mu_sigma_'+str(lag)+'_'+str(time_steps)+'_'+str(states)+ \
    '_'+str(epochs)+'_'+str(batches)+'.h5'
    model.save('models/'+model_name)