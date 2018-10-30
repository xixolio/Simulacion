# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:38:01 2018

@author: iaaraya
"""

from keras.layers import SimpleRNN, LSTM, Input, Dense, Lambda, GaussianNoise
from keras.models import Model
from keras.optimizers import Adadelta,SGD, Adam
from multiplicative_lstm import MultiplicativeLSTM
from itertools import product
from periodic_utils import *
from data_processing import get_data

states = [5,10,15,20,25]
lags = [1, 12, 24]
time_steps = [1, 10, 20, 30]
epochs = [10,20,30]

combinations = product(states, lags, time_steps,epochs)
parameters = [params for params in combinations]

for i,params in enumerate(parameters):
    
    print(str(i+1)+' out of '+str(len(parameters)))
    states = params[0]
    lag = params[1]
    time_steps = params[2]
    epochs = params[3]
    
    inputs = Input(shape=(time_steps,lag))
    lstm = LSTM(states, activation ='sigmoid')(inputs)
    dense = Dense(2)(lstm)
    outputs = Lambda(mu_sigma)(dense)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=likelihood, optimizer = Adam())
    
    batches = 500
    wind_input, _ , wind_output, _ , min_speeds, max_speeds = get_data('', 'no_mvs_b08.csv',ts=time_steps, lag=lag)

    model.fit(wind_input,wind_output,epochs=epochs,batch_size=batches)
    model_name = 'mu_sigma_'+str(lag)+'_'+str(time_steps)+'_'+str(states)+ \
    '_'+str(epochs)+'_'+str(batches)+'.h5'
    model.save('models/'+model_name)