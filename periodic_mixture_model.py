# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:12:25 2018

@author: iaaraya
"""

from keras.layers import SimpleRNN, LSTM, Input, Dense, Lambda, Reshape, Concatenate
from keras.models import Model
from keras.optimizers import Adadelta
from periodic_utils import *
from data_processing import get_data

time_steps = 10
states = 20
mixtures = 3

inputs1 = Input(shape=(time_steps,1))
inputs2 = Input(shape=(24,1,1))

srnn = LSTM(states, activation ='sigmoid')(inputs1)
dense1 = Dense(2*24*mixtures)(srnn)
dense2 = Dense(24*mixtures, activation ='softmax')(srnn)

reshaped1 = Reshape((24,mixtures,2))(dense1)
reshaped2 = Reshape((24,mixtures,1))(dense2)

concatenated = Concatenate(axis=-1)([reshaped1,reshaped2])

params = Lambda(get_periodic_params_mixture)([concatenated,inputs2])
outputs = Lambda(mu_sigma_mixture, name = 'outputs')(params)
model = Model(inputs=[inputs1,inputs2], outputs=outputs)
model.compile(loss=gaussian_mixture_likelihood, optimizer = Adadelta())
#%%
wind_input, _ , wind_output, _ , min_speeds, max_speeds = get_data('', 'no_mvs_b08.csv',ts=time_steps)
#wind_input, _ , wind_output, _ , min_speeds, max_speeds = get_data('', 'no_mvs_e01.csv',ts=time_steps)
wind_input = wind_input[0]
wind_output = wind_output[0]
wind_input.shape
#%%
periods = to_categorical(np.arange(0,len(wind_input))%24, num_classes=24).reshape(-1,24,1,1)
model.fit([wind_input,periods],wind_output,epochs=1,batch_size=2)