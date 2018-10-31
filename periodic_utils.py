# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:50:37 2018

@author: iaaraya
"""

from keras.utils import to_categorical
import keras.backend as K
import numpy as np

def likelihood(y_true, params):
    
    # Works also for periodic mu sigma models.
    
    
    mu = K.expand_dims(params[:,0])
    sigma = K.expand_dims(params[:,1])
    #print(sigma.shape)
    return -K.mean(-(y_true - mu)**2/(2*sigma**2)  - K.log(sigma))
    #return -K.mean(-(y_true - mu)**2)
    #return K.mean(K.log(sigma))
    

    
def gaussian_mixture_likelihood(y_true, params):
    
    #params is of dimension (N,k,3), with N examples and k gaussians.
    
    mus = K.expand_dims(params[:,:,0])
    sigmas = K.expand_dims(params[:,:,1])
    weight = K.expand_dims(params[:,:,2])
    
    return K.mean(K.log(K.sum(weight * 1/K.sqrt(2*np.pi*sigmas**2)*K.exp(-(y_true - mus)**2/(2*sigmas**2)), axis = 1)))
    
def mu_sigma(x): 
    
    # Also works for periodic mu sigma models.
    
    mu = x[:,0]
    sigma = K.log(1 + K.exp(x[:,1]))   
    return K.stack((mu,sigma),axis=1)

def get_periodic_params(x):
    
    [periodic_params,period] = x
    mu_params = periodic_params[:,:,0]
    sigma_params = periodic_params[:,:,1]
    
    mu_param = K.sum(mu_params*period, axis=1)
    sigma_param = K.sum(sigma_params*period, axis=1)
    params = K.stack((mu_param,sigma_param),axis=1)
    return params
    
def get_periodic_params_mixture(x):
    
    # period needs to be of shape N * periods * 1 * 1
    # mixture perdioc params is of shape N * periods * k * 3
    [mixture_periodic_params, period] = x
    mixture_params = K.sum(mixture_periodic_params * period, axis=1)
    
    # now mixture params is of shape N*k*3, having selected only
    # the corresponding period params
    
    return mixture_params

def mu_sigma_mixture(x):
    
    mu = x[:,:,0]
    sigma = K.log(1 + K.exp(x[:,:,1]))
    weight = x[:,:,2]
    return K.stack((mu,sigma,weight),axis=2)

def simulate(n_simulations,simulation_length,initial_data,model,model_name):
    

    x = np.zeros((n_simulations, simulation_length))
    lag = initial_data.shape[-1]
    time_steps = initial_data.shape[-2]
    current_input = initial_data.reshape(1,-1)
    #print(current_input)
    current_input = np.repeat(current_input,n_simulations,axis=0).reshape(n_simulations,time_steps,lag)
    
    for i in range(simulation_length):
            #print("imprimiendo actual")
            #print(current_input)
            #print(periods[:,i,:])
            params = model.predict(current_input)
            mu, sigma = params[:,0],params[:,1]
            x[:,i] = np.random.normal(mu,sigma)
            #print("mostrando x")
            #print(mu)
            #print(sigma)
            #print(x)
            if time_steps != 1:
                values = np.concatenate((current_input[:,0,:].reshape(-1,lag),current_input[:,1:,-1].reshape(-1,time_steps-1)),axis=1)
                values = np.concatenate((values[:,1:], x[:,i].reshape(-1,1)), axis=1)   
                current_input = np.concatenate([values[:,t:lag + t]  for t in range(time_steps)],axis=1)
                current_input = current_input.reshape(-1, time_steps, lag)
            elif time_steps == 1 and lag != 1:
                values = np.concatenate((current_input.reshape(-1,lag)[:,1:],x[:,i].reshape(-1,1)),axis=1)
                current_input = values.reshape(-1,time_steps,lag)
               
            else:
                current_input = x[:,i].reshape(-1,1,1)
                
    np.savetxt('simulations/'+model_name+'.txt',x)
    return x

def periodic_simulate(n_simulations,simulation_length,initial_data,initial_period,model,model_name):
    
    periods = to_categorical(np.arange(initial_period, initial_period+simulation_length)%24, num_classes=24)
    x = np.zeros((n_simulations, simulation_length))
    lag = initial_data.shape[-1]
    time_steps = initial_data.shape[-2]
    current_input = initial_data.reshape(1,-1)
    #print(current_input)
    current_input = np.repeat(current_input,n_simulations,axis=0).reshape(n_simulations,time_steps,lag)
    periods = np.repeat(periods.reshape(1,-1),n_simulations,axis=0).reshape(-1,simulation_length,24)
    
    for i in range(simulation_length):
            #print("imprimiendo actual")
            #print(current_input)
            #print(periods[:,i,:])
            params = model.predict([current_input,periods[:,i,:].reshape(-1,24)])
            mu, sigma = params[:,0],params[:,1]
            x[:,i] = np.random.normal(mu,sigma)
            #print("mostrando x")
            #print(mu)
            #print(sigma)
            #print(x)
            if time_steps != 1:
                values = np.concatenate((current_input[:,0,:].reshape(-1,lag),current_input[:,1:,-1].reshape(-1,time_steps-1)),axis=1)
                values = np.concatenate((values[:,1:], x[:,i].reshape(-1,1)), axis=1)   
                current_input = np.concatenate([values[:,t:lag + t]  for t in range(time_steps)],axis=1)
                current_input = current_input.reshape(-1, time_steps, lag)
            elif time_steps == 1 and lag != 1:
                values = np.concatenate((current_input.reshape(-1,lag)[:,1:],x[:,i].reshape(-1,1)),axis=1)
                current_input = values.reshape(-1,time_steps,lag)
               
            else:
                current_input = x[:,i].reshape(-1,1,1)
                
    np.savetxt('simulations/'+model_name+'.txt',x)
    return x

def periodic_predict(n_simulations, horizon, data,periods,model,model_name,path,true_outputs):
    
    '''Parameters
    n_simulations (int): how many times each predictions is going to be sampled
    horizon (int): how many hours ahead the prediction is computed
    data (n_samples,time_steps,lag): wind_data
    periods (n_samples): a vector with a number between 0 and 23 indicating the sample initial period
    model: trained keras model
    model_name (string): model name
    path (string): something like "results/"
    true_outputs (n_samples,24): true outputs to be used for comparison against predictions
    '''
    
    #periods = to_categorical(np.arange(initial_period, initial_period+simulation_length)%24, num_classes=24)
    periods = [to_categorical((periods+i)%24,num_classes=24) for i in range(horizon)]
    periods = np.concatenate(periods,axis=1).reshape(-1,horizon,24)
    
    x = np.zeros((len(data), horizon))
    predictions = np.zeros((n_simulations,len(data), horizon))
    lag = data.shape[-1]
    time_steps = data.shape[-2]
    current_input = data
    
    for j in range(n_simulations):
        #print(j)
        current_input = data
        for i in range(horizon):
                #print("imprimiendo actual")
                #print(current_input)
                #print(periods[:,i,:])
                params = model.predict([current_input,periods[:,i,:].reshape(-1,24)])
                mu, sigma = params[:,0],params[:,1]
                x[:,i] = np.random.normal(mu,sigma)
                #print("mostrando x")
                #print(mu)
                #print(sigma)
                #print(x)
                if time_steps != 1:
                    values = np.concatenate((current_input[:,0,:].reshape(-1,lag),current_input[:,1:,-1].reshape(-1,time_steps-1)),axis=1)
                    values = np.concatenate((values[:,1:], x[:,i].reshape(-1,1)), axis=1)   
                    current_input = np.concatenate([values[:,t:lag + t]  for t in range(time_steps)],axis=1)
                    current_input = current_input.reshape(-1, time_steps, lag)
                elif time_steps == 1 and lag != 1:
                    values = np.concatenate((current_input.reshape(-1,lag)[:,1:],x[:,i].reshape(-1,1)),axis=1)
                    current_input = values.reshape(-1,time_steps,lag)
                   
                else:
                    current_input = x[:,i].reshape(-1,1,1)
        predictions[j,:,:] = x
              
    # fixed 90% confidence interval
    upper_bounds = np.percentile(predictions,95,axis=0)
    lower_bounds = np.percentile(predictions,5,axis=0)
    
    PICP_per_horizon = np.mean((true_outputs>lower_bounds)*(true_outputs<upper_bounds),axis=0)
    PICP = np.mean(PICP_per_horizon)
    
    MPIW_per_horizon = np.mean(upper_bounds - lower_bounds,axis=0)
    MPIW = np.mean(MPIW_per_horizon)
    u = 0.9
    n = 5
    y_per_horizon = PICP_per_horizon < u
    CWC_per_horizon = MPIW_per_horizon*(1+y_per_horizon*np.exp(-n*(PICP_per_horizon-u)))
    y = PICP < u
    CWC = MPIW*(1+y*np.exp(-n*(PICP-u)))
    
    predictions = np.mean(predictions,axis=0)
    mse_per_horizon = np.mean((predictions.reshape(-1,24) - true_outputs.values.reshape(-1,24))**2,axis=0)
    return predictions, mse_per_horizon, upper_bounds, lower_bounds, CWC_per_horizon, CWC
    
def predict(n_simulations, horizon, data,model,model_name,path,true_outputs):
    
    '''Parameters
    n_simulations (int): how many times each predictions is going to be sampled
    horizon (int): how many hours ahead the prediction is computed
    data (n_samples,time_steps,lag): wind_data
    model: trained keras model
    model_name (string): model name
    path (string): something like "results/"
    true_outputs (n_samples,24): true outputs to be used for comparison against predictions
    '''
    
    #periods = to_categorical(np.arange(initial_period, initial_period+simulation_length)%24, num_classes=24)
    #periods = [to_categorical((periods+i)%24,num_classes=24) for i in range(horizon)]
    #periods = np.concatenate(periods,axis=1).reshape(-1,horizon,24)
    
    x = np.zeros((len(data), horizon))
    predictions = np.zeros((n_simulations,len(data), horizon))
    lag = data.shape[-1]
    time_steps = data.shape[-2]
    current_input = data
    
    for j in range(n_simulations):
        current_input = data
        for i in range(horizon):
                #print("imprimiendo actual")
                #print(current_input)
                #print(periods[:,i,:])
                params = model.predict(current_input)
                mu, sigma = params[:,0],params[:,1]
                x[:,i] = np.random.normal(mu,sigma)
                #print("mostrando x")
                #print(mu)
                #print(sigma)
                #print(x)
                if time_steps != 1:
                    values = np.concatenate((current_input[:,0,:].reshape(-1,lag),current_input[:,1:,-1].reshape(-1,time_steps-1)),axis=1)
                    values = np.concatenate((values[:,1:], x[:,i].reshape(-1,1)), axis=1)   
                    current_input = np.concatenate([values[:,t:lag + t]  for t in range(time_steps)],axis=1)
                    current_input = current_input.reshape(-1, time_steps, lag)
                elif time_steps == 1 and lag != 1:
                    values = np.concatenate((current_input.reshape(-1,lag)[:,1:],x[:,i].reshape(-1,1)),axis=1)
                    current_input = values.reshape(-1,time_steps,lag)
                   
                else:
                    current_input = x[:,i].reshape(-1,1,1)
        predictions[j,:,:] = x
              
    # fixed 90% confidence interval
    upper_bounds = np.percentile(predictions,95,axis=0)
    lower_bounds = np.percentile(predictions,5,axis=0)
    
    PICP_per_horizon = np.mean((true_outputs>lower_bounds)*(true_outputs<upper_bounds),axis=0)
    PICP = np.mean(PICP_per_horizon)
    
    MPIW_per_horizon = np.mean(upper_bounds - lower_bounds,axis=0)
    MPIW = np.mean(MPIW_per_horizon)
    u = 0.9
    n = 5
    y_per_horizon = PICP_per_horizon < u
    CWC_per_horizon = MPIW_per_horizon*(1+y_per_horizon*np.exp(-n*(PICP_per_horizon-u)))
    y = PICP < u
    CWC = MPIW*(1+y*np.exp(-n*(PICP-u)))
    
    predictions = np.mean(predictions,axis=0)
    mse_per_horizon = np.mean((predictions.reshape(-1,24) - true_outputs.values.reshape(-1,24))**2,axis=0)
    return predictions, mse_per_horizon, upper_bounds, lower_bounds, CWC_per_horizon, CWC
    
def input_mask(inputs,units_per_input,units_per_output, prev_activations = 0,initial = False):
    
    #output_dim = activations_per_input*(input_dim-1)      
        
    # si es una capa de entrada, se ignora el último valor del input para lograr la activación.
    # si es una capa intermedia o de salida, el input y el output son iguales.
    # ver cómo manejar el caso de la capa final
    if initial == False:
        
        #assert inputs == outputs
        assert prev_activations == 0
        outputs = inputs
        
    else:
        
        outputs = inputs - 1
        
    mask = np.ones((prev_activations+inputs*units_per_input,outputs*units_per_output))

    for i in range(inputs):
        
        if i == 0:
            
             mask[0:units_per_input+prev_activations,:units_per_output*i] = 0
        
        else:
            
            mask[prev_activations+i*units_per_input:prev_activations+i*units_per_input+units_per_input,:units_per_output*i] = 0 

    return mask

def made_likelihood(y_true, params):
    
    print(y_true.shape)
    mu = K.expand_dims(params[:,:,0])
    sigma = K.expand_dims(params[:,:,1])
    
    return -K.mean(-(y_true - mu)**2/(2*sigma**2)  - K.log(sigma))
    #return K.mean((y_true - mu)**2)
    #return K.mean(K.square(params - y_true), axis=-1)

def made_mu_sigma(x): 
    
    mu = x[:,:,0]
    sigma = K.log(1 + K.exp(x[:,:,1]))   
    return K.stack((mu,sigma),axis=2)

from keras import backend as K
from keras.engine.topology import Layer

class MaskedAE(Layer):

    def __init__(self, output_dim, mask, **kwargs):
        self.output_dim = output_dim
        self.mask = mask
        super(MaskedAE, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MaskedAE, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        weights = self.kernel * self.mask
        return K.dot(x, weights)

    def compute_output_shape(self, input_shape):
        print(input_shape[0], self.output_dim)
        return (input_shape[0], self.output_dim)
    
    
    

def made_predict(n_simulations, horizon, data,model,model_name,path,true_outputs):  
    '''Parameters
    n_simulations (int): how many times each predictions is going to be sampled
    horizon (int): how many hours ahead the prediction is computed
    data (n_samples,time_steps,lag): wind_data
    model: trained keras model
    model_name (string): model name
    path (string): something like "results/"
    true_outputs (n_samples,24): true outputs to be used for comparison against predictions
    '''
    
    #periods = to_categorical(np.arange(initial_period, initial_period+simulation_length)%24, num_classes=24)
    #periods = [to_categorical((periods+i)%24,num_classes=24) for i in range(horizon)]
    #periods = np.concatenate(periods,axis=1).reshape(-1,horizon,24)
    horizon = 24

    x = np.zeros((len(data), horizon))
    predictions = np.zeros((n_simulations,len(data), horizon))
    lag = data.shape[-1]
    time_steps = data.shape[-2]
    current_input = data
    
    for j in range(n_simulations):
        #print(j)
        current_input = data
        x = np.zeros((len(data), horizon))
        for i in range(horizon):
                #print("imprimiendo actual")
                #print(current_input)
                #print(periods[:,i,:])
                params = model.predict([current_input,x])
                mu, sigma = params[:,i,0],params[:,i,1]
                x[:,i] = np.random.normal(mu,sigma)
                #print("mostrando x")
                #print(mu)
                #print(sigma)
                #print(x)
                
        predictions[j,:,:] = x
    
    # fixed 90% confidence interval
    upper_bounds = np.percentile(predictions,95,axis=0)
    lower_bounds = np.percentile(predictions,5,axis=0)
    
    PICP_per_horizon = np.mean((true_outputs>lower_bounds)*(true_outputs<upper_bounds),axis=0)
    PICP = np.mean(PICP_per_horizon)
    
    MPIW_per_horizon = np.mean(upper_bounds - lower_bounds,axis=0)
    MPIW = np.mean(MPIW_per_horizon)
    u = 0.9
    n = 5
    y_per_horizon = PICP_per_horizon < u
    CWC_per_horizon = MPIW_per_horizon*(1+y_per_horizon*np.exp(-n*(PICP_per_horizon-u)))
    y = PICP < u
    CWC = MPIW*(1+y*np.exp(-n*(PICP-u)))
    
    predictions = np.mean(predictions,axis=0)
    mse_per_horizon = np.mean((predictions.reshape(-1,24) - true_outputs.values.reshape(-1,24))**2,axis=0)
    return predictions, mse_per_horizon, upper_bounds, lower_bounds, CWC_per_horizon, CWC
    
    
    
        
        
        