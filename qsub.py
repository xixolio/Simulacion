# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:58:34 2018

@author: iaaraya
"""

from itertools import product
import subprocess
import sys
import numpy as np

states = [5,10,15,20,25]
lags = [1, 12, 24]
time_steps = [1, 10, 20, 30]
epochs = [10,20,30]

#states = [5]
#lags = [1]
#time_steps = [1]
#epochs = [1]

combs = product(states,lags,time_steps,epochs)
    
for c in combs:
        
        if c:
            
            string = ''
            
            for element in c:
                
                string += str(element) + ' '
            
            string = '/user/i/iaraya/Simulacion/ '+string
            
            subprocess.call(["qsub","main.sh","-F",string])