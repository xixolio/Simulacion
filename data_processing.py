# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:36:23 2018

@author: iaaraya
"""

import pandas as pd
import math
import numpy as np

""" Function that reads the data and returns it in pandas frame, starting    
    from the last missing value that was found (so that no missing values
    are present). Only wind speed is considered and exogenous attributes
    are discarded. UPDATE (8-6-2018) now it finds missing values and replaces
    them by appending the day before data. Seems to work well at least for
    d08. UPDATE (16-6-2018) works well :)
"""

def get_missing_values(path,name):
    
    leap_years = [2004,2008,2012,2016]
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    temporal_data = pd.read_csv(path+name,usecols=[0,1],encoding='utf-8')
    temporal_data.columns = ['date','speed'] 

    
    # Initial values
    
    hour_minute = temporal_data['date'][0].split(' ')[1].split(':')
    year_month_day = temporal_data['date'][0].split(' ')[0].split('-')
        
    hour = int(hour_minute[0])
    minute = int(hour_minute[1])
    day = int(year_month_day[2])
    month = int(year_month_day[1])
    year = int(year_month_day[0])
    
    # Final values
    
    f_index = len(temporal_data) - 1
    
    hour_minute = temporal_data['date'][f_index].split(' ')[1].split(':')
    year_month_day = temporal_data['date'][f_index].split(' ')[0].split('-')
        
    last_hour = int(hour_minute[0])
    last_minute = int(hour_minute[1])
    last_day = int(year_month_day[2])
    last_month = int(year_month_day[1])
    last_year = int(year_month_day[0])
    
    expected_date = [[minute,hour,day,month,year]]
    
    # A list is built with all the expected dates 
    # to compare the original data side by side with it and detect
    # gaps (missing values)
    
    counter = 0
    
    while hour != last_hour or minute != last_minute \
    or day != last_day or month != last_month \
    or year != last_year:
        
        minute = (minute + 10)%60
        
        if minute == 0:
            
            hour = (hour + 1)%24
            
            if hour == 0:
                
                day = (day + 1)
                
                if month != 2:
                    
                    if day > days_in_month[month-1]:
                    
                        day = 1
                    
                        month = month + 1
                    
                        if month == 13:
                        
                            month = 1
                        
                            year = year + 1
                            
                elif year not in leap_years:
                        
                    if day > days_in_month[month-1]:
                
                        day = 1
                    
                        month = month + 1
                    
                        if month == 13:
                        
                            month = 1
                        
                            year = year + 1
                
                else:
                    
                    if day > days_in_month[month-1] + 1:
                
                        day = 1
                    
                        month = month + 1
                    
                        if month == 13:
                        
                            month = 1
                        
                            year = year + 1
                    
                        
                        
        expected_date.append([minute, hour, day, month, year])
        
        counter += 1
        
#        if counter%10000 == 0:
#            print(counter)
#            print(month)
            
    
    # Null, NaN and missing dates are found
    # Date format is YY-MM-DD HH:mm
    # Side to side comparison between dates in the data and expected
    # dates
    
    missing_positions = []
    e_index = 0
    e_minute, e_hour, e_day, e_month, e_year = [0,0,0,0,0]
    
    for i in range(len(temporal_data)):
        
        speed = temporal_data['speed'][i]
        
        
            
        hour_minute = temporal_data['date'][i].split(' ')[1].split(':')
        year_month_day = temporal_data['date'][i].split(' ')[0].split('-')
        
        hour = int(hour_minute[0])
        minute = int(hour_minute[1])
        day = int(year_month_day[2])
        month = int(year_month_day[1])
        year = int(year_month_day[0])
        
        e_minute, e_hour, e_day, e_month, e_year = expected_date[e_index]
        
        # expected index is shifted by 1 until it matches the current date
        # of the data. Missing positions are stored.
        
        while hour != e_hour or minute != e_minute \
        or day != e_day or month != e_month \
        or year != e_year:
            
#            if e_index == 0:
#                print(minute,hour,day,month,year)
#                print(expected_date[e_index])
            
            missing_positions.append(e_index)
            e_index += 1
            e_minute, e_hour, e_day, e_month, e_year = expected_date[e_index]
        
        if  math.isnan(float(speed)) or speed is None:
            
            missing_positions.append(e_index)
            
        e_index += 1
        
        # None and NaN checking
        
    return missing_positions,expected_date, temporal_data
        
        
""" Function that fills in the gaps in the data. The filling is done by
 replicating the values from the previous day in the gaps """
    
def correct_data(data,missing_positions,expected_date,name):
    
    init_index = 0
    
    block_indexes = []
    
    for i in range(1,len(missing_positions)):
        
        if missing_positions[i]-1 != missing_positions[i - 1]:
            
            block_indexes.append([init_index,i-1])
            init_index = i
            
    block_indexes.append([init_index,i])
    
    for i in range(len(block_indexes)):
        
        print(i)
        
        init = missing_positions[block_indexes[i][0]]
        end = missing_positions[block_indexes[i][1]]
        
        block_data = pd.DataFrame([],columns = ['date','speed'])
        
        # Block of filling values constructed by taking the value
        # 144 positions before (exactly 1 day before). % operation
        # is used to repeat the same last day after a new missing
        # day starts being filled.

        for j in range(end-init+1):
            
            block_data = block_data.append(data.iloc[init+j%144-144]).reset_index(drop=True)
            
            date = expected_date[init+j]
            
            [minute,hour,day,month,year] = date
            
            if minute < 10:
                
                minute = '0'+str(minute)
              
            if hour < 10:
                
                hour = '0'+str(hour)
                
            if day < 10:
                
                day = '0'+str(day)
                
            if month < 10:
                
                month = '0'+str(month)
                
            date = '{}-{}-{} {}:{}'.format(str(year),str(month),str(day),str(hour),str(minute))
            
            block_data.at[j,'date'] = date
            
        data = pd.concat([data[0:init],block_data,data[init:]]).reset_index(drop=True)
        
        # Verify it did what it was supposed to
#        for j in range(init,end+1):
#            print(data.at[init,'speed'] == data.at[init,'speed'])
        
    data.to_csv(path_or_buf = "no_mvs_" + name)
    return data

#missing_positions,expected_date, temporal_data = get_missing_values('','originalb08.csv')
#data = correct_data(temporal_data,missing_positions,expected_date,'originalb08.csv')
 

def get_data(path, name, ts=1, lag=1, overlap=True, train_output=1, test_output=24,get_period=False):
    
    #path = ''
    data = pd.read_csv(path+name, usecols = ['date','speed'])
    
    # It is verified if it starts at minute 00, otherwise index is
    # shifted.
    
    index = 0
    
    while int(data.iloc[index]['date'].split(' ')[1].split(':')[1]) != 0: 
        
        index += 1
        
    shift = 6 - index
    
    data.index = data.index + shift
    
    #dataset del promedio cada 1 hora

    data = data.drop(['date'],axis=1)
    
    data['hour'] = (data.index/6).astype(int)
    data_hour = data.groupby('hour').aggregate(np.mean)

    #Separacion en train y test
    
    N = len(data_hour)
    #periods = np.arange(N)%24
    
    training_data_amount = int(0.7*N)

    train_data = data_hour[:training_data_amount].copy()
    test_data = data_hour[training_data_amount:].copy()
    train_data.index = train_data.index - train_data.index.min()
    test_data.index = test_data.index - test_data.index.min()
    
    #train_periods = periods[:training_data_amount]
    #validation_periods = periods[training_data_amount:]
    
    min_speed = train_data.values.min()
    max_speed = train_data.values.max()
    
    train_data = (train_data - min_speed)/(max_speed - min_speed)
    test_data = (test_data - min_speed)/(max_speed - min_speed)
    
    # Train data separation in input with (?, ts, lags) shape and
    # output with (?,test_input) shape.
           
    shifted_data = [train_data.shift(lag+train_output-1-k) \
                    for k in range(lag+train_output)]

    
    temp_input = pd.concat(shifted_data,axis=1).dropna()
    temp_output = temp_input.iloc[:,-train_output:]
    temp_input = temp_input.iloc[:,:-train_output]
        
    if overlap==True:
        
        shifted = [temp_input.shift(ts-k-1) for k in range(ts)]
        temp_output = temp_output[ts-1:]
        
    else:
        
        shifted = [temp_input.shift((ts-k-1)*lag) for k in range(ts)]
        temp_output = temp_output[(ts-1)*lag:]
    
    temp_input = pd.concat(shifted,axis=1).dropna()
    
    train_input_data = temp_input.values.reshape(-1,ts,lag)
    train_output_data = temp_output.values
    
    # Test data separation in input with (?, ts, lags) shape and
    # output with (?,test_output) shape.
           
    shifted_data = [test_data.shift(lag+test_output-1-k) \
                    for k in range(lag+test_output)]
    
    temp_input = pd.concat(shifted_data,axis=1).dropna()
    temp_output = temp_input.iloc[:,-test_output:]
    temp_input = temp_input.iloc[:,:-test_output]
        
    if overlap == True:
        
        shifted = [temp_input.shift(ts-k-1) for k in range(ts)]
        temp_output = temp_output[ts-1:]
        
    else:
        
        shifted = [temp_input.shift((ts-k-1)*lag) for k in range(ts)]
        temp_output = temp_output[(ts-1)*lag:]
    
    temp_input = pd.concat(shifted,axis=1).dropna()
    
    test_input_data = temp_input.values.reshape(-1,ts,lag)
    test_output_data = temp_output
          
        
    return train_input_data, test_input_data, train_output_data,test_output_data, \
             min_speed, max_speed
#
#training_data_input, testing_data_input, training_data_output, \
#testing_data_output, min_speeds, max_speeds = \
# get_data('', 'no_mvs_b08.csv', ts=2, lag=3, overlap=True, train_output =2)
           



