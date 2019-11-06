# -*- coding: utf-8 -*-
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
def get_difference(array):
    difference = [0]
    for i in range(len(array)-1):
        difference.append(array[i+1] - array[i])
    return difference
def get_lag1(array):
    lag1 = [0]
    for i in range(len(array)-1):
        lag1.append(array[i])
    return lag1
def get_next_difference(array):
    difference = []
    for i in range(len(array)-1):
        difference.append(array[i+1] - array[i])
    difference.append(0)
    return difference
def transform(data):
    difference = get_difference(data)##current price - the price one day before
    next_difference = get_next_difference(data)## next day's price - current price
    lag1 = get_lag1(difference)##the change one day before
    lag2 = get_lag1(lag1)##change in two day before
    lag3 = get_lag1(lag2)##change in three day
    lag4 = get_lag1(lag3)##change in 4 day
    first_moment = get_difference(difference)## the derivative of todays change
    second_moment = get_difference(first_moment)##second derivative of todays change
    data = np.array([difference,lag1,lag2,lag3,lag4,first_moment,second_moment,next_difference]).T
    return data
def discretize(data):   
    discretizer = KBinsDiscretizer(n_bins= 10, encode='ordinal', strategy='uniform')
    discretizer.fit(data)##since they are all difference one discretizer is enough for lag and differnence
    discretized_data = discretizer.transform(data)
    discretized_df = pd.DataFrame(data = discretized_data,columns = ['difference','lag1','lag2','lag3','lag4','first_moment','second_moment','next_differnence'])
    discretized_df.to_csv('trainDataCURADiscretized.csv')
    return  discretized_data, discretizer
    
