from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, RobustScaler

import glob
import h5py as h5

def reduce_cardinality(features, feature_name, N, bins = None):
    if bins is None:
        bins = np.linspace(features[feature_name].min(), features[feature_name].max(), N)
    features['bin'] = np.digitize(features[feature_name], bins = bins) 
    # so just radius_bin as the feature, the model doesn't need to know its true value
    features.drop([feature_name], axis = 1, inplace = True)
    features.rename(columns = {'bin': feature_name}, inplace= True)
    return features

def mode(arr, bins = 10):
    arr = arr[~np.isnan(arr)] #~ means not

    if len(arr) > 0:
        hist, bin_edges = np.histogram(arr, bins=bins)
        centers = 0.5*(bin_edges[1:]+ bin_edges[:-1])
        max_idx = np.argmax(hist)
        mode = centers[max_idx]
        return mode
    else:
        #print('Just nans')
        return np.nan

def get_weights(time, ds_duration):
    # return np.cos((np.pi/2) * time/ds_duration)
    # return np.where(time < 70, 1, 0.1)
    A = 0.5
    B = 100
    C = 10
    return (1-A) + A/(1+ np.exp((time-B)/C))

def cart2sph(r,t,n):
    # if all (0,0,0) then return 0
    
    
    mag = np.sqrt(r*r + t*t + n*n)
    theta = np.arcsin(n / mag) * 180/np.pi
    
    phi = np.arctan2(t,r) * 180/np.pi
    
    phi = np.where(r==0, 0, phi)
    theta = np.where(r==0, 0, theta)
    
    return mag, phi, theta

def add_history(features, col_name, n_days_back, bfill = True):
    for days_back in np.arange(1, n_days_back+1):
        new_col_name = col_name + f' -{days_back}'
        features[new_col_name] = features[col_name].shift(days_back)
        if bfill:
            features[new_col_name].bfill(inplace=True)
    
    # do 1 in the future too
    new_col_name = col_name + f' 1'
    features[new_col_name] = features[col_name].shift(-1)
    if bfill:
        features[new_col_name].ffill(inplace=True)
    return features

def get_var_time_bins(ds_duration):
    var_hp_time_bins = np.concatenate(
            (
                np.arange(-1, 20, 0.5),
                np.arange(20, 50, 1),
                np.arange(50, 90, 0.5),
                np.arange(90, ds_duration, 5),
            )
        )
    return var_hp_time_bins