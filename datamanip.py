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
import helpers as h
import featuremanip as f
from settings import LOCAL_HK_FOLDER, RDS_FOLDER,HP_FOLDER,EXPORT_FOLDER
from sklearn.model_selection import train_test_split as tts

def load_profile(fname, IBSorOBS = 'OBS'):
    with h5.File(fname, 'r') as data:
        # IBS_profile = data['IBS_profile'][:]
        # IBS_time = data['IBS_time'][:]
        profile = data[f'{IBSorOBS}_profile'][:]
        time = data[f'{IBSorOBS}_time'][:]
    return time, profile


###########################################################################################
def in_sklearn_format (features,ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER, test = False):
    if ds_period < 0:
        hp_time_bins = f.get_var_time_bins(ds_duration)
    else:
        hp_time_bins = np.arange(0, ds_duration, ds_period)

    dfs = []

    for i, row in features.iterrows():
        date = row['Date']
        fname = h.get_heater_fname(date, hp_folder)

        if fname != '':
            time, profile = load_profile(fname, IBSorOBS)

            R = profile[:,0]
            T = profile[:,1]
            N = profile[:,2]

            hp_df = pd.DataFrame({
                'R': R,
                'T': T,
                'N': N,
                'Time': time
            })
            
            hp_df = hp_df.loc[(hp_df['Time'] < ds_duration) & (hp_df['Time'] > 0)]
            hp_df['time_bin'] = np.digitize(hp_df['Time'], bins = hp_time_bins) 
            hp_ds = hp_df.groupby('time_bin').mean()
            hp_ds.reset_index(inplace=True)
            
            #! I have a choice to make here
            #! turns out that it was only going up to the time of the real profile
            #! do I keep this, or force every profile to last to ds_duration?
            #! but I can't have a np.nan in the training data
            # going for just times of real profile
            bin_centers = (hp_time_bins[1:] + hp_time_bins[:-1])/2
            
            # because the downsampling goes one too far I think
            N_profile = hp_ds.shape[0]-1
            if bin_centers.shape[0] != hp_ds.shape[0]-1:
                times = bin_centers[:hp_ds.shape[0]-1]
            else:
                times = bin_centers
            
            profile_ds = pd.DataFrame(
                {
                    'R': hp_ds['R'].values[:N_profile],
                    'T': hp_ds['T'].values[:N_profile],
                    'N': hp_ds['N'].values[:N_profile],
                }
            )
            
            profile_ds['hp_id'] = i
            profile_ds['Time'] = times
            
            # add in all the features for that day
            for key in features.keys():
                if key != "Date":
                    profile_ds[key] = row[key]
            
            dfs.append(profile_ds)
            
            
        else:
            if test == True:
                # print(f'No profile on {date}')
                profile_ds = pd.DataFrame({
                    'R': np.nan,
                    'T': np.nan,
                    'N': np.nan,
                    'Time': (hp_time_bins[1:] + hp_time_bins[:-1])/2
                    })
                
                profile_ds['hp_id'] = i
            
                # add in all the features for that day
                for key in features.keys():
                    if key != "Date":
                        profile_ds[key] = row[key]

                dfs.append(profile_ds)
        
    data = pd.concat(dfs).reset_index(drop=True)
    return data


def features2train(features, val_features, ds_duration, ds_period, IBSorOBS,coord = 'rtn'):
    if val_features is not None:
        val = in_sklearn_format(val_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
        all_data = in_sklearn_format(features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)

        train, test = tts(all_data, test_size=0.2, random_state=0)

    B, phi, theta = f.cart2sph(
        train['R'].values,
        train['T'].values,
        train['N'].values,
    )
    train['|B|'] = B
    train['phi'] = phi
    train['theta'] = theta
    
    B, phi, theta = f.cart2sph(
        val['R'].values,
        val['T'].values,
        val['N'].values,
    )
    val['|B|'] = B
    val['phi'] = phi
    val['theta'] = theta
    
    B, phi, theta = f.cart2sph(
        test['R'].values,
        test['T'].values,
        test['N'].values,
    )
    test['|B|'] = B
    test['phi'] = phi
    test['theta'] = theta
    
    # scaling
    no_scaling_names = [
        'hp_id',
        'Heater',
        'SA change',
        'HGA azimuth change',
        'HGA evelvation change',
        'No time A off'
    ]

    mapper_list = []

    for key in train.keys():
        if any(substring in key for substring in no_scaling_names):
            mapper_list.append((key, None))
        else:
            mapper_list.append(([key], RobustScaler()))

    mapper = DataFrameMapper(mapper_list, df_out = True)



    scaler = mapper.fit(train)
    train_scaled = mapper.transform(train)
    test_scaled = mapper.transform(test)
    val_scaled = mapper.transform(val)
    
    x_train = train_scaled.drop(['hp_id', 'R', 'T', 'N', '|B|', 'phi', 'theta'],axis = 1)
    x_val = val_scaled.drop(['hp_id', 'R', 'T', 'N', '|B|', 'phi', 'theta'], axis = 1)
    x_test = test_scaled.drop(['hp_id', 'R', 'T', 'N', '|B|', 'phi', 'theta'], axis = 1)
    y_train = train_scaled[['R', 'T', 'N']].copy()
    y_val = val_scaled[['R', 'T', 'N']].copy()
    y_test = test_scaled[['R', 'T', 'N']].copy()
    return x_train, x_val, x_test, y_train, y_val, y_test, mapper