# imports
import pandas as pd
from datetime import datetime
import numpy as np
import helpers as h

import argparse
import logging

from sklearn.model_selection import train_test_split, KFold
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, RobustScaler
import skopt
from skopt.callbacks import EarlyStopper
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
import h5py as h5


import lightgbm as lgb
from joblib import dump
import os

from settings import HP_FOLDER

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inst",
        "-i",
        default = 'OBS',
        choices = ['OBS', 'IBS'],
        type=str,
        help="Which instrument to use, IBS or OBS")
    
    parser.add_argument(
        "--period",
        "-p",
        default=-1,
        type=int,
        help="Period to downsample the profiles",
    )
    
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=15*60,
        help="Duration to downsample the profiles",
    )
    
    parser.add_argument(
        "--tune",
        '-t',
        type=str,
        default='untuned',
        choices = ['untuned', 'bayes', 'tuned'],
        help="How to tune the model",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default='days',
        choices = ['days', 'profiles'],
        help="How to split training and test data. By selecting whole days, or just parts of all profiles",
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default='100',
        help="How big should the step be when splitting the profiles",
    )
    
    # Define an argument for the boolean variable
    parser.add_argument('--cv', action='store_true', help='To perform outer CV or not')
    
    # Define an argument for the boolean variable
    parser.add_argument('--weights', '-w', action='store_true', help='To use weights or not')
    
    parser.add_argument(
        "--val",
        "-v",
        type=str,
        default='latest',
        choices = ['latest', 'random'],
        help="How to get the validation set",
    )
    
    # whether to use 
    parser.add_argument('--rfe', action='store_true', help='To use Recursive Feature Elimination (RFE) or not')
    
    parser.add_argument(
        "--coord",
        type=str,
        default='rtn',
        choices = ['rtn', 'sph'],
        help="Coordinate system",
    )

    args = parser.parse_args()
    print('Training with these settings:')
    print(args)
    

    # load data and features
    features_ = pd.read_csv("features.csv", parse_dates = ['Date'])
    

    # get rid of the bad times
    ranges = pd.read_csv("bad_dates_1.csv", parse_dates=['start', 'end'])
    #! need to also deal with duplicate days
    bad_dates= h.get_forbidden_dates(ranges)
    without_bad_dates = features_[~features_['Date'].isin(bad_dates)]
    bad_features = features_[features_['Date'].isin(bad_dates)]


    
    ################################################################
    # Validation set ###############################################
    ################################################################
    

    

    # take some random days for validation
    # from test_cv I have changed features_after_val to features
    # val_features = wb_features.sample(n=30, random_state = 0)
    # features = wb_features[~wb_features.index.isin(val_features.index)]
    
    
    
    #! could make these input variables to the script
    ds_period = args.period
    ds_duration = args.duration
    IBSorOBS = 'IBS'

    HP_TIME_BINS = np.arange(0, ds_duration, ds_period)

    if args.coord == 'rtn':
        components = ['R', 'T', 'N']
    elif args.coord == 'sph':
        components = ['|B|', 'phi', 'theta']
    
    def load_profile(fname,start, IBSorOBS='OBS'):
        with h5.File(fname, 'r') as data:
            filtered_profile = data[f'{IBSorOBS}_profile'][:]
            filtered_time = data[f'{IBSorOBS}_time'][:]
        valid_indicies = (filtered_time >= start) & (filtered_time <= start+period)
        # Filter data where time value is less than 10 seconds
        time = filtered_time[valid_indicies]
        profile = filtered_profile[valid_indicies]
        return time, profile

    starting = 0
    steps =  args.steps
    period = np.round(args.duration//args.steps)
    val_date = datetime(2024,7,31)
    val_features = without_bad_dates[(without_bad_dates['Date'] > val_date)]
    features = without_bad_dates.loc[without_bad_dates['Date'] < val_date]
    print(bad_dates)
    while starting < args.duration -7 :
        end = starting + period
        if ds_period < 0:
            hp_time_bins = h.get_var_time_bins(ds_duration)
        else:
            hp_time_bins = np.arange(starting, starting+period, ds_period)
        dfs = []
        for i, row in features.iterrows():
            date = row['Date']
            fname = h.get_heater_fname(date, folder = HP_FOLDER)
            if fname != '':
                time, profile = load_profile(fname, starting, IBSorOBS)
                R = profile[:,0]
                T = profile[:,1]
                N = profile[:,2]

                hp_df = pd.DataFrame({
                    'R': R,
                    'T': T,
                    'N': N,
                    'Time': time
                })
                
                hp_df = hp_df.loc[(hp_df['Time'] < starting+period) & (hp_df['Time'] > starting)]
                hp_df['time_bin'] = np.digitize(hp_df['Time'], bins = hp_time_bins) 
                hp_ds = hp_df.groupby('time_bin').mean()
                hp_ds.reset_index(inplace=True)               
                
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
                        profile_ds[key] = row[key]
                
                profile_ds.reset_index(drop=True, inplace=True)
                dfs.append(profile_ds)

        data = pd.concat(dfs).reset_index(drop=True)
        val_data = data[data['Date'] > val_date]
        all_data = data[data['Date'] < val_date]
            # After concatenating dfs into data
        if 'R' in all_data.columns and 'T' in all_data.columns and 'N' in all_data.columns:
            # Assuming 'R', 'T', 'N' are your target features and others are input features
            X = all_data.drop(['R', 'T', 'N', 'hp_id', 'Time', 'DistanceToVenus', 'Date', 'DistanceToEarth'], axis=1)
            y = all_data['T']

            # Check if there's enough data to perform a split
            if len(X) > 10:  # Arbitrary small number to ensure there's enough data
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            else:
                print("Not enough data to perform a train-test split.")
        else:
            print("Data does not contain required target columns ('R', 'T', 'N'). Check data preparation steps.")

        no_scaling_names = [
            'hp_id',
            'Heater',
            'SA change',
            'HGA azimuth change',
            'HGA evelvation change',
            'No time A off'
        ]
        #NIS = x_train
        #rfe = RFE(estimator = RandomForestRegressor(), n_features_to_select=20, step = 1, verbose=1)
        #fittings1 = rfe.fit(x_train, y_train)
        #
        #for i in range(x_train.shape[1]):
        #    print(f'Column: {x_train.keys()[i]}, Selected {rfe.support_[i]}, Rank: {rfe.ranking_[i]:.3f}')        
        #
        #x_train = rfe.transform(x_train)
        #x_test = rfe.transform(x_test)
        #
        ## save the columns to keep
        #selected_features = rfe.support_
        #
        #dump(selected_features, f'Models/nans/{IBSorOBS}_{args.tune}_{ds_period}_{ds_duration}_features_{starting}.joblib')
#
        mapper_list = []
        for key in x_test.keys():
            if any(substring in key for substring in no_scaling_names):
                mapper_list.append((key, None))
            else:
                mapper_list.append(([key], RobustScaler()))
#
        mapper = DataFrameMapper(mapper_list , df_out = True)
        x_train = mapper.fit_transform(x_train)
        x_test = mapper.fit_transform(x_test)
        search = RandomForestRegressor()
        search.fit(x_train, y_train)
#
        feature_importances = search.feature_importances_
        ## calculate the final scor
        score = search.score(x_test, y_test)
        print("Test Score: ", score)
        #score = search.score(x_val, y_val)
        #print("Val Score: ", score)
#
        feature_importances = search.feature_importances_
        importance_percentages = 100 * (feature_importances / feature_importances.sum())
        feature_names = X.columns
        print(len(feature_names), len(importance_percentages))
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance (%)': importance_percentages}).sort_values(by='Importance (%)', ascending=False)
        importance_df.to_csv(f'Models/RF_IMP_T_IBS/{IBSorOBS}_untuned_{ds_period}_{ds_duration}_importances_{starting}_T_IBS.csv')
        ## save the model
        
        starting += period

        del search
        del x_train
        del x_test
        del y_train
        del y_test






    