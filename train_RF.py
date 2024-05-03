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
    features_ = pd.read_csv(h.LOCAL_HK_FOLDER + "features.csv", parse_dates = ['Date'])
    

    # get rid of the bad times
    ranges = pd.read_csv("bad_dates.csv", parse_dates=['start', 'end'])
    #! need to also deal with duplicate days
    
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
    IBSorOBS = args.inst

    HP_TIME_BINS = np.arange(0, ds_duration, ds_period)

    if args.coord == 'rtn':
        components = ['R', 'T', 'N']
    elif args.coord == 'sph':
        components = ['|B|', 'phi', 'theta']
        

    # Define outer CV for model selection and evaluation
    # maybe I don't want to shuffle because then there wont be a day nearby to learn from
    # would learn the relationships better?
    
    if args.cv:
        # take random parts of the training to validate the outer CV
        outer_cv = KFold(n_splits=10, shuffle=False)
    
    # the inner is used to train the BayesSearchCV method
    # therefore, I don't want to shuffle the data as it will just learn which days are good
    inner_cv = KFold(n_splits=3, shuffle=False)
    
    # looking at the last fitted model, there were 250-300+ estimators (boosting rounds)
    stopping_rounds = 20 #50
    early_stopping_callback = lgb.early_stopping(stopping_rounds=stopping_rounds)
    logging_callback = lgb.log_evaluation(period=stopping_rounds)
    
    # SPACE = {
    #     'learning_rate': skopt.space.Real(0.01, 0.3, prior='log-uniform'),
    #     'max_depth': skopt.space.Integer(2, 5),
    #     'n_estimators': skopt.space.Integer(stopping_rounds, 1000, prior='log-uniform'),
    #     'reg_lambda': skopt.space.Real(0, 100),
    #     'reg_alpha': skopt.space.Real(0, 100),
    #     'num_leaves': skopt.space.Integer(2, 1000),
    #     'min_data_in_leaf': skopt.space.Integer(10, 1000),
    #     'feature_fraction': skopt.space.Real(0.1, 1.0, prior='uniform'),
    #     'subsample': skopt.space.Real(0.1, 1.0, prior='uniform'),
    # }

    
    
    
    # based on this article https://medium.com/@gabrieltseng/gradient-boosting-and-xgboost-c306c1bcfaf5
    # seems like early stopping is good way to do it
    # so only need to worry about these:
    # n_estimators = (keep it low, but early stopping should help here) (try high once)
    # learning rate = small
    # max_depth = 5 at max
    # reg_alpha and reg_lambda

    # since these jobs are independent, I can run on different cores!
    
    # could use priors from previous fit?
    
    #! for each training, I need:
    # train
    # validation: to use for early stopping
    # test (to do the scoring) 
    
    #! so take a few measurements out of training and use them for validation
    #! Again, I want the validation to act as unseen data (-ish) so that it can't
    #! just figure out we are near this day. DONT USE RANDOM SHUFFLE
    
    if args.cv:
        print("This cole is very old and probably doesn't work. It uses outer validation to prove that the model can adapt and not overfit. Turns out you shouldn't use cross validation to train the model, so I focussed on something else. https://stackoverflow.com/questions/46456381/cross-validation-in-lightgbm/50316411#50316411")
        raise NotImplementedError

    else:
        start = 0
        period = 110
        finish = start + period
        ranges = pd.read_csv("/rds/general/user/js6523/home/JayminHeatersBest/Full_550/bad_dates.csv", parse_dates=['start', 'end'])
        bad_dates = h.get_forbidden_dates(ranges)

        while finish < 550:    

            features_ = pd.read_csv(h.LOCAL_HK_FOLDER + "features.csv", parse_dates = ['Date'])
            # get rid of the bad times

            wb_features = features_[~features_['Date'].isin(bad_dates)]
            bad_features = features_[features_['Date'].isin(bad_dates)]
            ################################################################
            # Validation set ###############################################
            ################################################################
            if args.val == 'latest':
                # either take some profiles near the end
                val_date = datetime(2022,12,31)
                val_features = wb_features.loc[(wb_features['Date'] > val_date)]
                features = wb_features.loc[wb_features['Date'] < val_date]
            else:
                features = wb_features.loc[wb_features['Date'] < datetime(2023,2,23)]
                val_features = None

            def load_profile(fname, IBSorOBS='OBS'):
                with h5.File(fname, 'r') as data:
                    filtered_profile = data[f'{IBSorOBS}_profile'][:]
                    filtered_time = data[f'{IBSorOBS}_time'][:]
                valid_indicies = (filtered_time >= start) & (filtered_time <= finish)
                # Filter data where time value is less than 10 seconds
                time = filtered_time[valid_indicies]
                profile = filtered_profile[valid_indicies]
                return time, profile, start, finish  # Ensure profile array matches the filtered time array


            def in_sklearn_format(features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER, test = False):
                # Turn the heater profiles into a DataFrame that SKLearn can understand
                # The heater profile is first downsampled to the same time stamps, e.g. every second for 15 minutes
                # That means that we can use "Time" as a feature, since all the profiles will share the same set of times
                # Each day will have a value for house keeping data, e.g. Temperature of instrument
                # this is just repeated for each value of Time.
                # i.e. For the 5th second with the house keeping of such a value, what is the prediction?
                # if test = True then I will still build the time data and features in, but obviously I don't have a real profile so just leave that blank
                if ds_period < 0:
                    hp_time_bins = h.get_var_time_bins(ds_duration)
                else:
                    hp_time_bins = np.arange(0, ds_duration, ds_period)
                dfs = []
                for i, row in features.iterrows():
                    date = row['Date']
                    fname = h.get_heater_fname(date, hp_folder)
                    if fname != '':
                        time, profile, start, finish = load_profile(fname, IBSorOBS)
                        valid_indices = (time >= start) & (time <= finish)
                        time = time[valid_indices]
                        profile = profile[valid_indices]
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

            def features2train(features, val_features, ds_duration, ds_period, IBSorOBS, split = 'profiles', weights = False, coord = 'rtn'):
                if split == 'days':
                    if val_features is None:
                        val_features = features.sample(n=features.shape[0]//5, random_state = 0)
                        # Create a new DataFrame with the remaining rows
                        features = features[~features.index.isin(val_features.index)].copy(deep=True)
                    # take some random points out for testing
                    test_features = features.sample(n=features.shape[0]//5, random_state = 0)
                    # Create a new DataFrame with the remaining rows
                    train_features = features[~features.index.isin(test_features.index)].copy(deep=True)
                    train = in_sklearn_format(train_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
                    val = in_sklearn_format(val_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
                    test = in_sklearn_format(test_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
                elif split == 'profiles':
                    if val_features is not None:
                        val = in_sklearn_format(val_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
                    all_data = in_sklearn_format(features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
                    if val_features is None:
                        val = all_data.sample(n=all_data.shape[0]//5, random_state = 0)
                        all_data = all_data[~all_data.index.isin(val.index)].copy(deep=True)
                    train = all_data.sample(n=all_data.shape[0]//5, random_state = 0)
                    test = all_data[~all_data.index.isin(train.index)].copy(deep=True)
                else:
                    # I used weight for a bit, but took out. 
                    # Ideally, want the weight to be equal to "how reliable is this heater profile I am learning from?"
                    raise NotImplementedError
                # get the weights before scaling
                if weights:
                    print('make weights represent the amount of variability in B on that day (i.e. how much can we trust the averaging)')
                    raise NotImplementedError
                    # train_weights = get_weights(train['Time'].values, ds_duration)
                    # test_weights = get_weights(test['Time'].values, ds_duration)
                    # val_weights = get_weights(val['Time'].values, ds_duration)
                B, phi, theta = h.cart2sph(
                    train['R'].values,
                    train['T'].values,
                    train['N'].values,
                )
                train['|B|'] = B
                train['phi'] = phi
                train['theta'] = theta
                B, phi, theta = h.cart2sph(
                    val['R'].values,
                    val['T'].values,
                    val['N'].values,
                )
                val['|B|'] = B
                val['phi'] = phi
                val['theta'] = theta
                B, phi, theta = h.cart2sph(
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
                if coord == 'sph':
                    y_train = train_scaled[['|B|', 'phi', 'theta']].copy()
                    y_val = val_scaled[['|B|', 'phi', 'theta']].copy()
                    y_test = test_scaled[['|B|', 'phi', 'theta']].copy()
                else:
                    y_train = train_scaled[['R', 'T', 'N']].copy()
                    y_val = val_scaled[['R', 'T', 'N']].copy()
                    y_test = test_scaled[['R', 'T', 'N']].copy()
                if weights:
                    return x_train, x_val, x_test, y_train, y_val, y_test, mapper, train_weights, test_weights, val_weights
                else:
                    return x_train, x_val, x_test, y_train, y_val, y_test, mapper

            x_train, x_val, x_test, y_train, y_val, y_test, mapper = features2train(
                features,
                val_features, 
                ds_duration,
                ds_period,
                IBSorOBS,
                'profiles',
                coord = 'rtn'
            )
            if args.tune == 'untuned':
                search = RandomForestRegressor()

            search.fit(x_train, y_train)

            feature_importances = search.feature_importances_
            # calculate the final scores
            if args.tune == 'bayes':
                score = search.best_estimator_.score(x_test, y_test)
                print("Test Score: ", score)
                score = search.best_estimator_.score(x_val, y_val)
                print("Val Score: ", score)
            else:
                score = search.score(x_test, y_test)
                print("Test Score: ", score)
                score = search.score(x_val, y_val)
                print("Val Score: ", score)
            feature_importances = search.feature_importances_
            importance_percentages = 100 * (feature_importances / feature_importances.sum())
            feature_names = features.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance (%)': importance_percentages}).sort_values(by='Importance (%)', ascending=False)
            importance_df.to_csv(f'Models/RF_importance/{IBSorOBS}_untuned_{ds_period}_{ds_duration}_importances_{start}.csv')
            # save the mapper
            dump(mapper, f"Models/RF_untuned_full/{IBSorOBS}_{args.tune}_{ds_period}_{ds_duration}_scaler.joblib")
            # save the model
            dump(search, f"Models/RF_untuned_full/{IBSorOBS}_{args.tune}_{ds_period}_{ds_duration}.joblib")