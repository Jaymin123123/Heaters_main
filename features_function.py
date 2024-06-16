import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import helpers as h

def features_to_csv_5(current_date, segments_df_minus, output_csv):
    # Convert 'Time' column to datetime
    segments_df_minus['Time'] = pd.to_datetime(segments_df_minus['Time'])
    all_times = segments_df_minus['Time'].tolist()
    all_dates = h.get_dates(current_date, current_date)

    if not os.path.exists(output_csv):
        # Create a CSV with headers if it does not exist
        pd.DataFrame(columns=["Times"]).to_csv(output_csv, index=False)

    features = pd.DataFrame({
        "Times": all_times,
    })

    ######################### thermal Currents ########################
    print("TH Currents")
    print('')
    th_mag = h.load_HK('solo_ANC_sc-thermal-mag', current_date, current_date)
    th_mag['Time'] = pd.to_datetime(th_mag['Time'])  
    th_mag.set_index('Time', inplace=True)  

    # Select only numeric columns for resampling
    numeric_cols = th_mag.select_dtypes(include=[np.number]).columns
    count = 0
    for time_interval in all_times:
        interval_start = time_interval
        interval_end = time_interval + pd.Timedelta(minutes=5)
        # Filter the data within the current interval
        th_mag_interval = th_mag.loc[interval_start:interval_end]

        if not th_mag_interval.empty:
            # Resample the data for the current interval
            th_mag_resampled_curr = th_mag_interval[numeric_cols].resample("5T").max()

            # Fill NaN values with the closest non-NaN value
            th_mag_resampled_curr_filled = th_mag_resampled_curr.ffill().bfill()

            # Compute additional features
            additional_features = {
                'Date': [current_date.strftime('%Y-%m-%d')],
                'Index': [count],
                'Time': [time_interval],
                'Mean_OBS_IBS_Current': [th_mag_resampled_curr_filled["IFB_HTR1_LCL5_TM"].mean()],
                'Max_OBS_IBS_Current': [th_mag_resampled_curr_filled["IFB_HTR1_LCL5_TM"].max()],
                'Min_OBS_IBS_Current': [th_mag_resampled_curr_filled["IFB_HTR1_LCL5_TM"].min()]
            }

            # Append the additional features to the features dataframe
            for key, value in additional_features.items():
                if key not in features.columns:
                    features[key] = [None] * len(features)
                features.loc[features['Times'] == time_interval, key] = value
        count += 1
    ###################### thermal swa rpw ########################

    print('TH SWA RPW')
    print('')

    th_swa_rpw = h.load_HK('solo_ANC_sc-thermal-swa-rpw', current_date, current_date)
    th_swa_rpw['Time'] = pd.to_datetime(th_swa_rpw['Time'])  # Ensure 'Time' column is in datetime format
    th_swa_rpw.set_index('Time', inplace=True)  # Set 'Time' column as the index for resampling

    numeric_cols = th_swa_rpw.select_dtypes(include=[np.number]).columns

    for time_interval in all_times:
        interval_start = time_interval
        interval_end = time_interval + pd.Timedelta(minutes=5)
        # Filter the data within the current interval
        th_swa_rpw_interval = th_swa_rpw.loc[interval_start:interval_end]

        if not th_swa_rpw_interval.empty:
            # Resample the data for the current interval
            th_swa_rpw_resampled = th_swa_rpw_interval[numeric_cols].resample("1T").mean()
            th_swa_rpw_resampled_ = th_swa_rpw_interval[numeric_cols].resample("5T").max()

            # Fill NaN values with the closest non-NaN value
            th_swa_rpw_resampled_filled = th_swa_rpw_resampled.ffill().bfill()
            th_swa_rpw_resampled__filled = th_swa_rpw_resampled_.ffill().bfill()

            # Compute additional features
            additional_features = {
                'SCM_T': [th_swa_rpw_resampled_filled["ANP_1_1_7 RPW SCM SRP"].mean()],
                'EAS_T': [th_swa_rpw_resampled__filled["ANP_2_1_14 SWA EAS"].mean()]
            }

            # Append the additional features to the features dataframe
            for key, value in additional_features.items():
                if key not in features.columns:
                    features[key] = [None] * len(features)
                features.loc[features['Times'] == time_interval, key] = value

    ########## Thermal Temperature ############################
    print('TH Temperature')
    print('')
    th_temp = pd.read_csv('heaters_data_interpolated_xgb1.csv')
    th_temp['Time'] = pd.to_datetime(th_temp['Time'])  # Ensure 'Time' column is in datetime format
    th_temp = th_temp[th_temp['Time'].dt.date == current_date.date()]
    th_temp.set_index('Time', inplace=True)  # Set 'Time' column as the index for resampling

    numeric_cols = th_temp.select_dtypes(include=[np.number]).columns
    count = 0
    for time_interval in all_times:
        interval_start = time_interval
        interval_end = time_interval + pd.Timedelta(minutes=5)
        # Filter the data within the current interval
        th_temp_interval = th_temp.loc[interval_start:interval_end]

        if not th_temp_interval.empty:
            # Resample the data for the current interval
            th_temp_resampled_temp = th_temp_interval

            # Compute additional features
            additional_features = {
                'Mean_OBS_Temp': [th_temp_resampled_temp["ANP_2_1_6 MAG OBS"].mean()],
                'Max_OBS_Temp': [th_temp_resampled_temp["ANP_2_1_6 MAG OBS"].max()],
                'Min_OBS_Temp': [th_temp_resampled_temp["ANP_2_1_6 MAG OBS"].min()],
                'Mean_IBS_Temp': [th_temp_resampled_temp["ANP_2_1_9 MAG IBS"].mean()],
                'Max_IBS_Temp': [th_temp_resampled_temp["ANP_2_1_9 MAG IBS"].max()],
                'Min_IBS_Temp': [th_temp_resampled_temp["ANP_2_1_9 MAG IBS"].min()]
            }
            print(additional_features)
            print(th_temp_interval)
            # Append the additional features to the features dataframe
            for key, value in additional_features.items():
                if key not in features.columns:
                    features[key] = [None] * len(features)
                features.loc[features['Times'] == time_interval, key] = value
        count += 1

    features.to_csv(output_csv, mode='a', header=True, index=False)