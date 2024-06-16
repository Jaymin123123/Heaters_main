import pandas as pd
from datetime import datetime, timedelta
import astrospice
import astropy.units as u
from sunpy.coordinates import HeliographicCarrington
import argparse
import helpers as h
import numpy as np

# Define the frame and kernel
carr_frame = HeliographicCarrington(observer='self')
solo_kernels = astrospice.registry.get_kernels('solar orbiter', 'predict')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--start', '-s', default=datetime(2023, 6, 26), type=lambda d: datetime.strptime(d, '%Y-%m-%d'), help='Specify the start in the format YYYY-MM-DD')
    parser.add_argument('--end', '-e', default=datetime(2024, 1, 26), type=lambda d: datetime.strptime(d, '%Y-%m-%d'), help='Specify the end in the format YYYY-MM-DD')
    parser.add_argument("--history", type=int, default=3, help="How far back to include as features")

    args = parser.parse_args()
    start = args.start
    end = args.end

    all_times = h.get_times()
    all_dates = h.get_dates(start, end)

    # Adds the distance from the Sun as a feature    
    features = pd.DataFrame({
        "Times": all_times,
    })

    #########################thermal mag########################
    print("TH MAG")
    print('')
    th_mag = h.load_HK('solo_ANC_sc-thermal-mag', start, end)
    th_mag['Time'] = pd.to_datetime(th_mag['Time'])  # Ensure 'Time' column is in datetime format
    th_mag.set_index('Time', inplace=True)  # Set 'Time' column as the index for resampling

    # Select only numeric columns for resampling
    numeric_cols = th_mag.select_dtypes(include=[np.number]).columns

    for time_interval in all_times:
        interval_start = time_interval
        interval_end = time_interval + pd.Timedelta(minutes=10)
        # Filter the data within the current interval
        th_mag_interval = th_mag.loc[interval_start:interval_end]

        if not th_mag_interval.empty:
            # Resample the data for the current interval
            th_mag_resampled = th_mag_interval[numeric_cols].resample("1T").mean()
            th_mag_resampled_curr = th_mag_interval[numeric_cols].resample("10T").max()

            # Fill NaN values with the closest non-NaN value
            th_mag_resampled_filled = th_mag_resampled.ffill().bfill()
            th_mag_resampled_curr_filled = th_mag_resampled_curr.ffill().bfill()

            # Compute additional features
            additional_features = {
                'Time': time_interval,
                'Mean_OBS_T': th_mag_resampled_filled["ANP_2_1_6 MAG OBS"].mean(),
                'Mean_IBS_T': th_mag_resampled_filled["ANP_2_1_9 MAG IBS"].mean(),
                'Mean_OBS_IBS_Current': th_mag_resampled_curr_filled["IFB_HTR1_LCL5_TM"].mean(),
                'Max_OBS_IBS_Current': th_mag_resampled_curr_filled["IFB_HTR1_LCL5_TM"].max(),
                'Min_OBS_IBS_Current': th_mag_resampled_curr_filled["IFB_HTR1_LCL5_TM"].min()
            }
            
            # Append the additional features to the features dataframe
            for key, value in additional_features.items():
                if key not in features.columns:
                    features[key] = [None] * len(features)
                features.loc[features['Times'] == time_interval, key] = value

    ######################`thermal swa rpw########################

    print('TH SWA RPW')
    print('')
    
    th_swa_rpw = h.load_HK('solo_ANC_sc-thermal-swa-rpw', start, end)
    th_swa_rpw['Time'] = pd.to_datetime(th_swa_rpw['Time'])  # Ensure 'Time' column is in datetime format
    th_swa_rpw.set_index('Time', inplace=True)  # Set 'Time' column as the index for resampling

    numeric_cols = th_swa_rpw.select_dtypes(include=[np.number]).columns

    for time_interval in all_times:
        interval_start = time_interval
        interval_end = time_interval + pd.Timedelta(minutes=10)
        # Filter the data within the current interval
        th_swa_rpw_interval = th_swa_rpw.loc[interval_start:interval_end]

        if not th_swa_rpw_interval.empty:
            # Resample the data for the current interval
            th_swa_rpw_resampled = th_swa_rpw_interval[numeric_cols].resample("1T").mean()
            th_swa_rpw_resampled_ = th_swa_rpw_interval[numeric_cols].resample("10T").max()

            # Fill NaN values with the closest non-NaN value
            th_swa_rpw_resampled_filled = th_swa_rpw_resampled.ffill().bfill()
            th_swa_rpw_resampled__filled = th_swa_rpw_resampled_.ffill().bfill()

            # Compute additional features
            additional_features = {
                'SCM_T': th_swa_rpw_resampled_filled["ANP_1_1_7 RPW SCM SRP"].mean(),
                'EAS_T': th_swa_rpw_resampled__filled["ANP_2_1_14 SWA EAS"].mean()
            }
            
            # Append the additional features to the features dataframe
            for key, value in additional_features.items():
                if key not in features.columns:
                    features[key] = [None] * len(features)
                features.loc[features['Times'] == time_interval, key] = value

########## Radius ############################
    # Interpolate radii to match the length of all_times
    coords = astrospice.generate_coords('SOLAR ORBITER', all_dates)
    solo_coords = coords.transform_to(carr_frame)
    radii = solo_coords.radius.to(u.au).value
    
    for time_interval in all_times:
        interval_start = time_interval    
        features["Distance"] = radii_series.values

        if not th_swa_rpw_interval.empty:
            additional_features = {
                'SCM_T': th_swa_rpw_resampled_filled["ANP_1_1_7 RPW SCM SRP"].mean(),
                'EAS_T': th_swa_rpw_resampled__filled["ANP_2_1_14 SWA EAS"].mean()
            }
            
            # Append the additional features to the features dataframe
            for key, value in additional_features.items():
                if key not in features.columns:
                    features[key] = [None] * len(features)
                features.loc[features['Times'] == time_interval, key] = value
    

    features.to_csv("features_int.csv", date_format='%Y-%m-%d', index=False)
