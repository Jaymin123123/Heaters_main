import numpy as np
import pandas as pd
import helpers as h
import matplotlib.pyplot as plt
import finders_help as fh
from datetime import datetime, timedelta   
import spacepy.pycdf as cdf
import os
import features_function as ff
import astrospice
import re

start = datetime(2023, 4, 30)
end = datetime(2023, 5, 1)
duty_cycle = h.load_HK('solo_ANC_sc-thermal-mag', start, end)
duty_cycle = pd.DataFrame(duty_cycle)
####file location for L2 data
#file_location = '/rds/general/user/js6523/projects/spaceandplanetaryphysics20222025/live/so_mag_heaters/mag_data/solo_L2_mag-rtn-normal-internal/2023'

# file location for L1 data OBS
file_location = '/rds/general/user/js6523/projects/spaceandplanetaryphysics20222025/live/so_mag_heaters/mag_data/solo_L1_mag-ibs-normal/2023'

####file location for L1 data IBS
#file_location = '/rds/general/user/js6523/projects/spaceandplanetaryphysics20222025/live/so_mag_heaters/mag_data/solo_L1_mag-ibs-normal/2023'

threshold = 0.12

current = start

while current <= end:
    cdf_file = fh.get_latest_version_file(file_location, current)
    
    if not cdf_file or not os.path.isfile(cdf_file):
        print(f"No valid file found for date {current.strftime('%Y-%m-%d')}. Skipping to the next date.")
        current += timedelta(days=1)
        continue
    
    duty_cycle['Date'] = duty_cycle['Time'].dt.date
    duty_cycle_csv = duty_cycle[duty_cycle['Date'] == current.date()]
    duty_cycle_csv = duty_cycle_csv.dropna(subset=['IFB_HTR1_LCL5_TM'])
    
    segments = fh.capture_profiles(duty_cycle_csv, threshold)
    segments_df = pd.DataFrame(duty_cycle_csv['Time'][segments])
    segments_df_minus = pd.DataFrame(duty_cycle_csv['Time'][segments] - timedelta(seconds=7))
    
    fh.extract_profiles_5(segments_df, segments_df_minus, cdf_file, 'output_folder_ibs_l1', current)
    ff.features_to_csv_5(current, segments_df_minus, 'output_csv_ibs.csv')
    
    current += timedelta(days=1)

