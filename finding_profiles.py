import numpy as np
import pandas as pd
import helpers as h
import matplotlib.pyplot as plt
import finders_help as fh
from datetime import datetime, timedelta   
import spacepy.pycdf as cdf
import os

start = datetime(2023,1,1)
end = datetime(2023,12,31)
duty_cycle = h.load_HK('solo_ANC_sc-thermal-mag', start, end)
duty_cycle = pd.DataFrame(duty_cycle)
file_location = '/rds/general/user/js6523/projects/spaceandplanetaryphysics20222025/live/so_mag_heaters/mag_data/solo_L2_mag-rtn-normal-internal/2023'
threshold = 0.12

def capture_profiles(duty_cycle_csv, threshold):
    segments = []
    in_segment = False
    for index, row in duty_cycle_csv.iterrows():
        if row['IFB_HTR1_LCL5_TM'] > threshold:  # Using iloc to access by index position
            if not in_segment:
                segments.append(index)
                in_segment = True
        else:
            in_segment = False
    return segments

current = start
while current <= end:
    cdf_file = f'{file_location}/solo_L2_mag-rtn-normal-internal_{current.strftime("%Y%m%d")}_V00.cdf'

    if not os.path.isfile(cdf_file):
        print(f"File {cdf_file} does not exist. Skipping to the next date.")
        current += timedelta(days=1)
        continue
    duty_cycle['Date'] = duty_cycle['Time'].dt.date
    duty_cycle_csv = duty_cycle[duty_cycle['Date'] == current.date()]
    duty_cycle_csv = duty_cycle_csv.dropna(subset=['IFB_HTR1_LCL5_TM'])

    segments = capture_profiles(duty_cycle_csv, threshold)  # Passing the column directly
    segments_df = pd.DataFrame(duty_cycle_csv['Time'][segments])
    segments_df_minus = pd.DataFrame(duty_cycle_csv['Time'][segments]- timedelta(seconds=7))
    print(duty_cycle)
    print(segments_df)
    print(segments_df_minus)
    print(duty_cycle_csv)
    # Extract the profiles
    fh.extract_profiles_l2_10(segments_df,segments_df_minus, cdf_file, 'output_folder', current)

    current += timedelta(days=1)



