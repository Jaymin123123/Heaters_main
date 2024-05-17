import numpy as np
import pandas as pd
import helpers as h
import matplotlib.pyplot as plt
import finders_help as fh
from datetime import datetime, timedelta   

start = datetime(2023,6,26)
end = datetime(2023,6,26)
duty_cycle = h.load_HK('solo_ANC_sc-thermal-mag', start, end)

cdf_file = '26006/solo_L1_mag-obs-normal_20230626_V01.cdf'
cdf_file_data = fh.load_l1_data(cdf_file)

duty_cycle_csv = pd.read_csv('duty_cycle.csv')


threshold = 0.12

# Find the profiles
def capture_profiles(duty_cycle_csv, threshold):
    segments = []
    in_segment = False
    for i in range(len(duty_cycle_csv)):
        if duty_cycle_csv.iloc[i] > threshold:  # Using iloc to access by index position
            if not in_segment:
                segments.append(duty_cycle_csv.index[i])
                in_segment = True
        else:
            in_segment = False
    return segments




#duty_cycle_csv = duty_cycle_csv['IFB_HTR1_LCL5_TM'].dropna()
duty_cycle_csv = duty_cycle_csv.dropna(subset=['IFB_HTR1_LCL5_TM'])

segments = capture_profiles(duty_cycle_csv['IFB_HTR1_LCL5_TM'], threshold)  # Passing the column directly
segments_df = pd.DataFrame(duty_cycle_csv['Time'][segments])
fh.segments_minus_time('segments.csv', 7)

# Extract the profiles
fh.extract_profiles(segments_df, cdf_file, 'profiles')


