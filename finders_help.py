import spacepy
import os
import spacepy.pycdf as cdf
import numpy as np
import pandas as pd
import re
from datetime import datetime
from datetime import timedelta
import h5py

def load_l1_data(cdf_file):
    cdf_file_data = cdf.CDF(cdf_file)
    time = cdf_file_data['EPOCH'][:]
    parts = re.split(r'[_-]', cdf_file)
    if parts[-4] == "obs":
        B_OBS1 = cdf_file_data['B_OBS_URF'][:,0]
        B_OBS2 = cdf_file_data['B_OBS_URF'][:,1]
        B_OBS3 = cdf_file_data['B_OBS_URF'][:,2]
        time = cdf_file_data['EPOCH'][:]
    elif parts[-4] == "ibs":
        B_OBS1 = cdf_file_data['B_IBS_URF'][:,0]
        B_OBS2 = cdf_file_data['B_IBS_URF'][:,1]
        B_OBS3 = cdf_file_data['B_IBS_URF'][:,2]
        time = cdf_file_data['EPOCH'][:]
    else:
        print("Error: cdf file not recognised")
    cdf_file_data.close()
    return time, B_OBS1, B_OBS2, B_OBS3
    
def segments_minus_time(segments, minus):
    segments_df_1 = pd.read_csv('segments.csv')
# Convert the 'Time' column to datetime objects
    segments_df_1['Time'] = pd.to_datetime(segments_df_1['Time'])
# Subtract 7 seconds from each time segment
    segments_df_1['Time'] = segments_df_1['Time'] - timedelta(seconds=minus)
# Save the modified DataFrame to CSV
    segments_df_1.to_csv('segments_seconds.csv', index=False)
    

def extract_profiles(segments_df, cdf_file, output_folder):
    time_prof, B_OBS1, B_OBS2, B_OBS3 = load_l1_data(cdf_file)
    time_prof = pd.to_datetime(time_prof)  
    profiles = []
    segments_real = pd.read_csv('segments_seconds.csv')
    for i in range(len(segments_df)):
        time_str = segments_real.iloc[i]['Time'] 
        time = pd.to_datetime(time_str) # Accessing time from DataFrame
        next_time = segments_real.iloc[i+1]['Time'] if i < len(segments_df) - 1 else None
        index = np.where(time_prof >= time)[0][0]
        
        if next_time is not None:
            next_index = np.where(time_prof >= next_time)[0][0]
            profile_time = time_prof[index:next_index]  # Segment from time to next_time
            profile_B_OBS1 = B_OBS1[index:next_index]
            profile_B_OBS2 = B_OBS2[index:next_index]
            profile_B_OBS3 = B_OBS3[index:next_index]
        else:
            profile_time = time_prof[index:]  # Segment from time to the end
            profile_B_OBS1 = B_OBS1[index:]
            profile_B_OBS2 = B_OBS2[index:]
            profile_B_OBS3 = B_OBS3[index:]
        profile_time_str = [str(dt) for dt in profile_time]    
        # Save profile as an HDF5 file
        with h5py.File(f"{output_folder}/profile_{i}.h5", "w") as hf:
            hf.create_dataset("time", data=np.array(profile_time_str, dtype='S'))
            hf.create_dataset("B_OBS1", data=profile_B_OBS1)
            hf.create_dataset("B_OBS2", data=profile_B_OBS2)
            hf.create_dataset("B_OBS3", data=profile_B_OBS3)
        
        profiles.append((profile_time, profile_B_OBS1, profile_B_OBS2, profile_B_OBS3))
        
        if next_time is None or time == segments_real.iloc[-1]['Time']:
            break
    
    return profiles

def extract_profiles_5(segments_df,segments_seconds, cdf_file, output_folder,current_date):
    time_prof, B_OBS1, B_OBS2, B_OBS3 = load_l1_data(cdf_file)
    time_prof = pd.to_datetime(time_prof)  
    profiles = []
    segments_real = segments_seconds

    date_str = current_date.strftime("%Y-%m-%d")
    date_folder = os.path.join(output_folder, date_str)
    os.makedirs(date_folder, exist_ok=True)
    
    for i in range(len(segments_df)):
        time_str = segments_real.iloc[i]['Time'] 
        time = pd.to_datetime(time_str)  # Accessing time from DataFrame
        next_time = segments_real.iloc[i+1]['Time'] if i < len(segments_df) - 1 else None
        start_indices = np.where(time_prof >= time)[0]
        if start_indices.size == 0:
            print(f"No start index found for time {time}")
            continue
        index = start_indices[0]
        # Determine end time for the 5-minute segment
        end_time = time + pd.Timedelta(minutes=5)
        
        # Find index corresponding to end time
        end_indices = np.where(time_prof >= end_time)[0]
        if end_indices.size == 0:
            end_index = len(time_prof)  # Use the length of time_prof if no end time is found
        else:
            end_index = end_indices[0]
        
        if end_index is not None:
            profile_time = time_prof[index:end_index]  # Segment from time to end_time
            profile_B_OBS1 = B_OBS1[index:end_index]
            profile_B_OBS2 = B_OBS2[index:end_index]
            profile_B_OBS3 = B_OBS3[index:end_index]
            
            profile_time_str = [str(dt) for dt in profile_time]
            
            # Save profile as an HDF5 file
            with h5py.File(f"{date_folder}/profile_{i}.h5", "w") as hf:
                hf.create_dataset("time", data=np.array(profile_time_str, dtype='S'))
                hf.create_dataset("B_OBS1", data=profile_B_OBS1)
                hf.create_dataset("B_OBS2", data=profile_B_OBS2)
                hf.create_dataset("B_OBS3", data=profile_B_OBS3)
            
            profiles.append((profile_time, profile_B_OBS1, profile_B_OBS2, profile_B_OBS3))
        
        if next_time is None or time == segments_real.iloc[-1]['Time']:
            break
    
    return profiles



def extract_profiles_10(segments_df, cdf_file, output_folder):
    time_prof, B_OBS1, B_OBS2, B_OBS3 = load_l1_data(cdf_file)
    time_prof = pd.to_datetime(time_prof)  
    profiles = []
    segments_real = pd.read_csv('segments_seconds.csv')
    
    for i in range(len(segments_df)):
        time_str = segments_real.iloc[i]['Time'] 
        time = pd.to_datetime(time_str)  # Accessing time from DataFrame
        next_time = segments_real.iloc[i+1]['Time'] if i < len(segments_df) - 1 else None
        index = np.where(time_prof >= time)[0][0]
        
        # Determine end time for the 5-minute segment
        end_time = time + pd.Timedelta(minutes=10)
        
        # Find index corresponding to end time
        end_index = np.where(time_prof >= end_time)[0][0]
        
        if end_index is not None:
            profile_time = time_prof[index:end_index]  # Segment from time to end_time
            profile_B_OBS1 = B_OBS1[index:end_index]
            profile_B_OBS2 = B_OBS2[index:end_index]
            profile_B_OBS3 = B_OBS3[index:end_index]
            
            profile_time_str = [str(dt) for dt in profile_time]
            
            # Save profile as an HDF5 file
            with h5py.File(f"{output_folder}/profile_{i}.h5", "w") as hf:
                hf.create_dataset("time", data=np.array(profile_time_str, dtype='S'))
                hf.create_dataset("B_OBS1", data=profile_B_OBS1)
                hf.create_dataset("B_OBS2", data=profile_B_OBS2)
                hf.create_dataset("B_OBS3", data=profile_B_OBS3)
            
            profiles.append((profile_time, profile_B_OBS1, profile_B_OBS2, profile_B_OBS3))
        
        if next_time is None or time == segments_real.iloc[-1]['Time']:
            break
    
    return profiles

def load_l2_data(cdf_file):
    cdf_file_data = cdf.CDF(cdf_file)
    time = cdf_file_data['EPOCH'][:]
    B_R = cdf_file_data['B_RTN'][:,0]
    B_T = cdf_file_data['B_RTN'][:,1]
    B_N = cdf_file_data['B_RTN'][:,2]
    return time, B_R, B_T, B_N

def extract_profiles_l2_10(segments_df,segments_seconds, cdf_file, output_folder,current_date):
    time_prof, B_R, B_T, B_N = load_l2_data(cdf_file)
    time_prof = pd.to_datetime(time_prof)  
    profiles = []
    segments_real = segments_seconds

    date_str = current_date.strftime("%Y-%m-%d")
    date_folder = os.path.join(output_folder, date_str)
    os.makedirs(date_folder, exist_ok=True)

    for i in range(len(segments_df)):
        time_str = segments_real.iloc[i]['Time'] 
        time = pd.to_datetime(time_str)  # Accessing time from DataFrame
        next_time = segments_real.iloc[i+1]['Time'] if i < len(segments_df) - 1 else None
        start_indices = np.where(time_prof >= time)[0]
        if start_indices.size == 0:
            print(f"No start index found for time {time}")
            continue
        index = start_indices[0]
        
        # Determine end time for the 5-minute segment
        end_time = time + pd.Timedelta(minutes=10)
        
        # Find index corresponding to end time
        end_indices = np.where(time_prof >= end_time)[0]
        if end_indices.size == 0:
            end_index = len(time_prof)  # Use the length of time_prof if no end time is found
        else:
            end_index = end_indices[0]
        
        if end_index is not None:
            profile_time = time_prof[index:end_index]  # Segment from time to end_time
            profile_B_R = B_R[index:end_index]
            profile_B_T = B_T[index:end_index]
            profile_B_N = B_N[index:end_index]
            
            profile_time_str = [str(dt) for dt in profile_time]
            
            # Save profile as an HDF5 file
            with h5py.File(f"{date_folder}/profile_{i}.h5", "w") as hf:
                hf.create_dataset("time", data=np.array(profile_time_str, dtype='S'))
                hf.create_dataset("B_R", data=profile_B_R)
                hf.create_dataset("B_T", data=profile_B_T)
                hf.create_dataset("B_N", data=profile_B_N)
            
            profiles.append((profile_time, profile_B_R, profile_B_T, profile_B_N))
        
        if next_time is None or time == segments_real.iloc[-1]['Time']:
            break
    
    return profiles

def extract_profiles_l2_5(segments_df,segments_seconds, cdf_file, output_folder,current_date):
    time_prof, B_R, B_T, B_N = load_l2_data(cdf_file)
    time_prof = pd.to_datetime(time_prof)  
    profiles = []
    segments_real = segments_seconds

    date_str = current_date.strftime("%Y-%m-%d")
    date_folder = os.path.join(output_folder, date_str)
    os.makedirs(date_folder, exist_ok=True)

    for i in range(len(segments_df)):
        time_str = segments_real.iloc[i]['Time'] 
        time = pd.to_datetime(time_str)  # Accessing time from DataFrame
        next_time = segments_real.iloc[i+1]['Time'] if i < len(segments_df) - 1 else None
        start_indices = np.where(time_prof >= time)[0]
        if start_indices.size == 0:
            print(f"No start index found for time {time}")
            continue
        index = start_indices[0]
        
        # Determine end time for the 5-minute segment
        end_time = time + pd.Timedelta(minutes=5)
        
        # Find index corresponding to end time
        end_indices = np.where(time_prof >= end_time)[0]
        if end_indices.size == 0:
            end_index = len(time_prof)  # Use the length of time_prof if no end time is found
        else:
            end_index = end_indices[0]
        
        if end_index is not None:
            profile_time = time_prof[index:end_index]  # Segment from time to end_time
            profile_B_R = B_R[index:end_index]
            profile_B_T = B_T[index:end_index]
            profile_B_N = B_N[index:end_index]
            
            profile_time_str = [str(dt) for dt in profile_time]
            
            # Save profile as an HDF5 file
            with h5py.File(f"{date_folder}/profile_{i}.h5", "w") as hf:
                hf.create_dataset("time", data=np.array(profile_time_str, dtype='S'))
                hf.create_dataset("B_R", data=profile_B_R)
                hf.create_dataset("B_T", data=profile_B_T)
                hf.create_dataset("B_N", data=profile_B_N)
            
            profiles.append((profile_time, profile_B_R, profile_B_T, profile_B_N))
        
        if next_time is None or time == segments_real.iloc[-1]['Time']:
            break
    
    return profiles


def get_latest_version_file(file_location, current_date):
    base_pattern = f'solo_L1_mag-ibs-normal_{current_date.strftime("%Y%m%d")}_V'
    all_files = os.listdir(file_location)
    matching_files = [f for f in all_files if f.startswith(base_pattern) and f.endswith('.cdf')]
    
    version_numbers = []
    for file in matching_files:
        match = re.search(r'_V(\d{2})\.cdf$', file)
        if match:
            version_numbers.append(int(match.group(1)))
    
    if not version_numbers:
        return None  # No matching files found
    
    highest_version = max(version_numbers)
    latest_version_file = f'{file_location}/solo_L1_mag-ibs-normal_{current_date.strftime("%Y%m%d")}_V{highest_version:02d}.cdf'
    return latest_version_file

def capture_profiles(duty_cycle_csv, threshold):
    segments = []
    in_segment = False
    for index, row in duty_cycle_csv.iterrows():
        if row['IFB_HTR1_LCL5_TM'] > threshold:
            if not in_segment:
                segments.append(index)
                in_segment = True
        else:
            in_segment = False
    return segments

