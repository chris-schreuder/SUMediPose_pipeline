import json
import pandas as pd
from datetime import datetime
from fastprogress import progress_bar

root_path = 'path/to/data/'

# Load JSON and CSV data
with open(f'{root_path}data/stomp_refine_manual.json', 'r') as f:
    stomp_refine_all = json.load(f)

df = pd.read_csv(f'{root_path}data/times/all_times_w_drops.csv')

# Ensure the time column is properly converted to datetime
df['time_obj'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')

# Function to extract camera and time
def extract_info_from_path(df, num, combination):
    row = df.loc[(df['frame_num'] == num) & (df['combination'] == combination)]
    if row.empty:
        return None, None
    camera = row['camera'].values[0]
    time_str = row['time'].values[0]
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    return camera, time_obj


with open(f'{root_path}data/stomp_sync_init.json', 'r') as f:
    new_dict = json.load(f)

# Process each key-value pair in the JSON
for key, val in progress_bar(stomp_refine_all.items(), total=len(stomp_refine_all)):
    start_idx = val['first_idx']
    end_idx = val['last_idx']
    group_id = key.replace('C4', '')
    subject = group_id.split('A')[0]

    df_temp = df[df['group_id'] == group_id].copy()
    
    # Process start index
    try:
        given_camera, given_time = extract_info_from_path(df_temp, start_idx, key)
    except Exception as e:
        print(e)
        given_camera, given_time = None, None

    if given_camera is None:
        continue
    
    closest_paths = {}
    for camera in df_temp['camera'].unique():
        camera_df = df_temp.loc[df_temp['camera'] == camera].copy()
        camera_df['time_diff'] = (camera_df['time_obj'] - given_time).abs()
        closest_row = camera_df.loc[camera_df['time_diff'].idxmin()]
        closest_paths[camera] = closest_row['path']
        combination_temp = closest_row['combination']
        idx_temp = closest_row['frame_num']
        frame_num_w_drop = closest_row['frame_num_w_drop']
        if combination_temp not in new_dict:
            new_dict[combination_temp] = {}
        new_dict[combination_temp]['first_idx'] = idx_temp
        new_dict[combination_temp]['first_idx_w_drop'] = frame_num_w_drop

    # Process end index
    try:
        given_camera, given_time = extract_info_from_path(df_temp, end_idx, key)
    except Exception as e:
        print(e)
        given_camera, given_time = None, None

    if given_camera is None:
        continue
    
    for camera in df_temp['camera'].unique():
        camera_df = df_temp.loc[df_temp['camera'] == camera].copy()
        camera_df['time_diff'] = (camera_df['time_obj'] - given_time).abs()
        closest_row = camera_df.loc[camera_df['time_diff'].idxmin()]
        closest_paths[camera] = closest_row['path']
        combination_temp = closest_row['combination']
        idx_temp = closest_row['frame_num']
        frame_num_w_drop = closest_row['frame_num_w_drop']
        if combination_temp not in new_dict:
            new_dict[combination_temp] = {}
        new_dict[combination_temp]['last_idx'] = idx_temp
        new_dict[combination_temp]['last_idx_w_drop'] = frame_num_w_drop

    new_dict = {k: {subk: int(subv) for subk, subv in v.items()} for k, v in new_dict.items()}

# Save the new dictionary to a JSON file
with open(f'{root_path}data/stomp_sync_init.json', 'w') as f:
    json.dump(new_dict, f, indent=4)
    print(f"Saved the new dictionary to '{root_path}data/stomp_sync_init.json'")
