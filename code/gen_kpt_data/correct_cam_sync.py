import json
import pandas as pd
from datetime import datetime
from fastprogress import progress_bar

root_path = 'path/to/data/'

# Load JSON and CSV data
with open(f'{root_path}data/stomp_sync_init.json', 'r') as f:
# with open('data_stomps/stomp_sync_init_new.json', 'r') as f:
    stomp_sync_init = json.load(f)

with open(f'{root_path}data/stomp_sync_refine_manual.json', 'r') as f:
    stomp_sync_refine = json.load(f)

with open(f'{root_path}data/stomp_3d_manual.json', 'r') as f:
    stomp_3d_manual = json.load(f)

df = pd.read_csv(f'{root_path}data/times/all_times_w_drops.csv')

# new_dict = {}
with open(f'{root_path}data/stomp_sync_corrected.json', 'r') as f:
    new_dict = json.load(f)

for key, val in progress_bar(stomp_sync_refine.items(), total=len(stomp_sync_refine)):
    start_idx = val
    combination = key
    group_id = key[2:]
    cam = key[:2]
    subject = group_id.split('A')[0]

    if f'C4{group_id}' not in stomp_3d_manual:
        print(f'C4{group_id} not in stomp_3d_manual')
        continue

    c4_start_idx = stomp_sync_init[f'C4{group_id}']['first_idx_w_drop']
    c4_end_idx = stomp_sync_init[f'C4{group_id}']['last_idx_w_drop']
    num_frames = c4_end_idx - c4_start_idx + 1

    vicon_start_idx = stomp_3d_manual[f'C4{group_id}']['first_idx']
    vicon_end_idx = stomp_3d_manual[f'C4{group_id}']['last_idx']

    # get frame_num_w_drop from df with combination and frame_num start_idx
    if cam == 'C4':
        start_idx_w_drop = c4_start_idx
        end_idx_w_drop = start_idx_w_drop + num_frames - 1
    else:
        start_idx_w_drop = df.loc[(df['combination'] == combination) & (df['frame_num'] == start_idx), 'frame_num_w_drop'].values[0]
        # start_idx_w_drop = stomp_sync_init[key]['first_idx_w_drop']
        end_idx_w_drop = start_idx_w_drop + num_frames - 1
    try:
        # start_idx = df.loc[(df['combination'] == combination) & (df['frame_num_w_drop'] == start_idx_w_drop), 'frame_num'].values[0]
        end_idx = df.loc[(df['combination'] == combination) & (df['frame_num_w_drop'] == end_idx_w_drop), 'frame_num'].values[0]
    except:
        print(f'Error: {combination} - {start_idx} - {end_idx_w_drop}')
        continue
    new_dict[combination] = {'first_idx': start_idx, 'first_idx_w_drop': start_idx_w_drop, 'first_idx_vicon': vicon_start_idx, 'last_idx': end_idx, 'last_idx_w_drop': end_idx_w_drop, 'last_idx_vicon': vicon_end_idx}

    new_dict = {k: {subk: int(subv) for subk, subv in v.items()} for k, v in new_dict.items()}

    with open(f'{root_path}data/stomp_sync_corrected.json', 'w') as f:
        json.dump(new_dict, f, indent=4)
        print(f'Saved: {combination}')