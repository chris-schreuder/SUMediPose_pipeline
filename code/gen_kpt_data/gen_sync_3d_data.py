import json
import pandas as pd
from datetime import datetime
from fastprogress import progress_bar
import numpy as np
from scipy.interpolate import interp1d
import os

root_path = 'path/to/data/'

def interpolate_keypoints(keypoints, new_length):
    original_length = keypoints.shape[0]
    original_indices = np.arange(original_length)
    new_indices = np.linspace(0, original_length - 1, new_length)
    new_keypoints = np.zeros((new_length, keypoints.shape[1], keypoints.shape[2]))

    for i in range(keypoints.shape[1]):  # Iterate over each keypoint (32 keypoints)
        for j in range(keypoints.shape[2]):  # Iterate over x and y coordinates
            interp_func = interp1d(original_indices, keypoints[:, i, j], kind='nearest')
            new_keypoints[:, i, j] = interp_func(new_indices)

    return new_keypoints

# Load JSON and CSV data
# with open(f'{root_path}data/stomp_sync_corrected.json', 'r') as f:
#     stomp_sync_init = json.load(f)

with open(f'{root_path}data/stomp_sync_corrected_manual.json', 'r') as f:
    stomp_sync_init = json.load(f)

for key, val in progress_bar(stomp_sync_init.items(), total=len(stomp_sync_init)):

    new_data = []

    start_idx = val
    combination = key
    group_id = key[2:]
    cam = key[:2]
    subject = group_id.split('A')[0]
    if cam != 'C4':
        continue

    print(f'{cam} - {subject} - {group_id}')

    first_idx_vicon = val['first_idx_vicon']
    last_idx_vicon = val['last_idx_vicon']
    first_idx_w_drop = val['first_idx_w_drop']
    last_idx_w_drop = val['last_idx_w_drop']

    vicon_num_frames = last_idx_vicon - first_idx_vicon + 1
    num_frames = last_idx_w_drop - first_idx_w_drop + 1

    path_3d = f'{root_path}3d_data/{subject}/{group_id}.json'
    with open(path_3d, 'r') as f:
        frames = json.load(f)

    try:
        frames = frames[first_idx_vicon:last_idx_vicon+1]
        point_ids = frames[0]['point_ids']
        xyz_wand = frames[0]['xyz_wand']
        all_keypoints = []
        for frame in frames:
            all_keypoints.append(frame['xyz'])
        all_keypoints = np.array(all_keypoints)
    except Exception as e:
        print(f'Error: {subject} - {combination} - {e}')
        continue

    new_vicon_keypoints = interpolate_keypoints(all_keypoints, num_frames)

    for i, frame in enumerate(new_vicon_keypoints):
        new_data.append({
            'frame_num': i,
            'point_ids': point_ids,
            'xyz': frame.tolist(),
            'xyz_wand': xyz_wand
        })

    os.makedirs(f'{root_path}WCS/{subject}', exist_ok=True)
    with open(f'{root_path}WCS/{subject}/{group_id}.json', 'w') as f:
        json.dump(new_data, f, indent=4)

    