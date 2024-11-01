import json
import pandas as pd
from datetime import datetime
from fastprogress import progress_bar
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

root_path = 'path/to/your/data/'

# Load JSON and CSV data
with open(f'{root_path}data/stomp_refine_manual.json', 'r') as f:
    stomp_sync_init_all = json.load(f)

with open(f'{root_path}data/stomp_3d_manual.json', 'r') as f:
    stomp_3d_manual = json.load(f)


pbar = tqdm(total=len(stomp_sync_init_all))

for key, val in stomp_sync_init_all.items():
    first_idx = val['first_idx']
    last_idx = val['last_idx']
    combination = key
    group_id = key[2:]
    cam = key[:2]
    subject = f"{group_id.split('A')[0]}"

    print(f'{cam} - {subject} - {group_id}')
    try:
        c4_start_idx = stomp_3d_manual[f'C4{group_id}']['first_idx']
        c4_end_idx = stomp_3d_manual[f'C4{group_id}']['last_idx']
    except:
        print(f'{group_id} not in stomp_3d_manual')
        continue
    internal_path = f'{root_path}internal_data/{cam}/{subject}/{group_id}.json'
    with open(internal_path, 'r') as f:
        internal_frames = json.load(f)

    internal_frame = internal_frames[c4_start_idx]
    internal_points = internal_frame['point_ids']
    internal_xy = internal_frame['xy']
    idxs = [i for i in range(first_idx-20, first_idx+20)]

    for idx in idxs:
        try:
            img_path = f'{root_path}frames/{cam}/{subject}/{combination}/{combination}_{str(idx).zfill(4)}.jpg'
            path = f'{root_path}data/sync_c4/{cam}/{subject}/{combination}/{combination}_{str(idx).zfill(4)}({str(first_idx).zfill(4)}).jpg'
            if os.path.exists(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
 
            image = cv2.imread(img_path)
            x_vals = [int(coord[0]) for coord in internal_xy if not np.isnan(coord[0])]
            y_vals = [int(coord[1]) for coord in internal_xy if not np.isnan(coord[1])]

            for x, y in zip(x_vals, y_vals):
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imwrite(path, image)
        except Exception as e:
            print(e)
            continue

    internal_frame = internal_frames[c4_end_idx]
    internal_points = internal_frame['point_ids']
    internal_xy = internal_frame['xy']
    idxs = [i for i in range(last_idx-20, last_idx+20)]

    for idx in idxs:
        try:
            img_path = f'{root_path}frames/{cam}/{subject}/{combination}/{combination}_{str(idx).zfill(4)}.jpg'
            path = f'{root_path}data/sync_c4/{cam}/{subject}/{combination}/{combination}_{str(idx).zfill(4)}({str(last_idx).zfill(4)}).jpg'
            if os.path.exists(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image = cv2.imread(img_path)
            x_vals = [int(coord[0]) for coord in internal_xy if not np.isnan(coord[0])]
            y_vals = [int(coord[1]) for coord in internal_xy if not np.isnan(coord[1])]

            for x, y in zip(x_vals, y_vals):
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imwrite(path, image)
        except Exception as e:
            print(e)
            continue
    pbar.update(1)