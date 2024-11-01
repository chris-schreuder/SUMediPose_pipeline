from glob import glob
from fastprogress import progress_bar
import json
import pandas as pd
import cv2
import os
import numpy as np

search_window = 400

root_path = 'path/to/your/root/directory/'

files = glob(f'{root_path}internal_data/C4/S9/S9A9D2.json')
# files = files[:1]
# df_3d_stomp = pd.read_csv(f'{root_path}data_stomps/stomps_3d.csv')
df_3d_stomp = pd.read_csv(f'{root_path}data/stomps_3d.csv')

with open(f'{root_path}data/stomp_refine_manual.json', 'r') as f:
    stomp_refine_all = json.load(f)

for file in progress_bar(files):
    combination = file.split('/')[-1].replace('.json', '')
    print(combination)
    subject = file.split('/')[-2]
    # if subject == 'S1':
    #     continue
    # if subject in ['S1', 'S10', 'S11', 'S12', 'S13', 'S14']:
    #     continue
    with open(file, 'r') as f:
        frames = json.load(f)

    df_temp = df_3d_stomp[df_3d_stomp['combination'] == combination]
    vicon_first_idx = df_temp['first_idx'].values[0]
    vicon_last_idx = df_temp['last_idx'].values[0]

    print(vicon_first_idx, vicon_last_idx)

    if vicon_first_idx == None or vicon_last_idx == None:
        continue

    # combination = file.split('/')[-1].replace('.json', '').replace('D3', 'D2')
    # combination = 'S1A4D3'
    try:
        stomp_refine = stomp_refine_all['C4'+combination]
    except:
        print(f'{combination} not in stomp_refine_all')
        continue
    image_first = stomp_refine['first_idx']
    image_last = stomp_refine['last_idx']
    print(image_first, image_last)
    image_path_first = f'{root_path}frames/C4/{subject}/C4{combination}/C4{combination}_{str(image_first).zfill(4)}.jpg'
    image_path_last = f'{root_path}frames/C4/{subject}/C4{combination}/C4{combination}_{str(image_last).zfill(4)}.jpg'
    if not os.path.exists(image_path_first) or not os.path.exists(image_path_last):
        print(f'Image not found: {image_path_first} or {image_path_last}')
        continue
    for i, frame in progress_bar(enumerate(frames), total=len(frames)):
        img_first = cv2.imread(image_path_first)
        img_last = cv2.imread(image_path_last)
        if (i > (vicon_first_idx-search_window) and i < (vicon_first_idx+search_window)) or (i > (vicon_last_idx-search_window) and i < (vicon_last_idx+search_window)):
            path =f'{root_path}data/sync_3d_2d_test/{combination}/{combination}_{str(i).zfill(4)}.jpg'
            if os.path.exists(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pixel_vals = frame['xy']
            x_vals = [int(coord[0]) for coord in pixel_vals if not np.isnan(coord[0])]
            y_vals = [int(coord[1]) for coord in pixel_vals if not np.isnan(coord[1])]
            
            if i > (vicon_first_idx-search_window) and i < (vicon_first_idx+search_window):
                for x, y in zip(x_vals, y_vals):
                    cv2.circle(img_first, (x, y), 5, (0, 0, 255), -1)
                cv2.imwrite(path, img_first)
            elif i > (vicon_last_idx-search_window) and i < (vicon_last_idx+search_window):
                for x, y in zip(x_vals, y_vals):
                    cv2.circle(img_last, (x, y), 5, (0, 0, 255), -1)
                cv2.imwrite(path, img_last)
        

