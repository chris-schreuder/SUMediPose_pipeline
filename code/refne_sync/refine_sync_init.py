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
with open(f'{root_path}data/stomp_sync_init.json', 'r') as f:
    stomp_sync_init_all = json.load(f)

with open(f'{root_path}data/stomp_3d_manual.json', 'r') as f:
    stomp_3d_manual = json.load(f)



new_dict = {}

pbar = tqdm(total=len(stomp_sync_init_all))

for key, val in stomp_sync_init_all.items():

    if key in new_dict:
        pbar.update(1)
        continue

    first_idx_w_drop = val['first_idx_w_drop']
    last_idx_w_drop = val['last_idx_w_drop']
    first_idx = val['first_idx']
    last_idx = val['last_idx']
    combination = key
    group_id = key[2:]
    cam = key[:2]
    subject = f"{group_id.split('A')[0]}"
    print(f'{cam} - {subject} - {group_id}')
    if cam == 'C4':
        try:
            new_dict[key] = stomp_sync_init_all[key]['first_idx']
            with open(f'{root_path}data/stomp_sync_refine_v2.json', 'w') as f:
                json.dump(new_dict, f, indent=4)
            continue
        except:
            continue
   
    try:
        c4_start_idx = stomp_3d_manual[f'C4{group_id}']['first_idx']
    except:
        print(f'{group_id} not in stomp_3d_manual')
        continue
    internal_path = f'{root_path}internal_data/{cam}/{subject}/{group_id}.json'
    with open(internal_path, 'r') as f:
        internal_frames = json.load(f)
    yolo_path = f'{root_path}data/data_yolo/frames/{cam}/{subject}/{combination}.json'
    with open(yolo_path, 'r') as f:
        yolo_frames = json.load(f)
   
    internal_frame = internal_frames[c4_start_idx]
    internal_points = internal_frame['point_ids']
    internal_xy = internal_frame['xy']
    yolo_frames = yolo_frames[first_idx-3:first_idx+4]
    idxs = [i for i in range(first_idx-3, first_idx+4)]
    best_idx = first_idx
    best_error = 10000000

    fig, axs = plt.subplots(2, 4, figsize=(50, 30))
    axs = axs.flatten()
    plot_idx = 0

    for i, (yolo_frame, idx) in enumerate(zip(yolo_frames, idxs)):
        img_path = f'{root_path}frames/{cam}/{subject}/{combination}/{combination}_{str(idx).zfill(4)}.jpg'
     
        yolo_xy = yolo_frame['keypoints'][0][0]
        yolo_xy = [coord[:3] for coord in yolo_xy]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        error_total = 0
        err_cnt = 0
        for int_i, yolo_i in zip(['LWJC', 'LWRA', 'LFRM', 'LWRB', 'RWJC', 'RWRA', 'RFRM', 'RELB', 'LAJC', 'RAJC', 'RFHD', 'LFHD'], [9, 9, 9, 9, 10, 10, 10, 10, 15, 16, 4, 3]):
            try:
                idx_temp = internal_points.index(int_i)
                int_temp = internal_xy[idx_temp]
                yolo_temp = yolo_xy[yolo_i]
                if yolo_temp[2] < 0.8:
                    continue
                error_total += (int_temp[0] - yolo_temp[0])**2 + (int_temp[1] - yolo_temp[1])**2
                err_cnt += 1
                cv2.circle(image, (int(int_temp[0]), int(int_temp[1])), 3, (0, 0, 255), -1)
                cv2.circle(image, (int(yolo_temp[0]), int(yolo_temp[1])), 3, (0, 255, 0), -1)
                cv2.putText(image, f'{int_i}', (int(int_temp[0]), int(int_temp[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, f'{yolo_i}', (int(yolo_temp[0]), int(yolo_temp[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            except:
                print(f'{int_i} not found - yolo')
                continue

        if err_cnt != 0:

            avg_error = error_total / err_cnt
            rms_error = np.sqrt(avg_error)

            if rms_error < best_error:
                best_error = rms_error
                best_idx = idx

            axs[plot_idx].imshow(image)
            axs[plot_idx].axis('off')
            axs[plot_idx].set_title(f'Frame {idx} ({first_idx}) - Error: {rms_error:.2f}', fontsize=25)
            plot_idx += 1
        else:
            print(f'No points found - {combination}')
            avg_error = None
            rms_error = None
            axs[plot_idx].imshow(image)
            axs[plot_idx].axis('off')
            axs[plot_idx].set_title(f'Frame {idx} ({first_idx}) - Error: None', fontsize=25)
            plot_idx += 1
        
        
        if plot_idx >= 8:  # Only plot 8 images
            break

    if plot_idx >= 8:
        break

    #save fig
    os.makedirs(f'{root_path}data/sync_cams_stomps_v2/{cam}/{subject}', exist_ok=True)
    plt.savefig(f'{root_path}data/sync_cams_stomps_v2/{cam}/{subject}/{key}.png')
    plt.close()

    new_dict[key] = best_idx
    with open(f'{root_path}data/stomp_sync_refine_v2.json', 'w') as f:
        json.dump(new_dict, f, indent=4)

    pbar.update(1)