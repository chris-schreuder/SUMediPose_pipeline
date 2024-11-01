import numpy as np
import cv2
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
from fastprogress import progress_bar
import json
from glob import glob
import os

ids_map = {
    "C7": "C7",
    "CLAV": "CLAV",
    "CentreOfMass": "CentreOfMass",
    "CentreOfMassFloor": "CentreOfMassFloor",
    "LAJC": "LAJC",
    "LANK": "LTOE",
    "LASI": "LPSI",
    "LBHD": "LBHD",
    "LEJC": "LEJC",
    "LELB": "LFRM",
    "LFHD": "LFHD",
    "LFIN": "RSHO",
    "LFRM": "LWRA",
    "LHEE": "RTHI",
    "LHJC": "LHJC",
    "LKJC": "LKJC",
    "LKNE": "LANK",
    "LMMED": "R_Foot_Out",
    "LPSI": "LTHI",
    "LSHO": "LSHO",
    "LSJC": "LSJC",
    "LTHI": "LTIB",
    "LTIB": "LHEE",
    "LTOE": "RKNE",
    "LUPA": "LELB",
    "LWJC": "LWJC",
    "LWRA": "LWRB",
    "LWRB": "LFIN",
    "L_Foot_Out": "AAAAAAAA",
    "MElbowL": "MKNEL",
    "MElbowR": "MKNER",
    "MKNEL": "LMMED",
    "MKNER": "RMMED",
    "PelL": "MElbowL",
    "PelR": "MElbowR",
    "RAJC": "RAJC",
    "RANK": "RTOE",
    "RASI": "RPSI",
    "RBAK": "RBAK",
    "RBHD": "RBHD",
    "REJC": "REJC",
    "RELB": "RWRA",
    "RFHD": "RFHD",
    "RFIN": "RASI",
    "RFRM": "RWRB",
    "RHEE": "PelR",
    "RHJC": "RHJC",
    "RKJC": "RKJC",
    "RKNE": "RANK",
    "RMMED": "L_Foot_Out",
    "RPSI": "LKNE",
    "RSHO": "RELB",
    "RSJC": "RSJC",
    "RTHI": "RTIB",
    "RTIB": "RHEE",
    "RTOE": "PelL",
    "RUPA": "RFRM",
    "RWJC": "RWJC",
    "RWRA": "RFIN",
    "RWRB": "LASI",
    "R_Foot_Out": "BBBBBBB",
    "STRN": "STRN",
    "T10": "T10"
}

root_path = 'path/to/data/'

# with open('data_stomps/stomp_sync_corrected.json', 'r') as f:
with open(f'{root_path}data/stomp_sync_corrected_manual.json', 'r') as f:
    stomp_sync_init = json.load(f)

df_drops = pd.read_csv(f'{root_path}data/times/all_times_w_drops.csv')


df_cam_matrix = pd.read_csv(f'{root_path}calibration/external_calibration.csv')
matrix_subjects = df_cam_matrix['subject'].unique().tolist()
matrix_subjects = sorted(matrix_subjects)
matrix_subjects = [int(subj.replace('S', '')) for subj in matrix_subjects]
map_matrix = {}
cur_num = 1
for i in range(1,29):
    if i in matrix_subjects:
        cur_num = i
    map_matrix[f'S{i}'] = f'S{cur_num}'

with open(f'{root_path}data/markers_points.json', 'r') as f:
    markers = json.load(f)
keys = list(markers.keys())
subject_nums = [int(key.replace('S', '')) for key in keys if 'points' in markers[key]['C1']]
subject_nums = sorted(subject_nums)
map_nums = {}
cur_num = 1
for i in range(1,29):
    if i in subject_nums:
        cur_num = i
    map_nums[f'S{i}'] = f'S{cur_num}'

c1_data = []
c2_data = []
c3_data = []
c4_data = []
c5_data = []
c6_data = []

files = glob(f'{root_path}WCS/S*/*.json')

for file in progress_bar(files):
    c1_data = []
    c2_data = []
    c3_data = []
    c4_data = []
    c5_data = []
    c6_data = []
    subject = file.split('/')[-2]
    combintation = file.split('/')[-1].replace('.json', '')

    path_temp = f'{root_path}internal_data/C6/{subject}/{combintation}.json'
    # if os.path.exists(path_temp):
    #     continue

    filtered_df_c1 = df_drops[df_drops['combination'] == f'C1{combintation}']
    filtered_df_c2 = df_drops[df_drops['combination'] == f'C2{combintation}']
    filtered_df_c3 = df_drops[df_drops['combination'] == f'C3{combintation}']
    filtered_df_c4 = df_drops[df_drops['combination'] == f'C4{combintation}']
    filtered_df_c5 = df_drops[df_drops['combination'] == f'C5{combintation}']
    filtered_df_c6 = df_drops[df_drops['combination'] == f'C6{combintation}']

    c1_yolo_data = []
    yolo_path = f'{root_path}data/data_yolo/frames/C1/{subject}/C1{combintation}.json'
    with open(yolo_path, 'r') as f:
        c1_yolo_data = json.load(f)
    c2_yolo_data = []
    yolo_path = f'{root_path}data/data_yolo/frames/C2/{subject}/C2{combintation}.json'
    with open(yolo_path, 'r') as f:
        c2_yolo_data = json.load(f)
    c3_yolo_data = []
    yolo_path = f'{root_path}data/data_yolo/frames/C3/{subject}/C3{combintation}.json'
    with open(yolo_path, 'r') as f:
        c3_yolo_data = json.load(f)
    c4_yolo_data = []
    yolo_path = f'{root_path}data/data_yolo/frames/C4/{subject}/C4{combintation}.json'
    with open(yolo_path, 'r') as f:
        c4_yolo_data = json.load(f)
    c5_yolo_data = []
    yolo_path = f'{root_path}data/data_yolo/frames/C5/{subject}/C5{combintation}.json'
    with open(yolo_path, 'r') as f:
        c5_yolo_data = json.load(f)
    c6_yolo_data = []
    yolo_path = f'{root_path}data/data_yolo/frames/C6/{subject}/C6{combintation}.json'
    with open(yolo_path, 'r') as f:
        c6_yolo_data = json.load(f)

    print(f'Processing {subject} {combintation}')
    with open(file, 'r') as f:
        frames = json.load(f)
    for cnt, frame in progress_bar(enumerate(frames), total=len(frames)):
        xyz = frame['xyz']
        xyz_wand = frame['xyz_wand']
        point_ids = frame['point_ids']
        point_ids_temp = [ids_map[point_id] for point_id in point_ids]
        point_ids = point_ids_temp
        floor_idx = point_ids.index('CentreOfMassFloor')
        # floor_y = xyz[floor_idx][1]
        points_skel = np.array(xyz)
        wcs_skel = []
        wcs_skel.append(points_skel)
        wcs_skel = np.array(wcs_skel)
        wcs_skel = wcs_skel.reshape(wcs_skel.shape[1], wcs_skel.shape[2])
        wcs_skel[:, 0] = wcs_skel[:, 0]/60
        wcs_skel[:, 1] = wcs_skel[:, 1]/60
        wcs_skel[:, 2] = wcs_skel[:, 2]/60

        wcs_wand = []
        wcs_wand.append(xyz_wand)
        wcs_wand = np.array(wcs_wand)
        wcs_wand = wcs_wand.reshape(wcs_wand.shape[1], wcs_wand.shape[2])
        wcs_wand[:, 0] = wcs_wand[:, 0]/60
        wcs_wand[:, 1] = wcs_wand[:, 1]/60
        wcs_wand[:, 2] = wcs_wand[:, 2]/60

        cal_subject = map_matrix[subject]
        cal_df = df_cam_matrix[df_cam_matrix['subject'] == cal_subject]

        for idx, row in cal_df.iterrows():
 
            main = row['main']
            pair = row['pair']
            path = row['path']
            cam = row['cam']
            method = row['method']
            rot_method = row['rot_method']

            if f'C{main}{combintation}' not in stomp_sync_init.keys():
                print(f'C{main}{combintation} not in stomp_sync_init')
                continue

            if main == 1:
                if pair == 2:
                    pairs = [2,3]
                else:
                    pairs = [3,2]
            else:
                if pair == 4:
                    pairs = [4,5]
                else:
                    pairs = [5,4]
            # print(f'{main} - {pairs}')
            cal_path_pair1 = f"{root_path}{cal_df[(cal_df['main'] == main) & (cal_df['pair'] == pairs[0])]['path'].values[0]}"
            cal_path_pair2 = f"{root_path}{cal_df[(cal_df['main'] == main) & (cal_df['pair'] == pairs[1])]['path'].values[0]}"
            
            data_pair1 = np.load(cal_path_pair1)
            data_pair2 = np.load(cal_path_pair2)

            R1 = np.eye(3)  
            T1 = np.array([0, 0, 0])  
            R2 = data_pair1['R']
            T2 = data_pair1['T']
            T2 = np.array([T2[0][0], T2[1][0], T2[2][0]]).reshape((3,1))
            R3 = data_pair2['R']
            T3 = data_pair2['T']
            T3 = np.array([T3[0][0], T3[1][0], T3[2][0]]).reshape((3,1))

            R1_initial = R1.copy()
            T1_initial = T1.reshape((3,1)).copy()
            R2_initial = R2.copy()
            T2_initial = T2.reshape((3,1)).copy()
            R3_initial = R3.copy()
            T3_initial = T3.reshape((3,1)).copy()

            cameraMatrix1 = data_pair1['mtx_A']
            cameraMatrix2 = data_pair1['mtx_B']
            cameraMatrix3 = data_pair2['mtx_B']

            RT1 = np.concatenate([R1, [[0],[0],[0]]], axis = -1)
            P1 = cameraMatrix1 @ RT1
            RT2 = np.concatenate([R2, T2], axis = -1)
            P2 = cameraMatrix2 @ RT2
            RT3 = np.concatenate([R3, T3], axis = -1)
            P3 = cameraMatrix3 @ RT3

            Ps = [P1, P2, P3]

            def DLT_2_cameras(imgpoints_Origin, imgpoints_Pair1, projMatr_Origin, projMatr_Pair1):
                points_3d_list = []
                for points_origin, points_pair1 in zip(imgpoints_Origin, imgpoints_Pair1):
                    point_origin = np.array(points_origin)
                    point_pair1 = np.array(points_pair1)

                    A = [point_origin[1]*projMatr_Origin[2,:] - projMatr_Origin[1,:],
                        projMatr_Origin[0,:] - point_origin[0]*projMatr_Origin[2,:],
                        point_pair1[1]*projMatr_Pair1[2,:] - projMatr_Pair1[1,:],
                        projMatr_Pair1[0,:] - point_pair1[0]*projMatr_Pair1[2,:],
                        ]

                    A = np.array(A) 
                    B = A.transpose() @ A
                    U, s, Vh = linalg.svd(B, full_matrices = False)
                    
                    points4D_homogeneous = Vh[-1,0:3]/Vh[-1,3]
                    points_3d_list.append(points4D_homogeneous[:3].T)

                return points_3d_list

            def DLT_3_cameras(imgpoints_Origin, imgpoints_Pair1, imgpoints_Pair2, projMatr_Origin, projMatr_Pair1, projMatr_Pair2):
                points_3d_list = []
                for points_origin, points_pair1, points_pair2 in zip(imgpoints_Origin, imgpoints_Pair1, imgpoints_Pair2):
                    point_origin = np.array(points_origin)
                    point_pair1 = np.array(points_pair1)
                    point_pair2 = np.array(points_pair2)

                    A = [point_origin[1]*projMatr_Origin[2,:] - projMatr_Origin[1,:],
                        projMatr_Origin[0,:] - point_origin[0]*projMatr_Origin[2,:],
                        point_pair1[1]*projMatr_Pair1[2,:] - projMatr_Pair1[1,:],
                        projMatr_Pair1[0,:] - point_pair1[0]*projMatr_Pair1[2,:],
                        point_pair2[1]*projMatr_Pair2[2,:] - projMatr_Pair2[1,:],
                        projMatr_Pair2[0,:] - point_pair2[0]*projMatr_Pair2[2,:]
                        ]

                    A = np.array(A) 
                    B = A.transpose() @ A
                    U, s, Vh = linalg.svd(B, full_matrices = False)
                    
                    points4D_homogeneous = Vh[-1,0:3]/Vh[-1,3]
                    points_3d_list.append(points4D_homogeneous[:3].T)

                return points_3d_list
            
            all_cam_points = [
                markers[map_nums[subject]][f'C{main}']['points'],
                markers[map_nums[subject]][f'C{pairs[0]}']['points'],
                markers[map_nums[subject]][f'C{pairs[1]}']['points']
            ]

            all_results = []

            pair_idx = pairs.index(pair)

            if method == 3:
                all_results.append(DLT_3_cameras(all_cam_points[0], all_cam_points[1], all_cam_points[2], Ps[0], Ps[1], Ps[2]))
            else:
                all_results.append(DLT_2_cameras(all_cam_points[0], all_cam_points[pair_idx+1], Ps[0], Ps[pair_idx+1]))
            
            if pair_idx == 1:
                R2_initial = R3_initial.copy()
                T2_initial = T3_initial.copy()
                cameraMatrix2 = cameraMatrix3.copy()

            points_initial = (all_results[0].copy())
            points_initial = np.array(points_initial)
            points = np.array(points_initial)

            if rot_method == 1:
                x_direction = points[1] - points[0] 
                z_direction = points[3] - points[1] 
            else:
                x_direction = points[2] - points[1] 
                z_direction = points[4] - points[1] 
            x_direction /= np.linalg.norm(x_direction)
            z_direction /= np.linalg.norm(z_direction)
            y_direction = np.cross(x_direction, z_direction)

            R_wcs = np.column_stack((x_direction, y_direction, z_direction))
            M = np.column_stack((R_wcs, points[1]))
            M = np.row_stack((M, np.array([0, 0, 0, 1])))
            M_inv = np.linalg.inv(M)

            rotation_matrix = np.column_stack((x_direction, y_direction, z_direction))

            R_initial = rotation_matrix.copy()

            def projectInternal(M, xyz):
                xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
                internal_xyz = M @ xyz
                internal_xyz[:3].flatten()
                internal_xyz = np.squeeze(internal_xyz)
                internal_xyz =np.squeeze(internal_xyz)
                return internal_xyz[:3]
            
            # calculate C1 internal 3D points
            c1_wand = np.array([projectInternal(M, p) for p in wcs_wand])
            c1_skel = np.array([projectInternal(M, p) for p in wcs_skel])

            order = [1, 0, 2, 3, 4] 
            c1_wand = c1_wand[order, :]
            diff = points - c1_wand
            y_avg_wand = np.mean(c1_wand[:, 1])
            y_avg_points = np.mean(points[:, 1])
            y_diff = y_avg_points - y_avg_wand
            x_avg_wand = np.mean(c1_wand[:, 0])
            x_avg_points = np.mean(points[:, 0])
            x_diff = x_avg_points - x_avg_wand
            z_avg_wand = np.mean(c1_wand[:, 2])
            z_avg_points = np.mean(points[:, 2])
            z_diff = z_avg_points - z_avg_wand
            c1_wand[:, 1] += y_diff
            c1_skel[:, 1] += y_diff
            c1_wand[:, 0] += x_diff
            c1_skel[:, 0] += x_diff
            c1_wand[:, 2] += z_diff
            c1_skel[:, 2] += z_diff


            def projectPixel(cam_params, xyz):
                K = cam_params['mtx']
                a = np.array([[0, 0, 0]]).T
                xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
                M = np.hstack((K, a))
                image_xy = M @ xyz
                image_xy[:3].flatten()
                image_xy /= image_xy[2]
                image_xy = np.squeeze(image_xy)
                return image_xy[0:2]

            cam_params={}
            cam_params['C2']={}
            cam_params['C2']['R'] = R1_initial
            cam_params['C2']['T'] = T1_initial
            cam_params['C2']['mtx'] = cameraMatrix1

            # calculate C1 pixel points
            c1_pixel_points = np.array([projectPixel(cam_params['C2'], p) for p in points])
            c1_pixel_skel = np.array([projectPixel(cam_params['C2'], p) for p in c1_skel])

            if cam == main:
                try:
                    if main == 1:
                        frame_num_w_drop = stomp_sync_init[f'C1{combintation}']['first_idx_w_drop'] + cnt
                        frame_info = filtered_df_c1[filtered_df_c1['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                    else:
                        frame_num_w_drop = stomp_sync_init[f'C6{combintation}']['first_idx_w_drop'] + cnt
                        frame_info = filtered_df_c6[filtered_df_c6['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                except Exception as e:
                    print(e)
                    continue

                frame_num = frame_info['frame_num']
                path = frame_info['path']
                factor = frame_info['factor']
                dup_num = frame_info['dup_num']

                #catch if any frame_num, etc are None:
                if frame_num is None or path is None or factor is None or dup_num is None:
                    print(f'frame_num: {frame_num}, path: {path}, factor: {factor}, dup_num: {dup_num}')

                if main == 1:
                    yolo_data = c1_yolo_data
                else:
                    yolo_data = c6_yolo_data
                try:
                    yolo_box = yolo_data[frame_num.astype(int)]['boxes'][0][0]
                    x1, y1, x2, y2 = yolo_box[:4]
                    w = x2 - x1
                    h = y2 - y1
                    bbox = [x1, y1, w, h]
                except:
                    bbox = None

                data_dict = {
                    'frame_num_w_drop': frame_num_w_drop,
                    'frame_num': frame_num.astype(int),
                    'path': path,
                    'factor': factor.astype(int),
                    'dup_num': dup_num.astype(int),
                    'point_ids': point_ids,
                    'xyz': c1_skel.tolist(),
                    'xy': c1_pixel_skel.tolist(),
                    'wand_xyz': points.tolist(),
                    'wand_xy': c1_pixel_points.tolist(),
                    'bbx': bbox                                         
                }

                if main == 1:
                    c1_data.append(data_dict)
                else:
                    c6_data.append(data_dict)
            
            else:

                def projectInternal(cam_params, xyz):
                    R = cam_params['R']
                    t = cam_params['T']
                    a = np.array([[0, 0, 0]])
                    b = np.array([[1]])
                    xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
                    M = np.vstack((np.hstack((R, t)), np.hstack((a, b))))
                    internal_xyz = M @ xyz
                    internal_xyz[:3].flatten()
                    internal_xyz = np.squeeze(internal_xyz)
                    internal_xyz =np.squeeze(internal_xyz)
                    return internal_xyz[:3]
                    # return xyz
                    
                cam_params={}
                cam_params['C2']={}
                cam_params['C2']['R'] = R2_initial
                cam_params['C2']['T'] = T2_initial
                cam_params['C2']['mtx'] = cameraMatrix2

                # calculate C2 internal 3D points
                c2_3d_points = np.array([projectInternal(cam_params['C2'], p) for p in points])
                c2_3d_skel = np.array([projectInternal(cam_params['C2'], p) for p in c1_skel])   
                
                # calculate C2 pixel points
                c2_pixel_points = np.array([projectPixel(cam_params['C2'], p) for p in c2_3d_points])
                c2_pixel_skel = np.array([projectPixel(cam_params['C2'], p) for p in c2_3d_skel]) 

                try:
                    if pair == 2:
                        frame_num_w_drop = stomp_sync_init[f'C2{combintation}']['first_idx_w_drop'] + cnt
                        frame_info = filtered_df_c2[filtered_df_c2['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                    elif pair == 3:
                        frame_num_w_drop = stomp_sync_init[f'C3{combintation}']['first_idx_w_drop'] + cnt
                        frame_info = filtered_df_c3[filtered_df_c3['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                    elif pair == 4:
                        frame_num_w_drop = stomp_sync_init[f'C4{combintation}']['first_idx_w_drop'] + cnt
                        frame_info = filtered_df_c4[filtered_df_c4['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                    else:
                        frame_num_w_drop = stomp_sync_init[f'C5{combintation}']['first_idx_w_drop'] + cnt
                        frame_info = filtered_df_c5[filtered_df_c5['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                except Exception as e:
                    print(e)
                    continue

                frame_num = frame_info['frame_num']
                path = frame_info['path']
                factor = frame_info['factor']
                dup_num = frame_info['dup_num']

                if pair == 2:
                    yolo_data = c2_yolo_data
                elif pair == 3:
                    yolo_data = c3_yolo_data
                elif pair == 4:
                    yolo_data = c4_yolo_data
                else:
                    yolo_data = c5_yolo_data
                try:
                    yolo_box = yolo_data[frame_num.astype(int)]['boxes'][0][0]
                    x1, y1, x2, y2 = yolo_box[:4]
                    w = x2 - x1
                    h = y2 - y1
                    bbox = [x1, y1, w, h]
                except:
                    bbox = None

                data_dict = {
                    'frame_num_w_drop': frame_num_w_drop,
                    'frame_num': frame_num.astype(int),
                    'path': path,
                    'factor': factor.astype(int),
                    'dup_num': dup_num.astype(int),
                    'point_ids': point_ids,
                    'xyz': c2_3d_skel.tolist(),
                    'xy': c2_pixel_skel.tolist(),
                    'wand_xyz': c2_3d_points.tolist(),
                    'wand_xy': c2_pixel_points.tolist(),
                    'bbx': bbox                                            
                }
 
                if pair == 2:
                    c2_data.append(data_dict)
                elif pair == 3:
                    c3_data.append(data_dict)
                elif pair == 4:
                    c4_data.append(data_dict)
                else:
                    c5_data.append(data_dict)

    def convert_int64(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_int64(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_int64(i) for i in obj]
        else:
            return obj

    #save data dont amke dirs

    c1_data = convert_int64(c1_data)
    path = f'{root_path}internal_data/C1/{subject}/{combintation}.json'
    os.makedirs(f'{root_path}internal_data/C1/{subject}', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(c1_data, f, indent=4)
    c2_data = convert_int64(c2_data)
    path = f'{root_path}internal_data/C2/{subject}/{combintation}.json'
    os.makedirs(f'{root_path}internal_data/C2/{subject}', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(c2_data, f, indent=4)    
    c3_data = convert_int64(c3_data)
    path = f'{root_path}internal_data/C3/{subject}/{combintation}.json'
    os.makedirs(f'{root_path}internal_data/C3/{subject}', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(c3_data, f, indent=4)
    c4_data = convert_int64(c4_data)
    path = f'{root_path}internal_data/C4/{subject}/{combintation}.json'
    os.makedirs(f'{root_path}internal_data/C4/{subject}', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(c4_data, f, indent=4) 
    c5_data = convert_int64(c5_data)                    
    path = f'{root_path}internal_data/C5/{subject}/{combintation}.json'
    os.makedirs(f'{root_path}internal_data/C5/{subject}', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(c5_data, f, indent=4)
    c6_data = convert_int64(c6_data)
    path = f'{root_path}internal_data/C6/{subject}/{combintation}.json'
    os.makedirs(f'{root_path}internal_data/C6/{subject}', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(c6_data, f, indent=4)