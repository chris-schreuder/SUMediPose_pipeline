import numpy as np
import cv2
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
from fastprogress import progress_bar
import json
from glob import glob
import os
from tqdm import tqdm

root_path = 'path/to/your/data/'  # Change this to the path of your data

with open(f'{root_path}data/stomp_sync_corrected.json', 'r') as f:
    stomp_sync_init = json.load(f)

df_drops = pd.read_csv(f'{root_path}data/times/all_times_w_drops.csv')

df_cam_matrix = pd.read_csv(f'{root_path}calibration/external/csvs/external_calibrations_w_errors_best.csv')
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

files = glob(f'{root_path}WCS/S9/S9A1D2.json')


pbar = tqdm(total=len(files)*9)

for file in progress_bar(files):

    
    subject = file.split('/')[-2]
    combintation = file.split('/')[-1].replace('.json', '')
    action = f"A{combintation.split('A')[1].split('D')[0]}"
    print(f'Processing {subject} {combintation}')

    path = f'{root_path}internal_data/C6/{subject}/{combintation}.json'

    filtered_df_c1 = df_drops[df_drops['combination'] == f'C1{combintation}']
    filtered_df_c2 = df_drops[df_drops['combination'] == f'C2{combintation}']
    filtered_df_c3 = df_drops[df_drops['combination'] == f'C3{combintation}']
    filtered_df_c4 = df_drops[df_drops['combination'] == f'C4{combintation}']
    filtered_df_c5 = df_drops[df_drops['combination'] == f'C5{combintation}']
    filtered_df_c6 = df_drops[df_drops['combination'] == f'C6{combintation}']

    for offset in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
    # for offset in [-1, 0, 1]:
        c1_data = []
        c2_data = []
        c3_data = []
        c4_data = []
        c5_data = []
        c6_data = []

        with open(file, 'r') as f:
            frames = json.load(f)
        for cnt, frame in enumerate(frames):
            if cnt > 130 and cnt < len(frames) - 130:
                continue
            xyz = frame['xyz']
            xyz_wand = frame['xyz_wand']
            point_ids = frame['point_ids']
            floor_idx = point_ids.index('CentreOfMassFloor')
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

            for main in [1, 6]:

                if main == 1:
                    pairs = [2, 3]
                else:
                    pairs = [4, 5]

                df_temp = df_cam_matrix[(df_cam_matrix['main'] == main) & (df_cam_matrix['pair'] == pairs[0]) & (df_cam_matrix['subject'] == map_matrix[subject])].copy()
                path1 = df_temp['path'].values[0]
                df_temp = df_cam_matrix[(df_cam_matrix['main'] == main) & (df_cam_matrix['pair'] == pairs[1]) & (df_cam_matrix['subject'] == map_matrix[subject])].copy()
                path2 = df_temp['path'].values[0]
                data = np.load(path1)
                data2 = np.load(path2)

                R1 = np.eye(3)  
                T1 = np.array([0, 0, 0])  
                R2 = data['R']
                T2 = data['T']
                T2 = np.array([T2[0][0], T2[1][0], T2[2][0]]).reshape((3,1))
                R3 = data2['R']
                T3 = data2['T']
                T3 = np.array([T3[0][0], T3[1][0], T3[2][0]]).reshape((3,1))

                cameraMatrix1 = data['mtx_A']
                cameraMatrix2 = data['mtx_B']
                cameraMatrix3 = data2['mtx_B']

                RT1 = np.concatenate([R1, [[0],[0],[0]]], axis = -1)
                P1 = cameraMatrix1 @ RT1
                RT2 = np.concatenate([R2, T2], axis = -1)
                P2 = cameraMatrix2 @ RT2
                RT3 = np.concatenate([R3, T3], axis = -1)
                P3 = cameraMatrix3 @ RT3

                Ps = [P1, P2, P3]

                def triangulate_multiple_points(imgpoints_Origin, imgpoints_Pair1, imgpoints_Pair2, projMatr_Origin, projMatr_Pair1, projMatr_Pair2):
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
                            projMatr_Pair2[0,:] - point_pair2[0]*projMatr_Pair2[2,:],
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

                all_results.append(triangulate_multiple_points(all_cam_points[0], all_cam_points[1], all_cam_points[2], Ps[0], Ps[1], Ps[2]))
                R1_initial = R1.copy()
                T1_initial = T1.reshape((3,1)).copy()
                R2_initial = R2.copy()
                T2_initial = T2.reshape((3,1)).copy()
                R3_initial = R3.copy()
                T3_initial = T3.reshape((3,1)).copy()
                points_initial = (all_results[0].copy())
                points_initial = np.array(points_initial)
                points = np.array(points_initial)

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

                try:
                    if main == 1:
                        frame_num_w_drop = stomp_sync_init[f'C1{combintation}']['first_idx_w_drop'] + cnt + offset
                        frame_info = filtered_df_c1[filtered_df_c1['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                    else:
                        frame_num_w_drop = stomp_sync_init[f'C6{combintation}']['first_idx_w_drop'] + cnt  + offset
                        frame_info = filtered_df_c6[filtered_df_c6['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                except Exception as e:
                    print(f'Error: {subject} - {combintation} - {e}')
                    continue

                frame_num = frame_info['frame_num']
                path = frame_info['path']
                factor = frame_info['factor']
                dup_num = frame_info['dup_num']

                #catch if any frame_num, etc are None:
                if frame_num is None or path is None or factor is None or dup_num is None:
                    print(f'frame_num: {frame_num}, path: {path}, factor: {factor}, dup_num: {dup_num}')

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
                    'wand_xy': c1_pixel_points.tolist()                                            
                }
                
                if main == 1:
                    c1_data.append(data_dict)
                else:
                    c6_data.append(data_dict)

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
                    if main == 1:
                        frame_num_w_drop = stomp_sync_init[f'C2{combintation}']['first_idx_w_drop'] + cnt + offset
                        frame_info = filtered_df_c2[filtered_df_c2['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                    else:
                        frame_num_w_drop = stomp_sync_init[f'C4{combintation}']['first_idx_w_drop'] + cnt + offset
                        frame_info = filtered_df_c4[filtered_df_c4['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                except Exception as e:
                    print(f'Error: {subject} - {combintation} - {e}')
                    continue

                frame_num = frame_info['frame_num']
                path = frame_info['path']
                factor = frame_info['factor']
                dup_num = frame_info['dup_num']

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
                    'wand_xy': c2_pixel_points.tolist()                                              
                }
               
                if main == 1:
                    c2_data.append(data_dict)
                else:
                    c4_data.append(data_dict)

                cam_params={}
                cam_params['C2']={}
                cam_params['C2']['R'] = R3_initial
                cam_params['C2']['T'] = T3_initial
                cam_params['C2']['mtx'] = cameraMatrix3

                # calculate C3 internal 3D points
                c2_3d_points = np.array([projectInternal(cam_params['C2'], p) for p in points])
                c2_3d_skel = np.array([projectInternal(cam_params['C2'], p) for p in c1_skel])  

                # calculate C3 pixel points
                c2_pixel_points = np.array([projectPixel(cam_params['C2'], p) for p in c2_3d_points])
                c2_pixel_skel = np.array([projectPixel(cam_params['C2'], p) for p in c2_3d_skel])

                try:
                    if main == 1:
                        frame_num_w_drop = stomp_sync_init[f'C3{combintation}']['first_idx_w_drop'] + cnt + offset
                        frame_info = filtered_df_c3[filtered_df_c3['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                    else:
                        frame_num_w_drop = stomp_sync_init[f'C5{combintation}']['first_idx_w_drop'] + cnt + offset
                        frame_info = filtered_df_c5[filtered_df_c5['frame_num_w_drop'] == frame_num_w_drop].iloc[0]
                except Exception as e:
                    print(f'Error: {subject} - {combintation} - {e}')
                    continue

                frame_num = frame_info['frame_num']
                path = frame_info['path']
                factor = frame_info['factor']
                dup_num = frame_info['dup_num']

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
                    'wand_xy': c2_pixel_points.tolist()                                            
                }

                if main == 1:
                    c3_data.append(data_dict)
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
        c2_data = convert_int64(c2_data)
        c3_data = convert_int64(c3_data)
        c5_data = convert_int64(c5_data)                    
        c6_data = convert_int64(c6_data)

        def gen_video(frames, cam, subject, combination):
                    # Check if there are any frames
            if not frames:
                print(f"No frames for {subject} {cam} {combination}")
                return

            # Get the path of the first image
            first_frame_path = frames[0]['path']
            first_frame_path = f'{root_path}{first_frame_path}'

            # Read the first image to get its shape
            first_img = cv2.imread(first_frame_path)
            if first_img is None:
                raise ValueError(f"Error loading the first image: {first_frame_path}")

            # Get image shape (height, width, channels)
            frame_height, frame_width, _ = first_img.shape
            frame_size = (frame_width, frame_height)

            fps = 15  # Frames per second
            # video_filename = f'{root_path}videos_2p/{subject}/{cam}/{combination}.mp4'
            video_filename = f'{root_path}videos_sync_final/{subject}/{cam}/{combination}/{combination}_{offset}.mp4'
            os.makedirs(os.path.dirname(video_filename), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 file
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
            count = 0
            for frame in frames:
                path = frame['path']
                path = f'{root_path}{path}'
                xy = frame['xy']
                factor = frame['factor']
                dup_num = frame['dup_num']
                if count == 0:
                    prev_xy = xy
                    prev_path = path
                    count += 1
                if factor != dup_num:
                    path = prev_path
                    xy = prev_xy
                    # continue
                else:
                    prev_xy = xy
                    prev_path = path        
                img = cv2.imread(path)
                if img is None:
                    print(f"Error loading image: {path}")
                    continue
                img = cv2.resize(img, frame_size)
                for (x, y) in xy:
                    try:
                        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circle with radius 5
                    except:
                        continue
                video_writer.write(img)
            video_writer.release()

        gen_video(c1_data, 'C1', subject, combintation)
        gen_video(c2_data, 'C2', subject, combintation)
        gen_video(c3_data, 'C3', subject, combintation)
        gen_video(c5_data, 'C5', subject, combintation)
        gen_video(c6_data, 'C6', subject, combintation)

        pbar.update(1)
        