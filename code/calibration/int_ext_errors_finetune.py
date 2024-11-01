import pandas as pd
import json
import numpy as np
from fastprogress import progress_bar
import os
from scipy import linalg
import cv2

root_path = 'path/to/data/'

# df = pd.read_csv('calibration/external/csvs/external_calibrations_finetune.csv')
df = pd.read_csv(f'{root_path}calibration/external/csvs/external_calibrations_finetune_markers.csv')

with open(f'{root_path}data/stomp_sync_refine_manual.json', 'r') as f:
    stomp_sync_refine_manual = json.load(f)

with open(f'{root_path}data/stomp_3d_manual.json', 'r') as f:
    stomp_3d_manual = json.load(f)

with open(f'{root_path}data/markers_finetune.json', 'r') as f:
    markers_finetune = json.load(f)
    

# data_dict = {}

keys = list(markers_finetune.keys())
subject_nums = [int(key.replace('S', '')) for key in keys if 'points' in markers_finetune[key]['C1']]
subject_nums = sorted(subject_nums)
map_nums = {}
cur_num = 1
for i in range(1,29):
    if i in subject_nums:
        cur_num = i
    map_nums[f'S{i}'] = f'S{cur_num}'
map_finetune = map_nums

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

errors = []
errors_pair = []
perc_errors = []
perc_errors_pair = []
errs_skel = []
errs_skel_pair = []
idxs = []

# df = df.head(2)

for idx, (index, row) in progress_bar(enumerate(df.iterrows()), total=len(df)):
    main = row['main']
    pair = row['pair']
    path_int_A = row['path_int_A']
    path_int_B = row['path_int_B']
    ret_A = row['ret_A']
    ret_B = row['ret_B']
    path = row['path']
    subject = row['subject']

    try:
        c4_start_idx = stomp_3d_manual[f'C4{map_finetune[subject]}A1D1']['first_idx']
    except:
        print(f'C4{subject}A1D1 not in stomp_3d_manual')
        continue

    path_3d = f'{root_path}WCS/{map_finetune[subject]}/{map_finetune[subject]}A1D1.json'
    with open(path_3d, 'r') as f:
        data_3d = json.load(f)

    frame = data_3d[c4_start_idx]
    xyz = frame['xyz']
    xyz_wand = frame['xyz_wand']
    point_ids = frame['point_ids']
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

    cal_size = 60

    data = np.load(path)

    R1 = np.eye(3)  
    T1 = np.array([0, 0, 0])  
    R2 = data['R']
    T2 = data['T']
    T2 = np.array([T2[0][0], T2[1][0], T2[2][0]]).reshape((3,1))

    cameraMatrix1 = data['mtx_A']
    cameraMatrix2 = data['mtx_B']

    RT1 = np.concatenate([R1, [[0],[0],[0]]], axis = -1)
    P1 = cameraMatrix1 @ RT1
    RT2 = np.concatenate([R2, T2], axis = -1)
    P2 = cameraMatrix2 @ RT2

    Ps = [P1, P2]

    def triangulate_multiple_points(imgpoints_Origin, imgpoints_Pair1, projMatr_Origin, projMatr_Pair1):
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

    
    all_cam_points = [
        markers[map_nums[subject]][f'C{main}']['points'],
        markers[map_nums[subject]][f'C{pair}']['points']
    ]

    all_results = []

    all_results.append(triangulate_multiple_points(all_cam_points[0], all_cam_points[1], Ps[0], Ps[1]))
    R1_initial = R1.copy()
    T1_initial = T1.reshape((3,1)).copy()
    R2_initial = R2.copy()
    T2_initial = T2.reshape((3,1)).copy()
    points_initial = (all_results[0].copy())

    points = np.array(points_initial)

    points_initial = np.array(points_initial)
    points = np.array(points_initial)

    x_direction = points[1] - points[0] 
    z_direction = points[3] - points[1]  
    x_direction /= np.linalg.norm(x_direction)
    z_direction /= np.linalg.norm(z_direction)
    y_direction = np.cross(x_direction, z_direction)
    # x_direction = np.cross(y_direction, z_direction)

    R_wcs = np.column_stack((x_direction, y_direction, z_direction))
    M = np.column_stack((R_wcs, points[1]))
    M = np.row_stack((M, np.array([0, 0, 0, 1])))
    M_inv = np.linalg.inv(M)

    rotation_matrix = np.column_stack((x_direction, y_direction, z_direction))

    R_initial = rotation_matrix.copy()

    points = np.array(points)
    Mint_1 = data['mtx_A']
    # Mint_1 = data['min1']
    Mint_1 = np.hstack((Mint_1, np.zeros((3, 1))))

    def projectInternal(M, xyz):
        xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
        internal_xyz = M @ xyz
        internal_xyz[:3].flatten()
        internal_xyz = np.squeeze(internal_xyz)
        internal_xyz =np.squeeze(internal_xyz)
        return internal_xyz[:3]
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

    def projectPixel(M, xyz):
        xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
        image_xy = M @ xyz
        image_xy[:3].flatten()
        image_xy /= image_xy[2]
        image_xy = np.squeeze(image_xy)
        return image_xy[0:2]

    points_2d = []
    for point in points:
        points_2d.append(projectPixel(Mint_1, point))

    points_c1_2d = np.array(points_2d)
    all_cam_points[0] = np.array(all_cam_points[0])
    #get mean error
    squared_errors = np.sum((points_c1_2d - all_cam_points[0])**2, axis=1)

    # Calculate the mean error
    mean_error = np.sqrt(np.mean(squared_errors))
    percentage_error = (mean_error / np.sqrt(1600*1200)) * 100

    # Calculate the mean error
    errors.append(mean_error)
    perc_errors.append(percentage_error)

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
    c1_pixel_skel = np.array([projectPixel(cam_params['C2'], p) for p in c1_skel])
    c1_pixel_wand = np.array([projectPixel(cam_params['C2'], p) for p in c1_wand])
    temp_ids = markers_finetune[map_finetune[subject]][f'C{main}']['ids']
    temp_points = markers_finetune[map_finetune[subject]][f'C{main}']['points']

    new_points = []
    for id in temp_ids:
        id_in_point_ids = point_ids.index(id)
        new_points.append(c1_pixel_skel[id_in_point_ids])
    new_points = np.array(new_points)
    err_skel = np.sum((new_points - temp_points)**2, axis=1)
    mean_error_skel = np.sqrt(np.mean(err_skel))
    errs_skel.append(mean_error_skel)


    #<------------------ project to c2 ------------------>

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

    def projectPixel(cam_params, xyz):
        K = cam_params['mtx']
        a = np.array([[0, 0, 0]]).T
        xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
        # combine matrices into a new matrix
        M = np.hstack((K, a))
        image_xy = M @ xyz
        image_xy[:3].flatten()
        image_xy /= image_xy[2]
        image_xy = np.squeeze(image_xy)
        return image_xy[0:2]
        
    cam_params={}
    cam_params['C2']={}
    cam_params['C2']['R'] = R2_initial
    cam_params['C2']['T'] = T2_initial
    cam_params['C2']['mtx'] = cameraMatrix2

    points = np.array(points_initial)

    points_internal_c2 = np.array([projectInternal(cam_params['C2'], p) for p in points])
    points_pixel_c2 = np.array([projectPixel(cam_params['C2'], p) for p in points_internal_c2])

    points = np.array(all_cam_points[1])
    squared_errors = np.sum((points_pixel_c2 - points)**2, axis=1)
    mean_error = np.sqrt(np.mean(squared_errors))
    percentage_error = (mean_error / np.sqrt(1600*1200)) * 100
    errors_pair.append(mean_error)
    perc_errors_pair.append(percentage_error)

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
    cam_params={}
    cam_params['C2']={}
    cam_params['C2']['R'] = R2_initial
    cam_params['C2']['T'] = T2_initial
    cam_params['C2']['mtx'] = cameraMatrix2
    c2_3d_skel = np.array([projectInternal(cam_params['C2'], p) for p in c1_skel])   
    c2_pixel_skel = np.array([projectPixel(cam_params['C2'], p) for p in c2_3d_skel]) 
    c2_3d_wand = np.array([projectInternal(cam_params['C2'], p) for p in c1_wand])
    c2_pixel_wand = np.array([projectPixel(cam_params['C2'], p) for p in c2_3d_wand])
    temp_ids = markers_finetune[map_finetune[subject]][f'C{pair}']['ids']
    temp_points = markers_finetune[map_finetune[subject]][f'C{pair}']['points']
    new_points = []
    for id in temp_ids:
        id_in_point_ids = point_ids.index(id)
        new_points.append(c2_pixel_skel[id_in_point_ids])
    new_points = np.array(new_points)
    err_skel = np.sum((new_points - temp_points)**2, axis=1)
    mean_error_skel = np.sqrt(np.mean(err_skel))
    errs_skel_pair.append(mean_error_skel)

df['error'] = errors
df['perc_error'] = perc_errors
df['error_pair'] = errors_pair
df['perc_error_pair'] = perc_errors_pair
df['error_skel'] = errs_skel
df['error_skel_pair'] = errs_skel_pair
# df['idx'] = idxs


df.to_csv(f'{root_path}calibration/external/csvs/external_calibrations_w_errors_finetune_markers.csv', index=False)

idx = df.groupby(['subject', 'main', 'pair'])['error_skel'].idxmin()
filtered_df = df.loc[idx]
filtered_df.to_csv(f'{root_path}calibration/external/csvs/external_calibrations_w_errors_finetune_best_main_markers.csv', index=False)

idx = df.groupby(['subject', 'main', 'pair'])['error_skel_pair'].idxmin()
filtered_df = df.loc[idx]
filtered_df.to_csv(f'{root_path}calibration/external/csvs/external_calibrations_w_errors_finetune_best_pair_markers.csv', index=False)