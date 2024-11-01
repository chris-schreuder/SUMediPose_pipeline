import cv2
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
from fastprogress import progress_bar
import json
import numpy as np

root_path = 'path/to/data/'

df = pd.read_csv(f'{root_path}calibration/external_calibration_full.csv')

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

for idx, (index, row) in progress_bar(enumerate(df.iterrows()), total=len(df)):
    main = row['main']
    pair = row['pair']
    repeat = row['repeat']
    num_frames = row['num_frames']
    path = row['path']
    subject = row['subject']

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
    x_direction = np.cross(y_direction, z_direction)

    R_wcs = np.vstack((x_direction, y_direction, z_direction))
    M = np.vstack((R_wcs, points[1]))
    M = np.hstack((M, np.array([[0],[0],[0],[1]])))
    M = M.T
    M_inv = np.linalg.inv(M)

    rotation_matrix = np.column_stack((x_direction, y_direction, z_direction))

    R_initial = rotation_matrix.copy()

    points = np.array(points)
    Mint_1 = data['mtx_A']
    # Mint_1 = data['min1']
    Mint_1 = np.hstack((Mint_1, np.zeros((3, 1))))

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

df['error'] = errors
df['perc_error'] = perc_errors
df['error_pair'] = errors_pair
df['perc_error_pair'] = perc_errors_pair

df.to_csv(f'{root_path}calibration/external/csvs/external_calibrations_w_errors.csv', index=False)

idx = df.groupby(['subject', 'main', 'pair'])['error'].idxmin()

filtered_df = df.loc[idx]
filtered_df.to_csv(f'{root_path}calibration/external/csvs/external_calibrations_w_errors_best.csv', index=False)