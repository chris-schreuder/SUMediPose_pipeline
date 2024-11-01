import pandas as pd
import ast
import cv2
import numpy as np
from fastprogress import progress_bar
import os
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

root_path = 'path/to/data/'

os.makedirs(f'{root_path}calibration/external/matrix_finetune_markers/', exist_ok=True)
os.makedirs(f'{root_path}calibration/external/csvs/', exist_ok=True)

def stereo_cal(objpoints, imgpoints_A, imgpoints_B, valid_frames_A, valid_frames_B, gray_A_shape, pair, main, main_intrinsic_path, pair_intrinsic_path, subject, main_ret, pair_ret):

    objpoints_sample = objpoints
    imgpoints_A_sample = imgpoints_A
    imgpoints_B_sample = imgpoints_B
    valid_frames_A_sample = valid_frames_A
    valid_frames_B_sample = valid_frames_B

    data_internal1 = np.load(main_intrinsic_path)
    mtx_A = data_internal1['mint']
    dist_A = data_internal1['dist']
    data_internal2 = np.load(pair_intrinsic_path)
    mtx_B = data_internal2['mint']
    dist_B = data_internal2['dist']

    # Stereo calibration
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints_sample, imgpoints_A_sample, imgpoints_B_sample, mtx_A, dist_A, mtx_B, dist_B, gray_A_shape, flags=stereocalibration_flags, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    )
    valid_frames_A_sample = sorted(valid_frames_A_sample, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    valid_frames_B_sample = sorted(valid_frames_B_sample, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # path = f'calibration/external/matrix/S{subject}_G{main}{pair}_external_calibration_repeat_{repeat}_{file_num}_frames.npz'
    path = f'{root_path}calibration/external/matrix_finetune_markers/S{subject}_G{main}{pair}_external_calibration_{main_intrinsic_path.split("/")[-1].split(".")[0]}.npz'
    np.savez(path, R=R, T=T, min1=cameraMatrix1, dist1=distCoeffs1, mint2=cameraMatrix2, dist2=distCoeffs2, mtx_A=mtx_A, dist_A=dist_A, mtx_B=mtx_B, dist_B=dist_B)

    return True, {'subject': subject, 'main': main, 'pair': pair, 'valid_A': len(imgpoints_A_sample), 'valid_B': len(imgpoints_B_sample), 'retval': retval, 'valid_frames_A': valid_frames_A_sample, 'valid_frames_B': valid_frames_B_sample, 'path': path, 'path_int_A': main_intrinsic_path, 'path_int_B': pair_intrinsic_path, 'ret_A': main_ret, 'ret_B': pair_ret}

df_int = pd.read_csv(f'{root_path}calibration/internal/csvs/internal_calibration_finetune.csv')

matrix_subjects = df_int['subject'].unique().tolist()
matrix_subjects = sorted(matrix_subjects)
map_matrix = {}
cur_num = 6
for i in range(1,29):
    if i in matrix_subjects:
        cur_num = i
    map_matrix[f'S{i}'] = cur_num
map_nums = map_matrix

df_intrinsic = pd.read_csv(f'{root_path}calibration/internal/csvs/internal_calibration_best.csv')

df_ext = pd.read_csv(f'{root_path}calibration/external/csvs/external_calibrations_w_errors_w_markers_best_pair.csv')
df_ext = df_ext[df_ext['subject'] == 'S6'].copy()
df_ext = df_ext[df_ext['main'] == 6].copy()

data = []

for idx, (index, row) in progress_bar(enumerate(df_ext.iterrows()), total=len(df_ext)):
    main = row['main']
    pair = row['pair']
    repeat = row['repeat']
    num_frames = row['num_frames']
    path = row['path']
    subject = row['subject']

    valid_frames_A = row['valid_frames_A']
    valid_frames_B = row['valid_frames_B']
    images_A = ast.literal_eval(valid_frames_A)
    images_B = ast.literal_eval(valid_frames_B)

    temp_df_main = df_int[(df_int['cam'] == main) & (df_int['subject']  == map_nums[subject])]
    temp_df_main = temp_df_main.sort_values(by=['ret'], ascending=True)
    temp_df_main = temp_df_main.reset_index(drop=True)
    temp_df_pair = df_int[(df_int['cam'] == pair) & (df_int['subject']  == map_nums[subject])]
    temp_df_pair = temp_df_pair.sort_values(by=['ret'], ascending=True)
    temp_df_pair = temp_df_pair.reset_index(drop=True)

    valid_frames_A = []
    valid_frames_B = []

    checkerboard_size = (8,6)  # Change this to the actual size of your checkerboard
    world_scaling = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = world_scaling* objp

    objpoints = [] 
    imgpoints_A = []  
    imgpoints_B = []  

    for img_path_A, img_path_B in zip(images_A, images_B):
        img_A = cv2.imread(img_path_A)
        img_B = cv2.imread(img_path_B)

        img_A = cv2.rotate(img_A, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_B = cv2.rotate(img_B, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        gray_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_A, corners_A = cv2.findChessboardCorners(gray_A, checkerboard_size, None)
        ret_B, corners_B = cv2.findChessboardCorners(gray_B, checkerboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret_A and ret_B:
            valid_frames_A.append(img_path_A)
            valid_frames_B.append(img_path_B)
            objpoints.append(objp)

            corners2_A = cv2.cornerSubPix(gray_A, corners_A, (11,11), (-1,-1), criteria=criteria)
            imgpoints_A.append(corners2_A)

            corners2_B = cv2.cornerSubPix(gray_B, corners_B, (11,11), (-1,-1), criteria=criteria)
            imgpoints_B.append(corners2_B)

    len_df = len(temp_df_main)
    for i in range(len_df):
        # main_intrinsic_path = df_intrinsic[(df_intrinsic['cam'] == main) & (df_intrinsic['subject'] == map_nums[subject])]['path'].values[0]
        # main_ret = df_intrinsic[(df_intrinsic['cam'] == main) & (df_intrinsic['subject'] == map_nums[subject])]['ret'].values[0]
        main_intrinsic_path = temp_df_main['path'][i]
        main_ret = temp_df_main['ret'][i]
        pair_intrinsic_path = df_intrinsic[(df_intrinsic['cam'] == pair) & (df_intrinsic['subject'] == map_nums[subject])]['path'].values[0]
        pair_ret = df_intrinsic[(df_intrinsic['cam'] == pair) & (df_intrinsic['subject'] == map_nums[subject])]['ret'].values[0]

        with ThreadPoolExecutor() as executor:
            result = executor.submit(stereo_cal, objpoints, imgpoints_A, imgpoints_B, valid_frames_A, valid_frames_B, gray_A.shape[::-1], pair, main, main_intrinsic_path, pair_intrinsic_path, subject, main_ret, pair_ret)
            flag, res = result.result()
            data.append(res)
    
df = pd.DataFrame(data)
df.to_csv(f'{root_path}calibration/external/csvs/external_calibrations_finetune_markers.csv', index=False)




