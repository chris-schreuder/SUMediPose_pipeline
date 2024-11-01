from glob import glob
import cv2
import numpy as np
from fastprogress import progress_bar
import os
import pandas as pd
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json

root_path = 'path/to/data/'

df_intrinsic = pd.read_csv(f'{root_path}calibration/internal/csvs/internal_calibration.csv')
matrix_subjects = df_intrinsic['subject'].unique().tolist()
matrix_subjects = sorted(matrix_subjects)
map_matrix = {}
cur_num = 6
for i in range(1,29):
    if i in matrix_subjects:
        cur_num = i
    map_matrix[f'S{i}'] = cur_num
map_nums = map_matrix

# os.makedirs('calibration/external/matrix/', exist_ok=True)
os.makedirs(f'{root_path}calibration/external/matrix/', exist_ok=True)
os.makedirs(f'{root_path}calibration/external/csvs/', exist_ok=True)

def stereo_cal(objpoints, imgpoints_A, imgpoints_B, valid_frames_A, valid_frames_B, file_num, gray_A_shape, pair, main, repeat, main_intrinsic_path, pair_intrinsic_path, subject):

    if len(imgpoints_A) >= file_num:
        random_indices = random.sample(range(len(imgpoints_A)), file_num)
    else:
        random_indices = random.sample(range(len(imgpoints_A)), len(imgpoints_A))

    objpoints_sample = [objpoints[i] for i in random_indices]
    imgpoints_A_sample = [imgpoints_A[i] for i in random_indices]
    imgpoints_B_sample = [imgpoints_B[i] for i in random_indices]
    valid_frames_A_sample = [valid_frames_A[i] for i in random_indices]
    valid_frames_B_sample = [valid_frames_B[i] for i in random_indices]

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
    path = f'{root_path}calibration/external/matrix/S{subject}_G{main}{pair}_external_calibration_repeat_{repeat}_{file_num}_frames.npz'
    np.savez(path, R=R, T=T, min1=cameraMatrix1, dist1=distCoeffs1, mint2=cameraMatrix2, dist2=distCoeffs2, mtx_A=mtx_A, dist_A=dist_A, mtx_B=mtx_B, dist_B=dist_B)
    pbar.update(1)

    return True, {'subject': subject, 'main': main, 'pair': pair, 'repeat': repeat, 'num_frames': file_num, 'valid_A': len(imgpoints_A_sample), 'valid_B': len(imgpoints_B_sample), 'retval': retval, 'valid_frames_A': valid_frames_A_sample, 'valid_frames_B': valid_frames_B_sample, 'path': path}


subjects = glob(f'{root_path}calibration/external/frames/S*')
subjects = [subject.split('/')[-1] for subject in subjects]

data = []

pbar = tqdm(total=(2*2*5*2500*len(subjects)), desc='External Calibration: ')

# df_intrinsic = pd.read_csv('calibration/internal/csvs/internal_calibration_best.csv')
df_intrinsic = pd.read_csv(f'{root_path}calibration/internal/csvs/internal_calibration.csv')

for subject in subjects:
    for main in [1,6]:
        # main_intrinsic_path = df_intrinsic[(df_intrinsic['cam'] == main)]['path'].values[0]
        main_intrinsic_path = df_intrinsic[(df_intrinsic['cam'] == main) & (df_intrinsic['subject'] == map_nums[subject])]['path'].values[0]
        if main == 1:
            pairs = [2,3]
        else:
            pairs = [4,5]
        for pair in pairs:
            # pair_intrinsic_path = df_intrinsic[(df_intrinsic['cam'] == pair)]['path'].values[0]
            pair_intrinsic_path = df_intrinsic[(df_intrinsic['cam'] == pair) & (df_intrinsic['subject'] == map_nums[subject])]['path'].values[0]

            valid_frames_A = []
            valid_frames_B = []

            checkerboard_size = (8,6)  # Change this to the actual size of your checkerboard
            world_scaling = 1
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

            objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
            objp = world_scaling* objp

            # Arrays to store object points and image points from all the images.
            objpoints = []  # 3d point in real world space
            imgpoints_A = []  # 2d points in image plane for Camera A.
            imgpoints_B = []  # 2d points in image plane for Camera B.

            images_A = glob(f'{root_path}calibration/external/frames/{subject}/C{main}{subject}_G{main}{pair}/*.jpg')
            images_A = sorted(images_A, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            images_B = glob(f'{root_path}calibration/external/frames/{subject}/C{pair}{subject}_G{main}{pair}/*.jpg')
            images_B = sorted(images_B, key=lambda x: int(x.split('_')[-1].split('.')[0]))

            # Assuming you have the same number of images for both cameras
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
            
            for file_num in range(5, 11):
                with ThreadPoolExecutor() as executor:
                    for repeat in range(2500):
                        result = executor.submit(stereo_cal, objpoints, imgpoints_A, imgpoints_B, valid_frames_A, valid_frames_B, file_num, gray_A.shape[::-1], pair, main, repeat, main_intrinsic_path, pair_intrinsic_path, subject)
                        flag, res = result.result()
                        data.append(res)

df = pd.DataFrame(data)
# df.to_csv(f'calibration/external/csvs/external_calibrations.csv', index=False)
df.to_csv(f'{root_path}calibration/external/csvs/external_calibrations_full.csv', index=False)



