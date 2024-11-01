import numpy as np
import cv2
from glob import glob
from fastprogress import progress_bar
import os
import pandas as pd
import random
from tqdm import tqdm

root_path = 'path/to/data/'

os.makedirs(f'{root_path}calibration/internal/matrix_finetune/', exist_ok=True)
os.makedirs(f'{root_path}calibration/internal/csvs/', exist_ok=True)

folders = glob('calibration/internal/frames/**/**')

data = []

pbar = tqdm(total=4*6*3*1000, desc='Internal Calibration: ')

for folder in progress_bar(folders):
    print(f'Processing {folder.split("/")[-2]}-{folder.split("/")[-1]}')

    cam = folder.split('/')[-2][1]
    subject = folder.split('/')[-1].replace('S', '')

    checkerboard_size = (8,6)  
    world_scaling = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = world_scaling* objp

    objpoints = []  
    imgpoints_A = [] 
    images_A = glob(f'{folder}/*.jpg')
    images_A = images_A[150:]

    for img_path_A in progress_bar(images_A, total=len(images_A)):
        img_A = cv2.imread(img_path_A)
        gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)

        ret_A, corners_A = cv2.findChessboardCorners(gray_A, checkerboard_size, None)

        if ret_A:
            objpoints.append(objp)
            corners2_A = cv2.cornerSubPix(gray_A, corners_A, (11,11), (-1,-1), criteria=criteria)
            imgpoints_A.append(corners2_A)

    # print(len(imgpoints_A))
    # sample_rate = 40
    # selected_objpoints = objpoints[::sample_rate]
    # selected_imgpoints_A = imgpoints_A[::sample_rate]
    for num in [15, 20, 25]:
        for rep in range(1, 1000):
            random_indices = random.sample(range(len(imgpoints_A)), num)
            selected_objpoints = [objpoints[i] for i in random_indices]
            selected_imgpoints_A = [imgpoints_A[i] for i in random_indices]
            ret_A, mtx_A, dist_A, rvecs_A, tvecs_A = cv2.calibrateCamera(selected_objpoints, selected_imgpoints_A, gray_A.shape[::-1], None, None)
            path = f'{root_path}calibration/internal/matrix_finetune/C{cam}_S{subject}_internal_calibration_{num}_{rep}.npz'
            np.savez(path, mint=mtx_A, dist=dist_A)
            data.append({'cam': cam, 'subject': subject, 'ret': ret_A, 'path': path, 'num': num, 'repeat': rep})
            pbar.update(1)

df = pd.DataFrame(data)
df.to_csv(f'{root_path}calibration/internal/csvs/internal_calibration_finetune.csv', index=False)

