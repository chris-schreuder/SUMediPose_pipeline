# from glob import glob
# from fastprogress import progress_bar
# from tqdm import tqdm
# import os
# import cv2

# files = glob('/media/chris/T7/data/**/*.h264')
# txts = glob('/media/chris/T7/data/**/*.txt')

# total = 0
# for txt in txts:
#     if 'rec_times' in txt:
#         continue
#     if 'calibration' in txt:
#         continue
#     with open(txt, 'r') as f:
#         lines = f.readlines()
#         total += len(lines)

# pbar = tqdm(total=total, desc='Extracting Frames: ')
# # files = files[:1]
# for file in files:
#     if 'calibration' in file:
#         continue
#     combintation = file.split('/')[-1].replace('.h264', '')
#     camera = combintation.split('S')[0]
#     subject = combintation.split('S')[-1].split('A')[0]
#     action = combintation.split('A')[-1].split('D')[0]
#     speed = combintation.split('D')[-1]
#     id = combintation
#     video_file = file
#     output_folder = f'/media/chris/T7/data_processed/frames/{camera}/S{subject}/{combintation}/'
#     os.makedirs(output_folder, exist_ok=True)
#     cap = cv2.VideoCapture(video_file)
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         frame_filename = os.path.join(output_folder, f'{id}_{str(frame_count).zfill(4)}.jpg')
#         cv2.imwrite(frame_filename, rotated_frame)
#         frame_count += 1
#         pbar.update(1)
#     cap.release()
#     print(f'Extracted {frame_count} frames to {output_folder}')

# import os
# import shutil
# from tqdm import tqdm
# from glob import glob
# from concurrent.futures import ThreadPoolExecutor

# source_dir = '/media/chris/T7/data_processed'
# destination_dir = '/media/chris/01DAAF12036713D0/data_processed'

# # Ensure the source directory exists
# if not os.path.exists(source_dir):
#     print(f"Source directory '{source_dir}' does not exist.")
#     exit()

# # Ensure the destination directory exists, create if it doesn't
# if not os.path.exists(destination_dir):
#     os.makedirs(destination_dir)
#     print(f"Created destination directory '{destination_dir}'.")

# files = glob('/media/chris/T7/data_processed/frames/**/**/**/*.jpg')
# pbar = tqdm(total=len(files), desc='Copying Files: ')

# def copy_file(file):
#     path_from = file
#     path_to = file.replace(source_dir, destination_dir)
#     os.makedirs(os.path.dirname(path_to), exist_ok=True)
#     shutil.copy2(path_from, path_to)
#     pbar.update(1)

# # Use ThreadPoolExecutor to copy files in parallel
# with ThreadPoolExecutor() as executor:
#     executor.map(copy_file, files)


# import os
# import zipfile
# from glob import glob
# from tqdm import tqdm

# # for i in range(1, 7):
# for i in [6]:

#     files = glob(f'/media/chris/01DAAF12036713D0/data_processed/frames/C{i}b/**/**/*.jpg')
#     pbar = tqdm(total=len(files), desc='Copying Files: ')

#     def zip_directory(directory_path, zip_path):
#         with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#             for root, dirs, files in os.walk(directory_path):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     arcname = os.path.relpath(file_path, start=directory_path)
#                     zipf.write(file_path, arcname)
#                     pbar.update(1)

#     directory_to_zip = f'/media/chris/01DAAF12036713D0/data_processed/frames/C{i}b'
#     zip_file_path = f'/media/chris/01DAAF12036713D0/data_processed/frames/C{i}b.zip'

#     zip_directory(directory_to_zip, zip_file_path)

# import os
# import zipfile
# from glob import glob
# from tqdm import tqdm

# # Function to zip a list of files
# def zip_files(file_list, zip_path, pbar):
#     with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for file in file_list:
#             zipf.write(file, os.path.relpath(file, start=os.path.dirname(file_list[0])))
#             pbar.update(1)

# for i in [6]:  # Adjust this range as needed
#     # Get the list of all jpg files in the directory
#     files = glob(f'/media/chris/01DAAF12036713D0/data_processed/frames/C{i}/**/**/*.jpg', recursive=True)
    
#     # Calculate the midpoint
#     midpoint = len(files) // 2
    
#     # Split the files into two halves
#     first_half = files[:midpoint]
#     second_half = files[midpoint:]
    
#     # Create progress bar
#     pbar = tqdm(total=len(files), desc=f'Copying Files for C{i}: ')

#     # Paths for the zip files
#     zip_file_path_a = f'/media/chris/01DAAF12036713D0/data_processed/frames/C{i}_a.zip'
#     zip_file_path_b = f'/media/chris/01DAAF12036713D0/data_processed/frames/C{i}_b.zip'
    
#     # Zip the first half
#     zip_files(first_half, zip_file_path_a, pbar)
    
#     # Zip the second half
#     zip_files(second_half, zip_file_path_b, pbar)

#     pbar.close()

#     print(f"Files from C{i} have been zipped into '{zip_file_path_a}' and '{zip_file_path_b}'")


# from glob import glob
# from fastprogress import progress_bar
# from tqdm import tqdm
# import os
# import cv2

# files = glob('/home/chris/Desktop/data/**/*_calibration.h264')


# # files = files[:1]
# for file in progress_bar(files):

#     combintation = file.split('/')[-1].replace('_calibration.h264', '')
#     camera = combintation.split('S')[0]
#     subject = combintation.split('S')[-1].split('A')[0]
#     action = combintation.split('A')[-1].split('D')[0]
#     speed = combintation.split('D')[-1]
#     id = subject+'_calibration'
#     video_file = file
#     output_folder = f'calibration/internal/frames/{camera}/S{subject}/'
#     os.makedirs(output_folder, exist_ok=True)
#     cap = cv2.VideoCapture(video_file)
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         frame_filename = os.path.join(output_folder, f'{id}_{str(frame_count).zfill(4)}.jpg')
#         cv2.imwrite(frame_filename, rotated_frame)
#         frame_count += 1
#     cap.release()
#     print(f'Extracted {frame_count} frames to {output_folder}')

# from glob import glob
# from fastprogress import progress_bar
# from tqdm import tqdm
# import os
# import cv2
# import shutil

# folders = glob('/home/chris/Desktop/data/**/*_calibration')
# # '/home/chris/Desktop/data/C1S6/C1S6A1D1_grouped_G12_calibration/C1S6A1D1_grouped_G12_calibration_0001.jpg'

# for folder in progress_bar(folders):
#     print(f'Processing {folder.split("/")[-1]}')
#     files = glob(f'{folder}/*.jpg')
#     camera = folder.split('/')[-2][1]
#     subject = folder.split('/')[-2].split('S')[-1]
#     camera_main = folder.split('/')[-1].split('_')[-2][1]
#     camera_pair = folder.split('/')[-1].split('_')[-2][2]
#     output_folder = f'calibration/external/frames/S{subject}/C{camera}S{subject}_G{camera_main}{camera_pair}/'
#     os.makedirs(output_folder, exist_ok=True)
#     for file in progress_bar(files):
#         frame_num = file.split('/')[-1].split('_')[-1].replace('.jpg', '')
#         frame_filename = f'C{camera}S{subject}_G{camera_main}{camera_pair}_{frame_num}.jpg'
#         shutil.copy2(file, os.path.join(output_folder, frame_filename))
        
