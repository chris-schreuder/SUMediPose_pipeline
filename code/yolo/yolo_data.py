from ultralytics import YOLO
from glob import glob
import os
import json
from fastprogress import progress_bar

root_path = 'path/to/data/'

model = YOLO("yolov8m-pose.pt")

folders = glob(f'{root_path}frames/C*/S*/**')
# folders = folders[:2]
to_folders = [folder.replace('frames', 'data_yolo') for folder in folders]
to_dict = [f'{folder}.json' for folder in to_folders]

for from_folder, to_file in progress_bar(zip(folders, to_dict), total=len(folders)):
    print(f'Processing {from_folder}')
    os.makedirs(os.path.dirname(to_file), exist_ok=True)
    files = glob(f'{from_folder}/*.jpg')
    #sort files by _XXXXX.jpg
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    folder_data = []
    for file in progress_bar(files):
        results = model(file, verbose=False)
        boxes = []
        keypoints = []
        file_data = {}
        for result in results:
            for det in result.boxes:
                boxes.append(det.data.cpu().numpy().tolist())  # Convert NumPy array to list
            for kp in result.keypoints:
                keypoints.append(kp.data.cpu().numpy().tolist())  # Convert NumPy array to list
        file_data['file'] = file.replace(root_path, '')
        file_data['num_people'] = len(results[0])
        file_data['boxes'] = boxes
        file_data['keypoints'] = keypoints
        folder_data.append(file_data)
    with open(to_file, 'w') as f:
        json.dump(folder_data, f, indent=4)
