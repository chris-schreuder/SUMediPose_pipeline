from glob import glob
import os
import json
from fastprogress import progress_bar
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

root_path = 'path/to/data/'

base_options = python.BaseOptions(model_asset_path='hands/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

folders = glob(f'{root_path}frames/C*/S*/**')
to_folders = [folder.replace(root_path, f'{root_path}data_hands') for folder in folders]
to_dict = [f'{folder}.json' for folder in to_folders]

for from_folder, to_file in progress_bar(zip(folders, to_dict), total=len(folders)):
    print(f'Processing {from_folder}')
    os.makedirs(os.path.dirname(to_file), exist_ok=True)
    files = glob(f'{from_folder}/*.jpg')
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    folder_data = []
    for file in progress_bar(files):
        
        image = mp.Image.create_from_file(file)
        w, h = image.width, image.height
        detection_result = detector.detect(image)
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            name = handedness[0].category_name

            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            hand_right_xy = []
            hand_left_xy = []
            if name == 'Right':
                hand_right_xy = [[x, y] for x, y in zip(x_coordinates, y_coordinates)]
            else:
                hand_left_xy = [[x, y] for x, y in zip(x_coordinates, y_coordinates)]

        file_data = {}
        file_data['file'] = file.replace(root_path, '')
        file_data['width'] = w
        file_data['height'] = h
        file_data['hand_right'] = hand_right_xy
        file_data['hand_left'] = hand_left_xy

        folder_data.append(file_data)

    with open(to_file, 'w') as f:
        json.dump(folder_data, f, indent=4)
