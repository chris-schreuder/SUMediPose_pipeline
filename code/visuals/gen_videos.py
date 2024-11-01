import cv2
import json
import numpy as np
from fastprogress import progress_bar
from glob import glob
import os

root_path = 'path/to/your/data/'  # Change this to the path of your data

files = glob(f'{root_path}internal_data/**/**/*.json')

for file in progress_bar(files):
    cam = file.split('/')[-3]
    subject = file.split('/')[-2]
    combination = file.split('/')[-1].split('.')[0]

    print(f"Processing: {cam} - {subject} - {combination}")

    # Load the frames from the JSON file
    with open(file, 'r') as f:
        frames = json.load(f)

    # Check if there are any frames
    if not frames:
        print(f"No frames found in the JSON file.")
        continue

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

    # Define the video writer
    fps = 15  # Frames per second
    video_filename = f'{root_path}videos/{subject}/{cam}/{combination}.mp4'
   
    os.makedirs(os.path.dirname(video_filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 file
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
    count = 0
    for frame in progress_bar(frames):
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
        # Read the image
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading image: {path}")
            continue
        
        # Resize image if needed (to match frame_size)
        img = cv2.resize(img, frame_size)

        # Plot the xy values on the image
        for (x, y) in xy:
            try:
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circle with radius 5
            except:
                continue

        # Write the frame to the video file
        video_writer.write(img)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {video_filename}")
