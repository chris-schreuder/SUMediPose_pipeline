import pandas as pd
import numpy as np
import json
from glob import glob
import os
from fastprogress import progress_bar

root_path = 'path/to/your/data/'

files = glob(f'{root_path}data/raw_3d_data/**/Wand*.csv')

subjects_num = [int(file.split('/')[-2][1:]) for file in files]
subjects_num = sorted(subjects_num)
map_nums = {}
cur_num = 16
for i in range(1,29):
    if i in subjects_num:
        cur_num = i
    map_nums[i] = cur_num

map_nums_3d = map_nums.copy()

def get_wand_data(subject):
    subject = map_nums_3d[subject]
    with open(f'{root_path}data/raw_3d_data/P{str(subject).zfill(2)}/Wand_01.csv') as f:
        lines = f.readlines()
        points = lines[2]

    points = points.split('\t')
    names = [point.split(':')[-1] for point in points if point not in ['', '\n']]

    cols = ['frame', 'sub']
    for name in names:
        # name = name.split(':')[-1]
        cols.append(name + '_X')
        cols.append(name + '_Y')
        cols.append(name + '_Z')

    skip_lines = 5
    df = pd.read_csv(f'{root_path}data/raw_3d_data/P{str(subject).zfill(2)}/Wand_01.csv', delimiter='\t+', engine='python', skiprows=skip_lines, names=cols)
    df = df.iloc[:, 2:]

    new_df = pd.DataFrame(columns=['point_id', 'frame', 'X', 'Y', 'Z'])
    for col in df.columns:
        point_id, coord = col.rsplit('_', 1)
        frame = df.index.astype(str)
        
        temp_df = pd.DataFrame({
            'point_id': point_id,
            'frame': frame,
            coord: df[col]
        })
        new_df = pd.concat([new_df, temp_df], axis=0, ignore_index=True)

    new_df = new_df.groupby(['point_id', 'frame']).agg({'X': 'first', 'Y': 'first', 'Z': 'first'}).reset_index()
    new_df['frame'] = new_df['frame'].astype(int)
    #sort by frame
    new_df = new_df.sort_values(by=['frame'])

    new_df['point_id'] = new_df['point_id'].str.replace('P0:', '').replace('Lize:', '')

    # new_df.rename(columns={'Y': 'Z', 'Z': 'Y'}, inplace=True)

    # new_df['X'] = -new_df['X'].astype(float)
    new_df['X'] = new_df['X'].astype(float)
    new_df['Y'] = new_df['Y'].astype(float)
    new_df['Z'] = new_df['Z'].astype(float)

    new_df = new_df.reset_index(drop=True)
    #calcl mean X Y Z
    df_temp = new_df.groupby('point_id').agg({'X': 'mean', 'Y': 'mean', 'Z': 'mean'}).reset_index()

    x = df_temp['X'].tolist()
    y = df_temp['Y'].tolist()
    z = df_temp['Z'].tolist()
    # make numpy matrix with xyz
    xyz_wand = np.array([x, z, y]).T
    xyz_wand = xyz_wand.tolist()
    return xyz_wand

action_map = {
    'Push_up': '7',
    'Squat_Jumps': '8',
    'Pushups': '7',
    'Squat': '3',
    'Deadlift': '4',
    'Lunges': '5',
    'Deadlifts': '4',
    'Should_Press': '6',
    'Walk': '1',
    # 'Dynamic': ,
    'Soulder_Press': '6',
    'Lunge': '5',
    'Shoulder_Press': '6',
    'Star_Jump': '9',
    # 'Wand': ,
    'Star_Jumps': '9',
    'Squat_jumps': '8',
    'Pushup': '7',
    'Squat_Jump': '8',
    'Squats': '3'
}

files  = glob(f'{root_path}data/raw_3d_data/P*/*.csv')
for file in progress_bar(files):
    if 'Dynamic' in file or 'Wand' in file:
        continue
    traj_path = file
    jc_path = file.replace('.csv', '.txt')
    subject = int(file.split('/')[-2].replace('P', ''))
    if subject == 1:
        continue
    action = file.split('/')[-1].rsplit('_', 1)[0]
    action = action_map[action]
    duration = int(file.split('/')[-1].rsplit('_', 1)[1].replace('.csv', ''))

    xyz_wand = get_wand_data(subject)

    with open(traj_path) as f:
        lines = f.readlines()
        points = lines[2]

    points = points.split('\t')
    names = [point.split(':')[-1] for point in points if point not in ['', '\n']]
    
    cols = ['frame', 'sub']
    for name in names:
        # name = name.split(':')[-1]
        cols.append(name + '_X')
        cols.append(name + '_Y')
        cols.append(name + '_Z')

    skip_lines = 5
    df = pd.read_csv(traj_path, delimiter='\t+', engine='python', skiprows=skip_lines, names=cols)
    df = df.iloc[:, 2:]

    new_df = pd.DataFrame(columns=['point_id', 'frame', 'X', 'Y', 'Z'])
    for col in df.columns:
        point_id, coord = col.rsplit('_', 1)
        frame = df.index.astype(str)
        
        temp_df = pd.DataFrame({
            'point_id': point_id,
            'frame': frame,
            coord: df[col]
        })
        new_df = pd.concat([new_df, temp_df], axis=0, ignore_index=True)

    new_df = new_df.groupby(['point_id', 'frame']).agg({'X': 'first', 'Y': 'first', 'Z': 'first'}).reset_index()
    new_df['frame'] = new_df['frame'].astype(int)
    #sort by frame
    new_df = new_df.sort_values(by=['frame'])

    new_df['point_id'] = new_df['point_id'].str.replace('P0:', '').replace('Lize:', '')

    # new_df.rename(columns={'Y': 'Z', 'Z': 'Y'}, inplace=True)

    # new_df['X'] = -new_df['X'].astype(float)
    new_df['X'] = new_df['X'].astype(float)
    new_df['Y'] = new_df['Y'].astype(float)
    new_df['Z'] = new_df['Z'].astype(float)

    new_df = new_df.reset_index(drop=True)
    df1 = new_df.copy()

    with open(jc_path) as f:
        lines = f.readlines()
        points = lines[2]

    points = points.split('\t')
    names = [point.split(':')[-1] for point in points if point not in ['', '\n']]
    
    cols = ['frame', 'sub']
    for name in names:
        # name = name.split(':')[-1]
        cols.append(name + '_X')
        cols.append(name + '_Y')
        cols.append(name + '_Z')

    skip_lines = 5
    df = pd.read_csv(jc_path, delimiter='\t+', engine='python', skiprows=skip_lines, names=cols)
    df = df.iloc[:, 2:]

    new_df = pd.DataFrame(columns=['point_id', 'frame', 'X', 'Y', 'Z'])
    for col in df.columns:
        point_id, coord = col.rsplit('_', 1)
        frame = df.index.astype(str)
        
        temp_df = pd.DataFrame({
            'point_id': point_id,
            'frame': frame,
            coord: df[col]
        })
        new_df = pd.concat([new_df, temp_df], axis=0, ignore_index=True)

    new_df = new_df.groupby(['point_id', 'frame']).agg({'X': 'first', 'Y': 'first', 'Z': 'first'}).reset_index()
    new_df['frame'] = new_df['frame'].astype(int)
    #sort by frame
    new_df = new_df.sort_values(by=['frame'])

    new_df['point_id'] = new_df['point_id'].str.replace('P0:', '').replace('Lize:', '')

    # new_df.rename(columns={'Y': 'Z', 'Z': 'Y'}, inplace=True)

    # new_df['X'] = -new_df['X'].astype(float)
    new_df['X'] = new_df['X'].astype(float)
    new_df['Y'] = new_df['Y'].astype(float)
    new_df['Z'] = new_df['Z'].astype(float)

    new_df = new_df.reset_index(drop=True)
    df2 = new_df.copy()

    df_3d = pd.concat([df1, df2], axis=0, ignore_index=True)
    df_3d = df_3d.sort_values(by=['frame', 'point_id'])

    frame_data = []
    for frame in df_3d['frame'].unique():
        df_temp = df_3d[df_3d['frame'] == frame].copy()
        point_ids = df_temp['point_id'].astype(str).tolist() 
        x = df_temp['X'].tolist()
        y = df_temp['Y'].tolist()
        z = df_temp['Z'].tolist()
        # make numpy matrix with xyz
        xyz = np.array([x, z, y]).T
        xyz_list = xyz.tolist()
        frame_dict = {
            'frame_num': int(frame),
            'point_ids': point_ids,
            'xyz': xyz_list,
            'xyz_wand': xyz_wand
        }
        frame_data.append(frame_dict)

    os.makedirs(f'{root_path}WCS/S{subject}', exist_ok=True)

    with open(f'{root_path}WCS/S{subject}/S{subject}A{action}D{duration}.json', 'w') as f:
        json.dump(frame_data, f, indent=4)