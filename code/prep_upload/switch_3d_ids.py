import json
import os
from glob import glob
from fastprogress import progress_bar

root_path = 'path/to/data/'

ids_map = {
    "C7": "C7",
    "CLAV": "CLAV",
    "CentreOfMass": "CentreOfMass",
    "CentreOfMassFloor": "CentreOfMassFloor",
    "LAJC": "LAJC",
    "LANK": "LTOE",
    "LASI": "LPSI",
    "LBHD": "LBHD",
    "LEJC": "LEJC",
    "LELB": "LFRM",
    "LFHD": "LFHD",
    "LFIN": "RSHO",
    "LFRM": "LWRA",
    "LHEE": "RTHI",
    "LHJC": "LHJC",
    "LKJC": "LKJC",
    "LKNE": "LANK",
    "LMMED": "R_Foot_Out",
    "LPSI": "LTHI",
    "LSHO": "LSHO",
    "LSJC": "LSJC",
    "LTHI": "LTIB",
    "LTIB": "LHEE",
    "LTOE": "RKNE",
    "LUPA": "LELB",
    "LWJC": "LWJC",
    "LWRA": "LWRB",
    "LWRB": "LFIN",
    "L_Foot_Out": "AAAAAAAA",
    "MElbowL": "MKNEL",
    "MElbowR": "MKNER",
    "MKNEL": "LMMED",
    "MKNER": "RMMED",
    "PelL": "MElbowL",
    "PelR": "MElbowR",
    "RAJC": "RAJC",
    "RANK": "RTOE",
    "RASI": "RPSI",
    "RBAK": "RBAK",
    "RBHD": "RBHD",
    "REJC": "REJC",
    "RELB": "RWRA",
    "RFHD": "RFHD",
    "RFIN": "RASI",
    "RFRM": "RWRB",
    "RHEE": "PelR",
    "RHJC": "RHJC",
    "RKJC": "RKJC",
    "RKNE": "RANK",
    "RMMED": "L_Foot_Out",
    "RPSI": "LKNE",
    "RSHO": "RELB",
    "RSJC": "RSJC",
    "RTHI": "RTIB",
    "RTIB": "RHEE",
    "RTOE": "PelL",
    "RUPA": "RFRM",
    "RWJC": "RWJC",
    "RWRA": "RFIN",
    "RWRB": "LASI",
    "R_Foot_Out": "BBBBBBB",
    "STRN": "STRN",
    "T10": "T10"
}

files = glob(f'{root_path}WCS/S*/*.json')

for file in progress_bar(files):
    with open(file, 'r') as f:
        frames = json.load(f)

    for frame in frames:
        point_ids = frame['point_ids']
        temp_ids = [ids_map[id] for id in point_ids]
        frame['point_ids'] = temp_ids
    path_to = file
    os.makedirs(os.path.dirname(path_to), exist_ok=True)
    with open(path_to, 'w') as f:
        json.dump(frames, f)
