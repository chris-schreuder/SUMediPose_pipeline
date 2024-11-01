from glob import glob
import json
import pandas as pd
from fastprogress import progress_bar

root_path = 'path/to/your/root/directory/'

files = glob(f'{root_path}data/data_yolo/frames/C4/S*/*.json')


data = []

for file in progress_bar(files):
    if 'A0' in file:
        continue
    if 'S1A1D1' in file:
        continue
    with open(file, 'r') as f:
        frames = json.load(f)
        frames = sorted(frames, key=lambda x: int(x['file'].split('_')[-1].split('.')[0]))
    right_foot_points = []
    prev = 0
    for frame in frames:

        if len(frame["keypoints"][0][0]) != 17:
            right_foot_points.append(prev)
        elif frame['num_people'] == 0:
            right_foot_points.append(prev)
        elif frame['keypoints'][0][0][16][2] > 0.8:
            right_foot_points.append(frame['keypoints'][0][0][16][1])
            prev = frame['keypoints'][0][0][16][1]
        else:
            right_foot_points.append(prev)
    avg_first_10 = sum(right_foot_points[:10])/10
    diffs = [abs(y-avg_first_10) for y in right_foot_points]

    low = 5
    high_thres = 30
    flag_high = False
    first_idx = None
    last_idx = None
    for diff in diffs:
        if diff > high_thres:
            flag_high = True
        if flag_high & (diff < low):
            first_idx = diffs.index(diff)
            break
    # get last foot down point
    avg_first_10 = sum(right_foot_points[-10:])/10
    diffs = [abs(y-avg_first_10) for y in right_foot_points]
    last_high_idx = None
    for diff in diffs:
        if diff > high_thres:
            last_high_idx = diffs.index(diff)
    for diff in diffs[last_high_idx:]:
        if diff < low:
            last_idx = diffs.index(diff)
            break

    avg_first_10 = sum(right_foot_points[:10])/10
    diffs = [abs(y-avg_first_10) for y in right_foot_points]

    path = f'{root_path}data/graphs/{file.split("/")[-1].replace(".json", "")}.png'
    data.append({'combintation': file.split("/")[-1].replace(".json", ""), 'first_idx': first_idx, 'last_idx': last_idx, 'path': path})
    #plot diffs over index
    import matplotlib.pyplot as plt
    plt.plot(diffs)
    # add title
    plt.title(f'{file.split("/")[-1].replace(".json", "")}: {first_idx} - {last_idx}')
    if first_idx is not None:
        plt.axvline(first_idx, color='r')
    if last_idx is not None:
        plt.axvline(last_idx, color='r')
    # save graph
    plt.savefig(path)
    plt.close()
    # plt.show()

df = pd.DataFrame(data)
df.to_csv(f'{root_path}data/stomps_C4.csv', index=False)