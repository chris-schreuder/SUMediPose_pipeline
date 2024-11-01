import os
import pandas as pd
from fastprogress import progress_bar
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

root_path = 'path/to/data/'
csv_path = f'{root_path}SUMediPose.csv'
df = pd.read_csv(csv_path)
df = df[df['subject'] == 'S28'].copy()
df = df[df['cam'].isin(['C1', 'C2', 'C3'])].copy()

# Get the unique paths
paths = df['path'].unique()
root_path_to = 'path/to/copy/to/'
paths_from = [os.path.join(root_path, path) for path in paths]
paths_to = [os.path.join(root_path_to, path) for path in paths]

# Function to copy files
def copy_file(path_from, path_to):
    os.makedirs(os.path.dirname(path_to), exist_ok=True)
    shutil.copy(path_from, path_to)

# Function to track progress and copy files in parallel
def parallel_copy(paths_from, paths_to, max_workers=8):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(copy_file, path_from, path_to) for path_from, path_to in zip(paths_from, paths_to)]
        
        # Use progress_bar to track progress as files are copied
        for _ in progress_bar(as_completed(futures), total=len(futures)):
            pass

# Perform the parallel file copying
print(f'Copying {len(paths)} files...')
parallel_copy(paths_from, paths_to)
