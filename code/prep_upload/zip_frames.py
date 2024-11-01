import os
import zipfile
from glob import glob
from fastprogress import progress_bar
from concurrent.futures import ThreadPoolExecutor, as_completed

root_path = 'path/to/data/'

# Root path for folders
folders = glob(f'{root_path}frames/C*/S*/C*')
print(len(folders))

# Function to create a zip file for each folder
def zip_folder(folder_path):
    zip_file_path = f'{folder_path}.zip'

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in progress_bar(files):
                file_path = os.path.join(root, file)
                # Write file to the zip, using a relative path (relative to the folder)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

    print(f"Zipped folder: {folder_path} -> {zip_file_path}")

def parallel_zip(folders):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(zip_folder, folder) for folder in folders]
        
        # Use progress_bar to track progress as files are copied
        for _ in progress_bar(as_completed(futures), total=len(futures)):
            pass

# Perform the parallel file copying
print(f'Zipping {len(folders)} files...')
parallel_zip(folders)


