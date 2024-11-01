import dvuploader as dv
from glob import glob
from fastprogress import progress_bar

root_path = 'path/to/data/'

DV_URL = "https://dataverse.harvard.edu"
API_TOKEN = 'API_TOKEN'
PID = 'DOI_IDENTIFIER'

# Get all zip files
all_files = glob(f'{root_path}frames/C*/S*/*.zip')
len_files = len(all_files)

# Function to chunk files into groups of 3
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# Process the files in batches of 3
for batch in progress_bar(list(chunks(all_files, 3))):
    files = []
    for file in batch:
        cam = file.split('/')[-3]
        subject = file.split('/')[-2]
        files.append(dv.File(mimeType='application/zip', directory_label=f"frames/{cam}/{subject}", filepath=file))
    try:
        # Create uploader and upload files in the current batch
        dvuploader = dv.DVUploader(files=files)
        dvuploader.upload(
            api_token=API_TOKEN,
            dataverse_url=DV_URL,
            persistent_id=PID,
            n_parallel_uploads=3,  # Adjust based on your instance capacity
        )
    except Exception as e:
        print(f"Error uploading files: {e}")
        with open('failed_files.txt', 'a') as f:
            f.write('\n'.join(batch) + '\n')
        continue

# # Get all zip files
# all_files = glob(f'{root_path}calibration/internal/frames/C*/*.zip')
# len_files = len(all_files)

# # Function to chunk files into groups of 3
# def chunks(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i:i+n]

# # Process the files in batches of 3
# for batch in progress_bar(list(chunks(all_files, 3))):
#     files = []
#     for file in batch:
#         cal_type = file.split('/')[-4]
#         cam = file.split('/')[-2]
#         files.append(dv.File(mimeType='application/zip', directory_label=f"calibration/{cal_type}/frames/{cam}", filepath=file))
#     try:
#         # Create uploader and upload files in the current batch
#         dvuploader = dv.DVUploader(files=files)
#         dvuploader.upload(
#             api_token=API_TOKEN,
#             dataverse_url=DV_URL,
#             persistent_id=PID,
#             n_parallel_uploads=3,  # Adjust based on your instance capacity
#         )
#     except Exception as e:
#         print(f"Error uploading files: {e}")
#         with open('failed_files.txt', 'a') as f:
#             f.write('\n'.join(batch) + '\n')
#         continue

#-------------------------------------------------------------------------------------------------------

# # Get all zip files
# all_files = glob(f'{root_path}calibration/external/matrix/*.npz')
# len_files = len(all_files)

# # Function to chunk files into groups of 3
# def chunks(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i:i+n]

# # Process the files in batches of 3
# for batch in progress_bar(list(chunks(all_files, 3))):
#     files = []
#     for file in batch:
#         cal_type = file.split('/')[-3]
#         files.append(dv.File(mimeType='application/zip', directory_label=f"calibration/external/matrix", filepath=file))
#     try:
#         # Create uploader and upload files in the current batch
#         dvuploader = dv.DVUploader(files=files)
#         dvuploader.upload(
#             api_token=API_TOKEN,
#             dataverse_url=DV_URL,
#             persistent_id=PID,
#             n_parallel_uploads=3,  # Adjust based on your instance capacity
#         )
#     except Exception as e:
#         print(f"Error uploading files: {e}")
#         with open('failed_files.txt', 'a') as f:
#             f.write('\n'.join(batch) + '\n')
#         continue


#----------------------------------------------------------------------------------------

# # Get all zip files
# all_files = glob(f'{root_path}calibration/external_calibration.csv')
# len_files = len(all_files)

# # Function to chunk files into groups of 3
# def chunks(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i:i+n]

# # Process the files in batches of 3
# for batch in progress_bar(list(chunks(all_files, 3))):
#     files = []
#     for file in batch:
#         cal_type = file.split('/')[-4]
#         cam = file.split('/')[-2]
#         files.append(dv.File(mimeType='text/csv', directory_label=f"calibration", filepath=file))
#     try:
#         # Create uploader and upload files in the current batch
#         dvuploader = dv.DVUploader(files=files)
#         dvuploader.upload(
#             api_token=API_TOKEN,
#             dataverse_url=DV_URL,
#             persistent_id=PID,
#             n_parallel_uploads=3,  # Adjust based on your instance capacity
#         )
#     except Exception as e:
#         print(f"Error uploading files: {e}")
#         with open('failed_files.txt', 'a') as f:
#             f.write('\n'.join(batch) + '\n')
#         continue

#----------------------------------------------------------------------------------------

# # Get all zip files
# all_files = glob(f'{root_path}internal_data/C*/S*.zip')
# len_files = len(all_files)

# # Function to chunk files into groups of 3
# def chunks(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i:i+n]

# # Process the files in batches of 3
# for batch in progress_bar(list(chunks(all_files, 3))):
#     files = []
#     for file in batch:
#         cam = file.split('/')[-2]
#         subject = file.split('/')[-1].split('.')[0]

#         files.append(dv.File(mimeType='application/zip', directory_label=f"internal_data/{cam}", filepath=file))
#     try:
#         # Create uploader and upload files in the current batch
#         dvuploader = dv.DVUploader(files=files)
#         dvuploader.upload(
#             api_token=API_TOKEN,
#             dataverse_url=DV_URL,
#             persistent_id=PID,
#             n_parallel_uploads=3,  # Adjust based on your instance capacity
#         )
#     except Exception as e:
#         print(f"Error uploading files: {e}")
#         with open('failed_files.txt', 'a') as f:
#             f.write('\n'.join(batch) + '\n')
#         continue

