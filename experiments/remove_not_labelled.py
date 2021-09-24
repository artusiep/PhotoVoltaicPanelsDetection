import glob
import os

jpg_files = glob.glob('./data/thermal-panels/*.JPG')
json_files = glob.glob('./data/thermal-panels/*.json')

jpg_file_names = [os.path.basename(x) for x in jpg_files]
json_file_names = [os.path.basename(x).replace('.json', '') for x in json_files]

jpg_file_names = [x for x in jpg_file_names if os.path.basename(x).replace('.JPG', '') in json_file_names]


for f in jpg_files:
    if os.path.basename(f) not in jpg_file_names:
        os.remove(f)
