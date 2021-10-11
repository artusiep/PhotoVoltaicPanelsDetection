import glob
import os
import random
from shutil import copyfile

from detector.extractor.extractor import ThermalImageExtractor

path = 'data/raw'
# https://en.wikipedia.org/wiki/Reservoir_sampling
files = glob.glob("../data/raw/*.JPG")

already_selected_file_names = [os.path.basename(x).replace('plasma-', '') for x in glob.glob('./data/thermal-panels/*.JPG')]
files = [x for x in files if os.path.basename(x) not in already_selected_file_names]

raw_output_directory = './data/raw-2'
thermal_output_directory = './data/thermal-panels'


def random_subset(iterator, K):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len(result) < K:
            result.append(item)
        else:
            s = int(random.random() * N)
            if s < K:
                result[s] = item

    return result


result_files = random_subset(files, 100)

if not os.path.exists(raw_output_directory):
    os.makedirs(raw_output_directory)

file_list = glob.glob(os.path.join(raw_output_directory, "*"))
for f in file_list:
    os.remove(f)

for source in result_files:
    file_name = os.path.basename(source)
    copyfile(source, f"{raw_output_directory}/{file_name}")

raw_files = glob.glob(f"{raw_output_directory}/*.JPG")
thermal_files = [ThermalImageExtractor.get_thermal_image_file_path(file, ['plasma'], thermal_output_directory) for file in raw_files]
