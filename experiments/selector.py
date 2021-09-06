import glob
import os
import random
from shutil import copyfile

path = 'data/raw'
# https://en.wikipedia.org/wiki/Reservoir_sampling
files = glob.glob("../data/raw/*.JPG")

directory = './data/raw'
percentage = 0.10


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


result_files = random_subset(files, int(percentage * (len(files))))

if not os.path.exists(directory):
    os.makedirs(directory)

file_list = glob.glob(os.path.join(directory, "*"))
for f in file_list:
    os.remove(f)

for source in result_files:
    file_name = os.path.basename(source)
    copyfile(source, f"{directory}/{file_name}")
