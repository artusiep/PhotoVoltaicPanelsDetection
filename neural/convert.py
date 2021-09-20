import glob
import json
import os
from pathlib import Path

import PIL.Image
import cv2
import labelme
import numpy as np

DATA_DIR = '../experiments/data/thermal-panels'
OUT_DIR = './data'
class_names = []
class_name_to_id = {'0': 1}


def create_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


Path(OUT_DIR, 'input').mkdir(parents=True, exist_ok=True)
Path(OUT_DIR, 'ground_truth').mkdir(parents=True, exist_ok=True)

for label_file in sorted(glob.glob(os.path.join(DATA_DIR, '*.json'))):
    with open(label_file) as f:
        base = os.path.splitext(os.path.basename(label_file))[0]
        data = json.load(f)
        img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])
        img = np.asarray(PIL.Image.open(img_file))
        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data['shapes'],
            label_name_to_value=class_name_to_id,
        )
        ground_truth = np.copy(lbl)[0]

        ground_truth = (ground_truth * 255).astype(np.uint8)
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR)

        PIL.Image.fromarray(img).save(os.path.join(OUT_DIR, 'input', base + '.png'))
        cv2.imwrite(os.path.join(OUT_DIR, 'ground_truth', base + '.png'), ground_truth)
