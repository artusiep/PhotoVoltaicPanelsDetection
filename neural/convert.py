import argparse
import glob
import json
import os
from pathlib import Path

import PIL.Image
import PIL.ImageDraw
import cv2
import numpy as np


GCS_BUCKET = 'photo-voltaic-panels-detection'
GCS_PATH = f'gs://{GCS_BUCKET}'
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, nargs='?', default='../experiments/data/thermal-panels')
parser.add_argument('--out-dir', type=Path, nargs='?', default='./data')
args = parser.parse_args()


def create_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    draw.polygon(xy=xy, outline=1, fill=1)

    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]

        cls_name = label
        instance = (cls_name)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def create_ground_truth_images(data_dir, out_dir, class_mapping):
    Path(out_dir, 'input').mkdir(parents=True, exist_ok=True)
    Path(out_dir, 'ground_truth').mkdir(parents=True, exist_ok=True)

    labeled_files = glob.glob(os.path.join(data_dir, '*.json'))

    print(f'Started converting polygons to mask images of {len(labeled_files)} files.')
    for label_file in sorted(labeled_files):
        with open(label_file) as f:
            base = os.path.splitext(os.path.basename(label_file))[0]
            data = json.load(f)
            img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])
            shape = data['imageHeight'], data['imageWidth'], 3
            pil_image = PIL.Image.open(img_file)
            lbl = shapes_to_label(
                img_shape=shape,
                shapes=data['shapes'],
                label_name_to_value=class_mapping,
            )
            ground_truth = np.copy(lbl)[0]

            ground_truth = cv2.cvtColor(ground_truth.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            training_input_path = os.path.join(out_dir, 'input', base + '.png')
            ground_truth_path = os.path.join(out_dir, 'ground_truth', base + '.png')
            pil_image.save(training_input_path)
            cv2.imwrite(ground_truth_path, ground_truth)

    print(f"Successfully created {len(labeled_files)} ground truth images ")


create_ground_truth_images(args.data_dir, args.out_dir, {'0': 255})
# with ZipFile(f'data/train-{DATASET_VERSION}.zip', 'w', compression=ZIP_LZMA) as train_zip:
#     for file_path in glob.glob(f'{args.out_dir}/ground_truth/*'):
#         file = Path(file_path).name
#         train_zip.write(f'{args.out_dir}/ground_truth/{file}')
#         train_zip.write(f'{args.out_dir}/input/{file}')
#
#
# subprocess.Popen(['gsutil', '-m', 'cp', f'{args.out_dir}/train-{DATASET_VERSION}.zip',
#                   f'gs://photo-voltaic-panels-detection/neural/'])

