import glob
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
import cv2
import labelme

DATA_DIR = 'raw'
OUT_DIR = DATA_DIR + '/data'
class_names = []
class_name_to_id = {'0': 1}
# for i, line in enumerate(open(DATA_DIR + '/labels.txt').readlines()):
# 	class_id = i - 1  # starts with -1
# 	class_name = line.strip()
# 	class_name_to_id[class_name] = class_id
# 	if class_id == -1:
# 		assert class_name == '__ignore__'
# 		continue
# 	elif class_id == 0:
# 		assert class_name == '_background_'
# 	class_names.append(class_name)
# class_names = tuple(class_names)
# print('class_names:', class_names)
# out_class_names_file = osp.join(DATA_DIR, 'class_names.txt')
# with open(out_class_names_file, 'w') as f:
# 	f.writelines('\n'.join(class_names))
# print('Saved class_names:', out_class_names_file)
#
# if osp.exists(OUT_DIR):
# 	print('Output directory already exists:', OUT_DIR)
# 	quit(1)
# os.makedirs(OUT_DIR)

# os.makedirs(osp.join(OUT_DIR))

for label_file in sorted(glob.glob(osp.join(DATA_DIR, '*.json'))):
    with open(label_file) as f:
        base = osp.splitext(osp.basename(label_file))[0]
        data = json.load(f)
        img_file = osp.join(osp.dirname(label_file), data['imagePath'])
        img = np.asarray(PIL.Image.open(img_file))
        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data['shapes'],
            label_name_to_value=class_name_to_id,
        )
        instance1 = np.copy(lbl)[0]

        instance1 = (instance1 * 255).astype(np.uint8)
        instance1 = cv2.cvtColor(instance1, cv2.COLOR_GRAY2BGR)

        PIL.Image.fromarray(img).save(osp.join(OUT_DIR, base + '.png'))

        # cv2.imwrite(osp.join(OUT_DIR, base + '.png'), instance1)

