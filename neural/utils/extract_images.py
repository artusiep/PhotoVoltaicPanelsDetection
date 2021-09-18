# import os
# from pathlib import Path
#
# import numpy as np
# from PIL import Image
# from imageio import imwrite
#
# from consts import DEST_IMAGES_WITH_MASKS_PATH, CASE_PATTERN, \
#     PNG_PATTERN, DEST_TUMOR_MASKS_PATH, DEST_KIDNEY_MASKS_PATH, DEST_IMAGES_PATH
# from .Magisterka/kits19/starter_code.utils import load_case
# from starter_code.visualize import hu_to_grayscale, class_to_color, overlay, visualize
#
# DEFAULT_KIDNEY_COLOR = [255, 0, 0]
# DEFAULT_TUMOR_COLOR = [0, 0, 255]
# DEFAULT_HU_MAX = 512
# DEFAULT_HU_MIN = -512
# DEFAULT_OVERLAY_ALPHA = 0.3
#
# def save_image(volume, segmentation, case_index, alpha):
#     # Prepare output location
#     out_path = Path(DEST_IMAGES_PATH + CASE_PATTERN.format(case_index))
#     if not out_path.exists():
#         out_path.mkdir()
#
#     segmentation = segmentation.astype(np.int32)
#
#     # Convert to a visual format
#     vol_ims = hu_to_grayscale(volume, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
#     seg_ims = class_to_color(segmentation, DEFAULT_KIDNEY_COLOR, DEFAULT_TUMOR_COLOR)
#
#     # Save individual images to disk
#     # Overlay the segmentation colors
#     viz_ims = overlay(vol_ims, seg_ims, segmentation, alpha)
#     for i in range(viz_ims.shape[0]):
#         fpath = out_path / ("{:05d}.png".format(i))
#         imwrite(str(fpath), viz_ims[i])
#
#
# def save_image_with_masks(volume, segmentation, case_index, alpha):
#     # Prepare output location
#     out_path = Path(DEST_IMAGES_WITH_MASKS_PATH + CASE_PATTERN.format(case_index))
#     if not out_path.exists():
#         out_path.mkdir()
#
#     segmentation = segmentation.astype(np.int32)
#
#     # Convert to a visual format
#     vol_ims = hu_to_grayscale(volume, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
#     seg_ims = class_to_color(segmentation, DEFAULT_KIDNEY_COLOR, DEFAULT_TUMOR_COLOR)
#
#     # Save individual images to disk
#     # Overlay the segmentation colors
#     viz_ims = overlay(vol_ims, seg_ims, segmentation, alpha)
#     for i in range(viz_ims.shape[0]):
#         fpath = out_path / (PNG_PATTERN.format(i))
#         imwrite(str(fpath), viz_ims[i])
#
#
# def save_kidney_mask(segmentation, case_index):
#     out_path = Path(DEST_KIDNEY_MASKS_PATH + CASE_PATTERN.format(case_index))
#     if not out_path.exists():
#         out_path.mkdir()
#
#     for j in range(0, len(segmentation.get_fdata())):
#         data_segmentation = segmentation.get_fdata()[j]
#         data_segmentation[data_segmentation == 2] = 0
#         img_segmentation = Image.fromarray(np.uint8(data_segmentation * 255))
#         img_segmentation.save(out_path / (PNG_PATTERN.format(j)))
#
#
# def save_tumor_mask(segmentation, case_index):
#     out_path = Path(DEST_TUMOR_MASKS_PATH + CASE_PATTERN.format(case_index))
#     if not out_path.exists():
#         out_path.mkdir()
#
#     for j in range(0, len(segmentation.get_fdata())):
#         data_segmentation = segmentation.get_fdata()[j]
#         data_segmentation[data_segmentation == 1] = 0
#         data_segmentation[data_segmentation == 2] = 1
#         img_segmentation = Image.fromarray(np.uint8(data_segmentation * 255))
#         img_segmentation.save(out_path / (PNG_PATTERN.format(j)))
#
#
# def extract_images(case_indexes):
#     case_with_errors = []
#     for case_index in case_indexes:
#         try:
#             volume, segmentation = load_case(CASE_PATTERN.format(case_index))
#
#             # save kidney mask
#             # save_kidney_mask(segmentation, case_index)
#
#             # # save tumor mask
#             save_tumor_mask(segmentation, case_index)
#
#             # save images
#             #visualize(CASE_PATTERN.format(case_index), DEST_IMAGES_PATH + CASE_PATTERN.format(case_index), alpha=0)
#
#             # save images with mask
#             #visualize(CASE_PATTERN.format(case_index), DEST_IMAGES_WITH_MASKS_PATH + CASE_PATTERN.format(case_index))
#             print("Done case - {:05d}".format(case_index))
#         except:
#             case_with_errors.append(case_index)
#             print("Error case - {:05d}".format(case_index))
#
#     if len(case_with_errors) != 0:
#         for case_index in case_with_errors:
#             try:
#                 os.remove(os.path.abspath('../data/' + CASE_PATTERN.format(case_index) + '/imaging.nii.gz'))
#             except:
#                 print("Fie imaging.nii.gz already removed - case - {:05d}".format(case_index))
#
#         os.chdir("..")
#         os.system("python -m starter_code.get_imaging")
#
#         extract_images(case_with_errors)
#
#
# extract_images(range(200, 201))
