import cv2
import numpy as np


def add_text_to_image(image, threshold, img_f1_result, font_scale=0.25):
    cv2.putText(image, "Threshold: {:02f}".format(threshold), (10, int(10 * (font_scale / 0.25))), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    cv2.putText(image, "F1 score: {:02f}".format(img_f1_result), (10, int(20 * (font_scale / 0.25))), cv2.FONT_HERSHEY_SIMPLEX, font_scale,  (255, 255, 255), 1)


def add_original_mask_to_image(image, original_mask, color=[255, 0, 0]):
    rgb_original_mask = cv2.cvtColor(np.array(original_mask * 255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    rgb_original_mask[np.where(np.all(rgb_original_mask == (255, 255, 255), axis=-1))] = color
    return cv2.addWeighted(image, 1, rgb_original_mask, 0.3, 0)


def add_predict_mask_to_image(image, predict_mask, color=[0, 0, 255]):
    rgb_predict_mask = cv2.cvtColor(np.array(predict_mask * 255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    rgb_predict_mask[np.where(np.all(rgb_predict_mask == (255, 255, 255), axis=-1))] = color
    return cv2.addWeighted(image, 1, rgb_predict_mask, 0.3, 0)
