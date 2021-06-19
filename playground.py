import glob
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from configs.jet import JetConfig
from detector.utils import save_img
from display import draw_segments
from thermography.detection import SegmentClusterer, SegmentDetector


def resize_image(input_image, image_scaling):
    resized_image = cv2.resize(src=input_image, dsize=(0, 0), fx=image_scaling, fy=image_scaling)
    return resized_image


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


# https://github.com/satvik007/Scanner_OP/tree/master/src/ex03
def strengthen_edges(input_image):
    # blurring (probably could be improved, reduced to one)
    blurred_image = cv2.bilateralFilter(input_image, 3, 200, 200)
    blurred_image = cv2.GaussianBlur(blurred_image, (7, 7), 0)

    # adaptive thresholding
    adaptive_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised_image = cv2.fastNlMeansDenoising(adaptive_threshold, 11, 31, 9)  #30, 7, 25

    # contrasting
    brightness = 0
    contrast = 64  # contrast_const

    contrasted_image = apply_brightness_contrast(denoised_image, brightness, contrast)
    return contrasted_image


# https://docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html
# https://github.com/karaage0703/python-image-processing/blob/master/watershed.py
def remove_background(input_image):
    image_scaling = 1
    scaled_image = resize_image(input_image, image_scaling)

    gaussian_blur = 3
    blurred_image = cv2.blur(scaled_image, (gaussian_blur, gaussian_blur))
    # blurred_image = cv2.GaussianBlur(scaled_image, (3,3), 0)
    # blurred_image = cv2.bilateralFilter(scaled_image, 5, 120, 120)

    grayed_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.adaptiveThreshold(grayed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 713, 1)

    # noise removal
    kernel = np.ones((7,7), np.uint8)
    opening = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # find sure foreground area
    distance_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    _, sure_foreground_image = cv2.threshold(distance_transform, 0.01 * distance_transform.max(), 255, 0)
    sure_foreground_image = np.uint8(sure_foreground_image)

    background_removed_image = cv2.bitwise_and(grayed_image, sure_foreground_image)

    # discard small contours/areas
    min_area = 200 * 100 * image_scaling
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    discarded_contours = [area < min_area for area in areas]
    contours = [contours[i] for i in range(len(contours)) if not discarded_contours[i]]

    mask = np.zeros_like(background_removed_image)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.blur(mask, (25, 25))
    mask = mask.astype(np.float) / 255.
    retained_contours_image = (background_removed_image * mask).astype(np.uint8)
    return retained_contours_image


def detect_edges(input_image):
    hysteresis_min_thresh = 35
    hysteresis_max_thresh = 45

    canny_image = cv2.Canny(image=input_image, threshold1=hysteresis_min_thresh,
                            threshold2=hysteresis_max_thresh, apertureSize=3)

    kernel_size = (7, 7)
    kernel_shape = cv2.MORPH_CROSS
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)

    dilation_steps = 4
    dilated = cv2.dilate(canny_image, (3, 3), iterations=dilation_steps)

    size = np.size(dilated)
    skel = np.zeros(dilated.shape, np.uint8)

    img = dilated
    done = False

    while not done:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel


# https://stackoverflow.com/questions/3648910/selecting-the-pixels-with-highest-intensity-in-opencv
def remove_reflections(input_image):
    grayed_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    hi_percentage = 0.03  # we want the hi_percentage brightest pixels
    hist = cv2.calcHist([grayed_image], [0], None, [256], [0, 256]).flatten()

    # * find brightness threshold
    # here: highest thresh for including at least hi_percentage image pixels,
    #       maybe you want to modify it for lowest threshold with for including
    #       at most hi_percentage pixels

    total_count = grayed_image.shape[0] * grayed_image.shape[1]  # height * width
    target_count = hi_percentage * total_count  # bright pixels we look for
    summed = 0
    for i in range(255, 0, -1):
        summed += int(hist[i])
        if target_count <= summed:
            hi_thresh = i
            break
    else:
        hi_thresh = 0

    brightest_pixels_image = cv2.threshold(grayed_image, hi_thresh, 0, cv2.THRESH_TOZERO)[1]
    inpainted_image = cv2.inpaint(input_image, brightest_pixels_image, 21, cv2.INPAINT_TELEA)
    return inpainted_image


# display
def detect_segments_functional(frame, params) -> Any:
    segment_detector = SegmentDetector(input_image=frame, params=params)
    segment_detector.detect()
    return segment_detector.segments


def cluster_segments_functional(segments, params, cleaning_params) -> Any:
    segment_clusterer = SegmentClusterer(input_segments=segments, params=params)
    segment_clusterer.cluster_segments()
    mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()
    segment_clusterer.clean_clusters(mean_angles=mean_angles, params=cleaning_params)
    return segment_clusterer.cluster_list


def process_image(img_path):
    input_image = cv2.imread(img_path)
    original_image = input_image.copy()

    # 1. remove reflections
    removed_reflections_image = remove_reflections(input_image)
    cv2.imshow('removed_reflections_image', removed_reflections_image)

    # 2. remove background
    removed_background_image = remove_background(removed_reflections_image)
    cv2.imshow('removed_background', removed_background_image)

    # 3. strengthen edges
    strengthened_edges_image = strengthen_edges(removed_background_image)
    _, global_thresh = cv2.threshold(removed_background_image, 0, 255, cv2.THRESH_BINARY)
    strengthened_edges_image = cv2.bitwise_and(strengthened_edges_image, global_thresh)
    cv2.imshow('strengthened_edges_image', strengthened_edges_image)

    #  4. strengthen edges on removed background
    image_scaling = 3
    original_image = resize_image(original_image, image_scaling)
    removed_background_image = resize_image(removed_background_image, image_scaling)
    strengthened_edges_image = resize_image(strengthened_edges_image, image_scaling)

    visible_edges_image = cv2.addWeighted(removed_background_image, 1,
                                    strengthened_edges_image, 0.5, 0, removed_background_image)

    cv2.imshow('visible_edges_image', visible_edges_image)

    blurred_image = cv2.bilateralFilter(strengthened_edges_image, 9, 200, 200)
    cv2.imshow('blurred_image', blurred_image)

    # 5. canny edge detection
    canny_image = detect_edges(strengthened_edges_image)
    cv2.imshow('canny_image', canny_image)

    # 6. segmentation
    config = JetConfig()
    segments = detect_segments_functional(canny_image, config.segment_detector_params)
    cluster_list = cluster_segments_functional(segments, params=config.segment_clusterer_params,
                                               cleaning_params=config.cluster_cleaning_params)

    draw_segments(cluster_list, original_image, "Segments", render_indices=True)

    # save images
    img_name = img_path.rsplit('/', 1)[-1]
    base_output_path = img_path.replace(f'plasma/{img_name}', f'plasma_results/{current_time}')
    save_img(removed_reflections_image, base_output_path + '/removed_reflections/' + img_name)
    save_img(removed_background_image, base_output_path + '/removed_background/' + img_name)
    save_img(strengthened_edges_image, base_output_path + '/strengthen_edges/' + img_name)
    save_img(visible_edges_image, base_output_path + '/visible_edges/' + img_name)
    save_img(canny_image, base_output_path + '/canny/' + img_name)
    save_img(original_image, base_output_path + '/result/' + img_name)
    return original_image


def process(file_paths):
    for index, file_path in enumerate(file_paths):
        new_file_path = file_path.replace('plasma', f'plasma_results/{current_time}')
        print(f"Result will be saved to {new_file_path}. Run index {index}")
        try:
            segmented_image = process_image(file_path)
        except Exception as e:
            print(f"Segmenting of img {file_path} failed with {e}")
            continue
        save_img(segmented_image, new_file_path)


if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    files = glob.glob("data/plasma/*.JPG")
    process(files)


# current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
# img_path = "data/plasma/5.JPG"
# processed = process_image(img_path)
# cv2.imshow('processed', processed)
# cv2.waitKey()
