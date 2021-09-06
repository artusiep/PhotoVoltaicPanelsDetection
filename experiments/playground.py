import json
import pickle

label_file_path = "data/thermal/plasma-DJI_1_R (437).json"
prediction_file_path = "data/result/plasma-DJI_1_R (437).pickle"

with open(label_file_path) as json_file:
    data = json.load(json_file)
    rectangles = [shape['points'] for shape in data['shapes']]

with open(prediction_file_path, 'rb') as pickle_file:
    test = pickle.load(pickle_file)
    prediction_rectangles = test['rectangles']

from shapely.geometry import Polygon
from shapely.ops import unary_union

i = 1

for predicted_rect in prediction_rectangles:
    predicted_polygon = Polygon(predicted_rect / 3)
    # shape = (640, 512)[::-1]
    # prediction_mask = np.zeros(shape, np.uint8)
    # np_rectangle = np.int32([predicted_rect])
    # cv2.fillConvexPoly(prediction_mask, np_rectangle, 255, cv2.LINE_4)
    # # display_image_in_actual_size(mask)
    for label_rectangle in rectangles:
        label_polygon = Polygon(label_rectangle)
        polygons = [predicted_polygon, label_polygon]

        union = unary_union(polygons)
        intersection = label_polygon.intersection(predicted_polygon)
        # print(union.area)
        # print(intersection.area)
        IOU = intersection.area / union.area
        if IOU >= 0.6:
            print(IOU, i)
            i = i + 1
            # continue

        # shape = (640, 512)[::-1]
        # mask = np.zeros(shape, np.uint8)
        # np_rectangle = np.int32([rectangle])
        # cv2.fillConvexPoly(mask, np_rectangle, 255, cv2.LINE_4)
        # # display_image_in_actual_size(mask)

print(rectangles)
