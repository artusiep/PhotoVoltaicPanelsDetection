import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = "data/raw/sample.JPG"
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

resized = cv2.resize(src=img, dsize=(0, 0), fx=8, fy=8)
blurred = cv2.blur(resized, (15, 15))
grayed = blurred #cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(image=grayed, threshold1=50, threshold2=100, apertureSize=3)

lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 60, minLineLength=100, maxLineGap=150)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(grayed, (x1, y1), (x2, y2), (255, 0, 0), 3)

cv2.imshow("Hough", grayed)
cv2.waitKey()
