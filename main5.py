import cv2
import numpy as np

img = cv2.imread("data/raw/sample.JPG")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsv.jpg', hsv)
ret, thresh1 = cv2.threshold(hsv[:,:,0],100,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

try: hierarchy = hierarchy[0]
except: hierarchy = []

for contour, hier in zip(contours, hierarchy):
    area = cv2.contourArea(contour)
    if area > 10000 and area < 250000:
       rect = cv2.minAreaRect(contour)
       box = cv2.boxPoints(rect)
       box = np.int0(box)
       cv2.drawContours(img,[box],0,(0,0,255),2)
       cv2.imshow('cont imge', img)
       cv2.waitKey(0)

cv2.imwrite("result.jpg",img)