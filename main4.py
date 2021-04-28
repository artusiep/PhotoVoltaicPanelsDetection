import cv2
import matplotlib.pyplot as plt
import numpy as np
from thermography.detection import FramePreprocessor, RectangleDetector

# Morphological function sets
def morph_operation(matinput):
  kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=2)
  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=1)

  return morph


# Analyze blobs
def analyze_blob(matblobs,display_frame):

  blobs, _ = cv2.findContours(matblobs,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
  valid_blobs = []

  for i, blob in enumerate(blobs):
    rot_rect = cv2.minAreaRect(blob)
    b_rect = cv2.boundingRect(blob)


    (cx,cy),(sw,sh),angle = rot_rect
    rx,ry,rw,rh = b_rect

    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)

    # Draw the segmented Box region
    frame = cv2.drawContours(display_frame,[box],0,(0,0,255),1)

    on_count = cv2.contourArea(blob)
    total_count = sw*sh
    if total_count <= 0:
      continue

    if sh > sw :
      temp = sw
      sw = sh
      sh = temp

    # minimum area
    if sw * sh < 300:
      continue

    # maximum area
    if sw * sh > 1000:
      continue

    # ratio of box
    rect_ratio = sw / sh
    if rect_ratio <= 1 or rect_ratio >= 3:
      continue

    # ratio of fill
    # fill_ratio = on_count / total_count
    # if fill_ratio < 0.4 :
    #   continue

    # remove blob that is too bright
    # if display_frame[int(cy),int(cx),0] > 75:
    #   continue


    valid_blobs.append(blob)

  if valid_blobs:
    print("Number of Blobs : " ,len(valid_blobs))
  cv2.imshow("display_frame_in",display_frame)

  return valid_blobs

def lbp_like_method(matinput,radius,stren,off):

  height, width = np.shape(matinput)

  roi_radius = radius
  peri = roi_radius * 8
  matdst = np.zeros_like(matinput)
  for y in range(height):
    y_ = y - roi_radius
    _y = y + roi_radius
    if y_ < 0 or _y >= height:
      continue


    for x in range(width):
      x_ = x - roi_radius
      _x = x + roi_radius
      if x_ < 0 or _x >= width:
        continue

      r1 = matinput[y_:_y,x_]
      r2 = matinput[y_:_y,_x]
      r3 = matinput[y_,x_:_x]
      r4 = matinput[_y,x_:_x]

      center = matinput[y,x]
      valid_cell_1 = len(r1[r1 > center + off])
      valid_cell_2 = len(r2[r2 > center + off])
      valid_cell_3 = len(r3[r3 > center + off])
      valid_cell_4 = len(r4[r4 > center + off])

      total = valid_cell_1 + valid_cell_2 + valid_cell_3 + valid_cell_4

      if total > stren * peri:
        matdst[y,x] = 255

  return matdst

def preprocess_frame(frame) -> None:
    frame_preprocessor = FramePreprocessor(input_image=frame)
    frame_preprocessor.preprocess()
    return frame_preprocessor.preprocessed_image, frame_preprocessor.scaled_image_rgb



image_path = "data/raw/sample.JPG"
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

preprocessed, last_rgb = preprocess_frame(img)

print('preprocessed')

# Parameter tuning
winsize = 5
peri = 0.6
off = 4

matlbp = lbp_like_method(preprocessed,winsize,peri,off)
cv2.imshow("matlbp",matlbp)
print('waitkey')
cv2.waitKey(1)

print('matlbp')

matmorph = morph_operation(matlbp)
cv2.imshow("matmorph",matmorph)
cv2.waitKey(1)

print('matmorph')

display_color = cv2.cvtColor(preprocessed,cv2.COLOR_GRAY2BGR)
valid_blobs = analyze_blob(matmorph,display_color)


for b in range(len(valid_blobs)):
    cv2.drawContours(display_color,valid_blobs,b,(0,255,255),-1)


cv2.imshow("display_color",display_color)
cv2.waitKey(0)
