import os
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import numpy as np


true_img=cv2.imread("data/small-set/label.png")
pred_img=cv2.imread("data/small-set/pred2.tiff")

pred_img = np.array(cv2.threshold(pred_img, 1, 255, cv2.THRESH_BINARY)[1][:,:,0]).flatten()
true_img = np.array(cv2.threshold(true_img, 1, 255, cv2.THRESH_BINARY)[1][:,:,0]).flatten()


print("Confusion Matrix: ",
      confusion_matrix(true_img, pred_img))

print ("Accuracy : ",
       accuracy_score(true_img, pred_img)*100)

print(
      classification_report(true_img, pred_img))

print(
      jaccard_similarity_score(true_img, pred_img))