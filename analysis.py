import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd


img = cv2.imread('Images/bajra3.jpg',1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.hist(img.ravel(),256,[0,256]); plt.show()
ret, thresh = cv2.threshold(gray, 68, 255, cv2.THRESH_BINARY)

i=0
# contours drawing
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	if cv2.contourArea(cnt) > 500:
		i+=1
		cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
print(i)

# img = thresh
scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('image',resized)


# cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destryAllWindows()