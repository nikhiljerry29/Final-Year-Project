import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import math

img = cv2.imread('crop image/pearl_millet/pearl_millet_single_image_18.png',1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.hist(img.ravel(),256,[0,256]); plt.show()
ret, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
df = pd.DataFrame(columns = ['Perimeter','Eccentricity','Extent','Solidity','BLue', 'Green', 'Red', 'Point Polygon Test'])

i=0
# contours drawing
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	if cv2.contourArea(cnt) > 500:
		i+=1

		area = cv2.contourArea(cnt)
		x,y,w,h = cv2.boundingRect(cnt)
		rect_area = w*h
		extent = float(area)/rect_area
		(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
		hull_area = cv2.contourArea(cv2.convexHull(cnt))
		solidity = float(area)/hull_area
		y = int(y)
		x = int(x)
		ecc = math.sqrt(1 - ((MA * MA)/(ma * ma)))
		print(ecc)
		mean_val = np.array(cv2.mean(img[y:y+h,x:x+w])).astype(np.uint8)
		dist = cv2.pointPolygonTest(cnt,(50,50),True)
		df.loc[len(df)] = [cv2.arcLength(cnt,True),ecc,extent,solidity,mean_val[0],mean_val[1],mean_val[2],dist]

		cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
print(i)
print(df)

# img = thresh
# scale_percent = 20 # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# # resize image
# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow('image',resized)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destryAllWindows()