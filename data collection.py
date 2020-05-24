import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

c = 'sorghum'
path, dirs, files = next(os.walk("crop image/"+ str(c) +"/"))

df = pd.DataFrame(columns = ['Type','Contour Id','Perimeter','Eccentricity','Extent','Solidity', 'Red', 'Green', 'Blue', 'Point Polygon Test'])
print(df)

for i in range(1,len(files) + 1) :
	nm = 'crop image/'+ str(c) +'/'+ str(c) +'_single_image_' + str(i) + '.png'
	img = cv2.imread(nm,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours :
		if cv2.contourArea(cnt) > 1100 :
			(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
			area = cv2.contourArea(cnt)
			x,y,w,h = cv2.boundingRect(cnt)
			rect_area = w*h
			extent = float(area)/rect_area
			hull_area = cv2.contourArea(cv2.convexHull(cnt))
			solidity = float(area)/hull_area
			# y = int(y)
			# x = int(x)
			if ma > MA :
				ecc = math.sqrt(1 - ((MA * MA)/(ma * ma)))
			else :
				ecc = math.sqrt(1 - ((ma * ma)/(MA * MA)))				
			mean_val = np.array(cv2.mean(img[y:y+h,x:x+w])).astype(np.uint8)
			dist = cv2.pointPolygonTest(cnt,(50,50),True)
			df.loc[len(df)] = [c, i, cv2.arcLength(cnt,True), ecc, extent, solidity, mean_val[2], mean_val[1], mean_val[0], dist]
print(df)
df.to_csv("database/dataset_"+ str(c) +".csv",index = False)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()