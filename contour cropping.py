import cv2
import os
os.mkdir('crop image/new')
img = cv2.imread('Images/jowar2.jpg',1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)


# contours drawing
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
a = 279
for cnt in contours:
	if cv2.contourArea(cnt) > 500 :
		a += 1
		x,y,w,h = cv2.boundingRect(cnt)
		cropped = img[y-20:y+h+20, x-20:x+w+20]
		nm = 'crop image/new/cropped_image_'+str(a)+'.png'
		cv2.imwrite(nm, cropped)

# img = thresh
scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('image',resized)


cv2.waitKey(0)
cv2.destryAllWindows()