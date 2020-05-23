import cv2

cm = 'Images/jowar3.jpg'
img = cv2.imread(cm,1)
gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gry, 61, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ln = 0
for cnt in contours:
	if cv2.contourArea(cnt) > 500:
		ln += 1
		cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
		x,y,w,h = cv2.boundingRect(cnt)
		cropped = img[y-20:y+h+20, x-20:x+w+20]
		ret, thresh1 = cv2.threshold(cropped, 63, 255, cv2.THRESH_BINARY)
print(ln)
cv2.imwrite('Threshold Images/sorghum/sorghum_with_contour_bajra3_.png', img)

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