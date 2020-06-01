import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
# sns.set(style = 'whitegrid', rc = {'figure.figsize' : (10 , 5)})

# # df = pd.read_csv('database/accuracyvsepochs3.csv')
# # df['Total Neurons'] = df['First Layer Neurons'] + df['Second Layer Neurons']
# # df = df.sort_values('Accuracy')
# # print(df[['Accuracy','Total Neurons']])
# # sns.scatterplot(x = 'First Layer Neurons', y = 'Accuracy', data = df)
# # plt.show()
# # # print(df[df['Accuracy'] == df['Accuracy'].max()])

# df = pd.read_csv('database/dataset.csv')
# df = df.drop(['Contour Id','Point Polygon Test'], axis = 1)
# sns.heatmap(df.corr(), annot = True)
# plt.show()
img = cv2.imread("crop image/sorghum/sorghum_single_image_22.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	if cv2.contourArea(cnt) > 500:
		hull = cv2.convexHull(cnt)
		cv2.drawContours(img, [hull], -1, (0,255,0), 1)
		cv2.drawContours(img, [cnt], -1, (0,0,255), 1)


cv2.imshow("sds", img)
# cv2.imwrite("5.png",img)
cv2.waitKey(0)
cv2.destroyAllwindows()