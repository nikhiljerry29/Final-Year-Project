# Importing Libraries
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

# Loading Dataset
df = pd.read_csv('database/dataset.csv')
df = df.drop(['Contour Id','Point Polygon Test'], axis = 1)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


def accuracy(first, second, n_epochs, n_batch) :
	# Part 2 - Building the ANN
	# Initializing the ANN
	ann = Sequential()
	ann.add(Dense(output_dim = first, init = 'uniform', activation = 'relu', input_dim = 7))
	ann.add(Dense(output_dim = second, init = 'uniform', activation = 'relu'))
	# for i in range(layers):
	# 	ann.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))
	ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


	# Part 3 - Training the ANN

	# Compiling the ANN
	ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

	# Training the ANN on the Training set
	ann.fit(X_train, y_train, batch_size = n_batch, epochs = n_epochs)

	# Part 4 - Making the predictions and evaluating the model

	# Predicting the Test set results
	y_pred = ann.predict(X_test)
	y_pred = (y_pred > 0.5)

	# Making the Confusion Matrix
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y_test, y_pred)
	return (cm[0][0] + cm[1][1])/cm.sum()

# Checking at Different number of Layers
df1 = pd.DataFrame(columns = ['First Layer Neurons','Second Layer Neurons','Epochs', 'Batch_Size', 'Accuracy'])
for fir in range(6,7):
	# first layer neurons
	# for sec in range(7,8):
		# second layer neurons
	sec = fir
	for batch in list([1,16,32,64]):
		acc = []
		# For Epochs count
		for j in list([100]):
			acc.append(accuracy(fir, sec, j, batch))
		# Changing into dataframe
		# x = [i for i in range(50,101)]
		x = [100]
		df2 = pd.DataFrame({'First Layer Neurons': fir,'Second Layer Neurons' : sec,'Epochs' : list(x), 'Batch_Size' : batch, 'Accuracy' : list(acc)})
		df1 = pd.merge(df1, df2, how = 'outer')
print(df1)
df1.to_csv('database/accuracyvsbatchsize.csv', index = False)

# import os
# os.system("shutdown now -h")