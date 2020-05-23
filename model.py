# Importing Libraries
import pandas as pd

# Loading Dataset
df = pd.read_csv('database/dataset.csv')
df = df.drop('Contour Id', axis = 1)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


from sklearn.svm import SVC
model = SVC(C = 1000, gamma = 0.0001, kernel = 'rbf')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Grid Search
# from sklearn.model_selection import GridSearchCV
# param_grid = {'C' : [0.1, 1, 10, 100, 1000], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]}
# grid = GridSearchCV(SVC(), param_grid, verbose = 3)
# grid.fit(X_train, y_train)
# y_pred = grid.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm1 = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# # k-fold  validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model,X = X_train,y = y_train, cv = 50)
print(accuracies.mean())
