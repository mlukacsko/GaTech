import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
wine_dataset = pd.read_csv("Datasets/wineOrigin.csv")

print(wine_dataset.info())
X = wine_dataset.drop({"Wine"}, axis=1)
y = wine_dataset["Wine"]

# normalize X values
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
KNN
'''

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)
knn_accuracy = accuracy_score(knn_prediction,y_test)
print("Accuracy: ", '%.2f'% (knn_accuracy*100),"%")
train_scores, test_scores = validation_curve(knn, X_train, y_train, param_range=np.arange(1,301,20), param_name='n_neighbors', cv=4)
plt.figure()
plt.plot(np.arange(1,301,20), np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.arange(1,301,20), np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Validation Curve: Number of Neighbors")
plt.xlabel("Number of Neighbors")
plt.ylabel("Score (accuracy)")
plt.grid()
plt.show()

train_scores, test_scores = validation_curve(knn, X_train, y_train, param_name="p", param_range=np.arange(1,10), cv=4)
plt.figure()
plt.plot(np.arange(1,10), np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.arange(1,10), np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Validation Curve: Power (Distance Metric)")
plt.xlabel("P")
plt.ylabel("Score (accuracy)")
plt.grid()
plt.xticks(np.arange(1,10))
plt.show()

param_grid = {'n_neighbors': np.arange(1,301,20), 'p': np.arange(1,10)}
best_knn = GridSearchCV(knn, param_grid=param_grid, cv=4)
best_knn.fit(X_train, y_train)
best_accuracy = accuracy_score(y_test, best_knn.predict(X_test))

print("Best params:",best_knn.best_params_)
print("Best accuracy:", '%.2f'% (best_accuracy*100), "%")

knn_learning = KNeighborsClassifier(n_neighbors=best_knn.best_params_['n_neighbors'], p=best_knn.best_params_['p'])
_, train_scores, test_scores = learning_curve(knn_learning, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=4)
plt.figure()
plt.plot(np.linspace(0.1,1.0,10)*100, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.linspace(0.1,1.0,10)*100, np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Learning Curve")
plt.xlabel("Percentage of Training Examples")
plt.ylabel("Score (accuracy)")
plt.xticks(np.linspace(0.1,1.0,10)*100)
plt.grid()
plt.show()

