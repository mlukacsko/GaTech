import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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
SVM
'''
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
svm_prediction = svm.predict(X_test)
svm_accuracy = accuracy_score(svm_prediction,y_test)
print("Accuracy: ", '%.2f'% (svm_accuracy*100),"%")


train_scores, test_scores = validation_curve(svm, X_train, y_train, param_name="C", param_range=np.arange(0.001,100,10), cv=4)
plt.figure()
plt.plot(np.arange(0.001,100,10), np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.arange(0.001,100,10), np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Validation Curve: C")
plt.xlabel("C value")
plt.ylabel("Score (accuracy)")
plt.grid()
plt.show()

rbf_svm = SVC(random_state=42, kernel='rbf')
rbf_svm.fit(X_train, y_train)
rbf_train_scores, rbf_test_scores = validation_curve(rbf_svm, X_train, y_train,param_name='gamma', param_range=np.logspace(-6,1,5),cv=4)
sigmoid_svm = SVC(random_state=42, kernel='sigmoid')
sigmoid_svm.fit(X_train, y_train)
sigmoid_train_scores, sigmoid_test_scores = validation_curve(sigmoid_svm, X_train, y_train,param_name='gamma', param_range=np.logspace(-6,1,5),cv=4)
poly_svm = SVC(random_state=42, kernel='poly')
poly_svm.fit(X_train, y_train)
poly_train_scores, poly_test_scores = validation_curve(poly_svm, X_train, y_train,param_name='gamma', param_range=np.logspace(-6,1,5),cv=4)
linear_svm = SVC(random_state=42, kernel='linear')
linear_svm.fit(X_train, y_train)
linear_train_scores, linear_test_scores = validation_curve(linear_svm, X_train, y_train,param_name='gamma', param_range=np.logspace(-6,1,5),cv=4)
plt.figure()
plt.semilogx(np.logspace(-6,1,5),np.mean(rbf_train_scores, axis=1), label='RBF Training Score')
plt.semilogx(np.logspace(-6,1, 5),np.mean(rbf_test_scores, axis=1), label='RBF CV Score')
plt.semilogx(np.logspace(-6,1,5),np.mean(sigmoid_train_scores, axis=1), label='Sigmoid Training Score')
plt.semilogx(np.logspace(-6,1, 5),np.mean(sigmoid_test_scores, axis=1), label='Sigmoid CV Score')
plt.semilogx(np.logspace(-6,1,5),np.mean(poly_train_scores, axis=1), label='Poly Training Score')
plt.semilogx(np.logspace(-6,1, 5),np.mean(poly_test_scores, axis=1), label='Poly CV Score')
plt.legend()
plt.title("Validation Curve: Gamma and Kernel")
plt.xlabel("Gamma")
plt.ylabel("Score (accuracy)")
plt.grid()
plt.show()


param_grid = {'C': np.arange(0.001,100,10), 'gamma': np.logspace(-6,1,5)}
best_svm = GridSearchCV(svm, param_grid=param_grid, cv=4)
best_svm.fit(X_train, y_train)
best_accuracy = accuracy_score(y_test, best_svm.predict(X_test))
print("Best parameter values:",best_svm.best_params_)
print("Best accuracy:", '%.2f'% (best_accuracy*100), "%")

svm_learning = SVC(random_state=42, C=best_svm.best_params_['C'], gamma=best_svm.best_params_['gamma'])
_, train_scores, test_scores = learning_curve(svm_learning, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=4)
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
