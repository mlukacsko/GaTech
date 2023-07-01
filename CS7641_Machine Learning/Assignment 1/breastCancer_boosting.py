import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

cancer_dataset = pd.read_csv("Datasets/breast-cancer.csv")
print(cancer_dataset.info())

X = cancer_dataset.drop(['id','diagnosis'], axis=1)
y = cancer_dataset["diagnosis"]

# diagnosis set to binary values for B, M
y[y == 'B'] = 0
y[y == 'M'] = 1
y = y.astype(int)

# normalize X values
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
Boosting
'''
adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)
adaboost_prediction = adaboost.predict(X_test)
adaboost_accuracy = accuracy_score(adaboost_prediction,y_test)
print("Accuracy: ", '%.2f'% (adaboost_accuracy*100),"%")

train_scores, test_scores = validation_curve(adaboost, X_train, y_train, param_name="n_estimators", param_range=np.arange(10,250,10), cv=4)
plt.figure()
plt.plot(np.arange(10,250,10), np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.arange(10,250,10), np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Validation Curve: Number of Estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("Score (accuracy)")
plt.grid()
plt.show()


train_scores, test_scores = validation_curve(adaboost, X_train, y_train, param_name="learning_rate", param_range=np.logspace(-5,5,5), cv=4)
plt.figure()
plt.plot(np.linspace(-5,5,5), np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.linspace(-5,5,5), np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Validation Curve: Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Score (accuracy")
plt.grid()
plt.show()

parameter_space = {
    'n_estimators': np.arange(1,250,5),
    'learning_rate': np.logspace(-10,10,5)
}

best_adaboost = GridSearchCV(adaboost, param_grid=parameter_space, cv=4)
best_adaboost.fit(X_train, y_train)
best_accuracy = accuracy_score(y_test, best_adaboost.predict(X_test))
print("Best params for neural network:",best_adaboost.best_params_)
print("Accuracy for best neural network:", '%.2f'%(best_accuracy*100), "%")

classifier_adaboost_learning = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1),
                       random_state=42,
                       n_estimators=best_adaboost.best_params_['n_estimators'],
                       learning_rate=best_adaboost.best_params_['learning_rate'])
_, train_scores, test_scores = learning_curve(classifier_adaboost_learning, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=4)
plt.figure()
plt.plot(np.linspace(0.1,1.0,10)*100, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.linspace(0.1,1.0,10)*100, np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Learning Curve")
plt.xlabel("Percentage of Training Examples")
plt.ylabel("Score (accuracy")
plt.xticks(np.linspace(0.1,1.0,10)*100)
plt.grid()
plt.show()
