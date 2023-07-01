import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
DECISION TREE
'''
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_prediction = dt.predict(X_test)
dt_accuracy = accuracy_score(dt_prediction,y_test)
print("Accuracy: ", '%.2f'% (dt_accuracy*100),"%")

# Validation Curve
# max depth
train_scores, test_scores = validation_curve(dt, X_train, y_train, param_range=np.arange(1,21), param_name='max_depth', cv=4)
plt.figure()
plt.plot(np.arange(1,21), np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.arange(1,21), np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Validation Curve: Max Depth")
plt.xlabel("Maximum Depth of Tree")
plt.ylabel("Score (accuracy)")
plt.xticks(np.arange(1, 21))
plt.grid()
plt.show()

# ccpa
train_scores, test_scores = validation_curve(dt, X_train, y_train, param_range=np.linspace(0.0001,0.01,10), param_name='ccp_alpha', cv=4)
plt.figure()
plt.plot(np.linspace(0.0001,0.01,10), np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.linspace(0.0001,0.01,10), np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Validation Curve: CCP_Alpha")
plt.xlabel("CCP Alpha")
plt.ylabel("Score (accuracy)")
plt.xticks(np.linspace(0.0001,0.01,10))
plt.grid()
plt.show()


param_grid = {'max_depth': np.arange(1,26),
              'ccp_alpha': np.linspace(0.0001,0.01,10)
              }
best_dt = GridSearchCV(dt, param_grid=param_grid, cv=4)
best_dt.fit(X_train, y_train)


dt_accuracy = accuracy_score(y_test, best_dt.predict(X_test))
print("Best param values for decision tree:",best_dt.best_params_)
print("Accuracy for optimized decision tree:", '%.2f'% (dt_accuracy*100), "%")


# Learning Curve with best params
dt_learning = DecisionTreeClassifier(random_state=42, max_depth=best_dt.best_params_['max_depth'], ccp_alpha=best_dt.best_params_['ccp_alpha'])
_, train_scores, test_scores = learning_curve(dt_learning, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=4)
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
