import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
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

classifier_accuracy = np.zeros(5)
time_to_train = np.zeros(5)

'''
Decision Tree
'''
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)
param_grid = {'max_depth': np.arange(1,26), 'ccp_alpha': np.linspace(0.0001,0.01,10)}
best_dec_tree = GridSearchCV(dec_tree, param_grid=param_grid, cv=4)
start_time = time.time()
best_dec_tree.fit(X_train, y_train)
end_time = time.time()
time_to_train[0] = end_time-start_time
classifier_accuracy[0] = accuracy_score(y_test, best_dec_tree.predict(X_test))
print("dt done")

'''
Neural Network
'''
neural_network = MLPClassifier()
neural_network.fit(X_train, y_train)
parameter_space = {
    'hidden_layer_sizes': [(0,50,10), (50,50,50), (50,100,50), (100,)],
    'alpha': np.logspace(-10,10,10),
}
classifier_neural_network_best = GridSearchCV(neural_network, param_grid=parameter_space, cv=4)
start_time = time.time()
classifier_neural_network_best.fit(X_train, y_train)
end_time = time.time()
time_to_train[1] = end_time-start_time
classifier_accuracy[1] = accuracy_score(y_test, classifier_neural_network_best.predict(X_test))
print("nn done")

'''
Boosting
'''
classifier_adaboost = AdaBoostClassifier()
classifier_adaboost.fit(X_train, y_train)
parameter_space = {
    'n_estimators': np.arange(1,250,5),
    'learning_rate': np.logspace(-10,10,5)
}
classifier_adaboost_best = GridSearchCV(classifier_adaboost, param_grid=parameter_space, cv=4)
start_time = time.time()
classifier_adaboost_best.fit(X_train, y_train)
end_time = time.time()
time_to_train[2] = end_time-start_time
classifier_accuracy[2] = accuracy_score(y_test, classifier_adaboost_best.predict(X_test))
print("boosting done")

'''
SVM
'''
classifier_svm = SVC(random_state=42)
classifier_svm.fit(X_train, y_train)
param_grid = {'C': np.arange(0.001,100,25), 'gamma': np.logspace(-6,1,5)}
best_svm = GridSearchCV(classifier_svm, param_grid=param_grid, cv=4)
start_time = time.time()
best_svm.fit(X_train, y_train)
end_time = time.time()
time_to_train[3] = end_time-start_time
classifier_accuracy[3] = accuracy_score(y_test, best_svm.predict(X_test))
print("svm done")

'''
k-Nearest Neighbor
'''
classifier_knn = KNeighborsClassifier()
classifier_knn.fit(X_train, y_train)
param_grid = {'n_neighbors': np.arange(1,301, 20), 'p': np.arange(1,10)}
best_knn = GridSearchCV(classifier_knn, param_grid=param_grid, cv=4)
start_time = time.time()
best_knn.fit(X_train, y_train)
end_time = time.time()
time_to_train[4] = end_time-start_time
classifier_accuracy[4] = accuracy_score(y_test, best_knn.predict(X_test))
print("knn done")
print()
print(time_to_train)
print(classifier_accuracy)

classifiers = ('DT', 'NN', 'AdaBoost', 'SVM', 'k-NN')
y_ticks = np.arange(len(time_to_train))
plt.figure()
plt.barh(y_ticks, time_to_train)
plt.gca().set_yticks(y_ticks)
plt.gca().set_yticklabels(classifiers)
plt.title('Classifier Training Times')
plt.xlabel('Training Time (Seconds)')
plt.show()

plt.figure()
plt.barh(y_ticks, classifier_accuracy)
plt.gca().set_yticks(y_ticks)
plt.gca().set_yticklabels(classifiers)
plt.title('Classifier Accuracy')
plt.xlabel('Accuracy')
plt.gca().set_xlim(0.50, 1.0)
plt.show()