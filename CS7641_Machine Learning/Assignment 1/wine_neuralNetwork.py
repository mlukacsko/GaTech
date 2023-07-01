import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
wine_dataset = pd.read_csv("Datasets/wineOrigin.csv")

# Drop column 1, we dont need row numbers
print(wine_dataset.info())

X = wine_dataset.drop({"Wine"}, axis=1)
y = wine_dataset["Wine"]

# normalize X values
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
NEURAL NETWORK
'''
nn = MLPClassifier()
nn.fit(X_train, y_train)
nn_prediction = nn.predict(X_test)
nn_accuracy = accuracy_score(nn_prediction,y_test)
print("Accuracy: ", '%.2f'% (nn_accuracy*100),"%")

train_scores, test_scores = validation_curve(nn, X_train, y_train, param_name="alpha", param_range=np.logspace(-10,10,10), cv=4)
plt.figure()
plt.plot(np.linspace(-10,10,10), np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.linspace(-10,10,10),np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Validation Curve: Alpha(Learning Rate)")
plt.xlabel("Alpha")
plt.ylabel("Score (accuracy")
plt.grid()
plt.show()

train_scores, test_scores = validation_curve(nn, X_train, y_train, param_name="hidden_layer_sizes", param_range=np.arange(2,31,2), cv=4)
plt.figure()
plt.plot(np.arange(2,31,2), np.mean(train_scores, axis=1), label='Training Score')
plt.plot(np.arange(2,31,2), np.mean(test_scores, axis=1), label='CV Score')
plt.legend()
plt.title("Validation Curve: Hidden Layer Size")
plt.xlabel("Hidden Layer Size")
plt.ylabel("Score (accuracy)")
plt.grid()
plt.show()


parameter_space = {
    'hidden_layer_sizes': [(0,50,10), (50,50,50), (50,100,50), (100,)],
    'alpha': np.logspace(-10,10,10),
}
nn_best = GridSearchCV(nn, param_grid=parameter_space, cv=4)
nn_best.fit(X_train, y_train)

best_accuracy = accuracy_score(y_test, nn_best.predict(X_test))
print("Best params for neural network:",nn_best.best_params_)
print("Accuracy for best neural network:", '%.2f'%(best_accuracy*100), "%")

plt.plot(nn.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

nn_learning = MLPClassifier(random_state=42,
                            max_iter=1000,
                            hidden_layer_sizes=nn_best.best_params_['hidden_layer_sizes'],
                            alpha=nn_best.best_params_['alpha'])
_, train_scores, test_scores = learning_curve(nn_learning, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=4)

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