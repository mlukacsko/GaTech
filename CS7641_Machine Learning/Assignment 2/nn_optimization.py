import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import time

data = pd.read_csv('Datasets/breast-cancer.csv')

Y = data.diagnosis
Y = Y.values
Y[Y == 'B'] = 0
Y[Y == 'M'] = 1
Y = Y.astype(int)

X = data.drop(['id', 'diagnosis'], axis=1)


X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

learning_rate = [0.00001, 0.001, 0.1, 1]
restarts = [2, 5, 10, 15]
decay_schedule = [mlrose_hiive.GeomDecay(), mlrose_hiive.ExpDecay(), mlrose_hiive.ArithDecay()]
pop_size = [25, 150, 250, 500]

rhc_training_accuracy = np.zeros((len(learning_rate), len(restarts)))
rhc_validation_accuracy = np.zeros((len(learning_rate), len(restarts)))
rhc_test_accuracy = np.zeros((len(learning_rate), len(restarts)))
rhc_fitness_time = np.zeros((len(learning_rate), len(restarts)))
rhc_best_validation_accuracy = 0.0
rhc_best_x = 0
rhc_best_y = 0


sa_training_accuracy = np.zeros((len(learning_rate), len(decay_schedule)))
sa_validation_accuracy = np.zeros((len(learning_rate), len(decay_schedule)))
sa_test_accuracy = np.zeros((len(learning_rate), len(decay_schedule)))
sa_fitness_time = np.zeros((len(learning_rate), len(decay_schedule)))
sa_best_validation_accuracy = 0.0
sa_best_x = 0
sa_best_y = 0

ga_training_accuracy = np.zeros((len(learning_rate), len(pop_size)))
ga_validation_accuracy = np.zeros((len(learning_rate), len(pop_size)))
ga_test_accuracy = np.zeros((len(learning_rate), len(pop_size)))
ga_fitness_time = np.zeros((len(learning_rate), len(pop_size)))
ga_validation_accuracy_best = 0.0
ga_best_x = 0
ga_best_y = 0


bp_training_accuracy = np.zeros((len(learning_rate), 1))
bp_validation_accuracy = np.zeros((len(learning_rate), 1))
bp_test_accuracy = np.zeros((len(learning_rate), 1))
bp_fitness_time = np.zeros((len(learning_rate), 1))
bp_validation_accuracy_best = 0.0
bp_best_x = 0
bp_best_y = 0


"""
Neural Network - Random Hill Climbing Optimization
"""
rhc_best = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='random_hill_climb',
                                               max_iters=1000, learning_rate=0.00001, early_stopping=True,
                                               clip_max = 5, max_attempts=100, random_state=3, curve=True)

for x, lr in enumerate(learning_rate):
    for y, r in enumerate(restarts):
        rhc = mlrose_hiive.NeuralNetwork(hidden_nodes=[2],
                                                  algorithm='random_hill_climb', max_iters=1000, restarts=r,
                                                  learning_rate=lr, early_stopping=True,
                                                  clip_max = 5, max_attempts=100, random_state=3, curve=True)

        start_time = time.time()
        rhc.fit(X_train, y_train)
        end_time = time.time()
        rhc_time = end_time - start_time

        rhc_yTrain_predict = rhc.predict(X_train)
        rhc_yTrain_accuracy = accuracy_score(y_train, rhc_yTrain_predict)
        rhc_training_accuracy[x][y] = rhc_yTrain_accuracy

        rhc_yValidate_predict = rhc.predict(X_validate)
        rhc_yValidate_accuracy = accuracy_score(y_validate, rhc_yValidate_predict)
        rhc_validation_accuracy[x][y] = rhc_yValidate_accuracy

        rhc_yTest_predict = rhc.predict(X_test)
        rhc_yTest_accuracy = accuracy_score(y_test, rhc_yTest_predict)
        rhc_test_accuracy[x][y] = rhc_yTest_accuracy
        rhc_fitness_time[x][y] = rhc_time

        if rhc_yValidate_accuracy > rhc_best_validation_accuracy:
            rhc_best = rhc
            rhc_best_x = x
            rhc_best_y = y
            rhc_best_validation_accuracy = rhc_yValidate_accuracy

plt.figure()
plt.plot(rhc_best.fitness_curve[:, 0])
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Random Hill Climbing: Loss vs Iterations')
plt.savefig('nn_rhc_lossCurve.png')

print("RHC Mean Time", np.mean(rhc_fitness_time))
print("RHC CV Accuracy:", rhc_test_accuracy[rhc_best_x][rhc_best_y])
print("RHC Train Accuracy", rhc_training_accuracy[rhc_best_x][rhc_best_y])
print('RHC Done')

"""
Neural Network - Simulated Annealing Optimization
 """

sa_best = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='simulated_annealing',
                                              max_iters=1000, bias=True,
                                              learning_rate=0.00001, early_stopping=True,
                                              clip_max = 5, max_attempts=100, random_state=3, curve=True)

for x, lr in enumerate(learning_rate):
    for y, d in enumerate(decay_schedule):
        sa = mlrose_hiive.NeuralNetwork(hidden_nodes=[2],
                                                 algorithm='simulated_annealing',
                                                 max_iters=1000, bias=True, schedule=d,
                                                 learning_rate=lr, early_stopping=True,
                                                 max_attempts=100, random_state=3, curve=True)

        start_time = time.time()
        sa.fit(X_train, y_train)
        end_time = time.time()
        sa_time = end_time - start_time

        sa_yTrain_predict = sa.predict(X_train)
        sa_yTrain_accuracy = accuracy_score(y_train, sa_yTrain_predict)
        sa_training_accuracy[x][y] = sa_yTrain_accuracy

        sa_yValidate_predict = sa.predict(X_validate)
        sa_yValidateAccuracy = accuracy_score(y_validate, sa_yValidate_predict)
        sa_validation_accuracy[x][y] = sa_yValidateAccuracy

        sa_yTest_predict = sa.predict(X_test)
        sa_yTest_accuracy = accuracy_score(y_test, sa_yTest_predict)
        sa_test_accuracy[x][y] = sa_yTest_accuracy
        sa_fitness_time[x][y] = sa_time

        if sa_yValidateAccuracy > sa_best_validation_accuracy:
            sa_best = sa
            sa_best_x = x
            sa_best_y = y
            sa_best_validation_accuracy = sa_yValidateAccuracy

plt.figure()
plt.plot(sa_best.fitness_curve[:, 0])
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Simulated Annealing: Loss vs Iterations')
plt.savefig('nn_sa_lossCurve.png')

sa_yTest_predict = sa_best.predict(X_test)

print("SA Mean Time", np.mean(sa_fitness_time))
print("SA CV Accuracy:", sa_test_accuracy[sa_best_x][sa_best_y])
print("SA Train Accuracy", sa_training_accuracy[sa_best_x][sa_best_y])
print('SA Done')

"""
Neural Network - Backpropagation
 """

bp_best = mlrose_hiive.NeuralNetwork(hidden_nodes=[2],
                                                    algorithm='gradient_descent',
                                                    max_iters=1000, bias=True,
                                                    learning_rate=0.00001, early_stopping=True,
                                                    clip_max = 5, max_attempts=100, random_state=3, curve=True)

for x, lr in enumerate(learning_rate):
    for y in range(1):
        bp = mlrose_hiive.NeuralNetwork(hidden_nodes=[2],
                                                       algorithm='gradient_descent',
                                                       max_iters=1000, bias=True,
                                                       learning_rate=lr, early_stopping=True,
                                                       clip_max = 5, max_attempts=100, random_state=3, curve=True)

        start_time = time.time()
        bp.fit(X_train, y_train)
        end_time = time.time()
        bp_time = end_time - start_time

        bp_yTrain_predict = bp.predict(X_train)
        bp_yTrain_accuracy = accuracy_score(y_train, bp_yTrain_predict)
        bp_training_accuracy[x][y] = bp_yTrain_accuracy

        bp_yValidate_predict = bp.predict(X_validate)
        bp_yValidate_accuracy = accuracy_score(y_validate, bp_yValidate_predict)
        bp_validation_accuracy[x][y] = bp_yValidate_accuracy

        bp_yTest_predict = bp.predict(X_test)
        bp_yTest_accuracy = accuracy_score(y_test, bp_yTest_predict)
        bp_test_accuracy[x][y] = bp_yTest_accuracy
        bp_fitness_time[x][y] = bp_time

        if bp_yValidate_accuracy > bp_validation_accuracy_best:
            bp_best = bp
            bp_best_x = x
            bp_best_y = y
            bp_validation_accuracy_best = bp_yValidate_accuracy

plt.figure()
plt.plot(-bp_best.fitness_curve)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Backpropagation: Loss vs Iterations')
plt.savefig('nn_bp_lossCurve.png')

bp_yTest_predict = bp_best.predict(X_test)

print("BP Mean Time", np.mean(bp_fitness_time))
print("BP CV Accuracy:", bp_test_accuracy[bp_best_x][bp_best_y])
print("BP Train Accuracy", bp_training_accuracy[bp_best_x][bp_best_y])
print('BP Done')

"""
Neural Network - Genetic Algorithm Optimization
 """

ga_best = mlrose_hiive.NeuralNetwork(hidden_nodes=[2],
                                              algorithm='genetic_alg',
                                              max_iters=500, bias=True,
                                              learning_rate=0.00001, early_stopping=True,
                                              clip_max = 5, max_attempts=100, random_state=3, curve=True)

for x, lr in enumerate(learning_rate):
    for y, p in enumerate(pop_size):
        ga = mlrose_hiive.NeuralNetwork(hidden_nodes=[2],
                                                 algorithm='genetic_alg',
                                                 max_iters=500, bias=True, pop_size=p,
                                                 learning_rate=lr, early_stopping=True,
                                                 clip_max = 5, max_attempts=100, random_state=3, curve=True)

        start_time = time.time()
        ga.fit(X_train, y_train)
        end_time = time.time()
        ga_time = end_time - start_time

        ga_yTrain_predict = ga.predict(X_train)
        ga_yTrain_accuracy = accuracy_score(y_train, ga_yTrain_predict)
        ga_training_accuracy[x][y] = ga_yTrain_accuracy

        ga_yValidate_predict = ga.predict(X_validate)
        ga_yValidate_accuracy = accuracy_score(y_validate, ga_yValidate_predict)
        ga_validation_accuracy[x][y] = ga_yValidate_accuracy

        ga_yTest_predict = ga.predict(X_test)
        ga_yTest_accuracy = accuracy_score(y_test, ga_yTest_predict)
        ga_test_accuracy[x][y] = ga_yTest_accuracy
        ga_fitness_time[x][y] = ga_time

        if ga_yValidate_accuracy > ga_validation_accuracy_best:
            ga_best = ga
            ga_best_x = x
            ga_best_y = y
            ga_validation_accuracy_best = ga_yValidate_accuracy

plt.figure()
plt.plot(ga_best.fitness_curve[:, 0])
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Genetic Algorithm: Loss vs Iterations')
plt.savefig('nn_ga_lossCurve.png')

ga_yTest_predict = ga_best.predict(X_test)

print("GA Mean Time", np.mean(ga_fitness_time))
print("GA CV Accuracy:", ga_test_accuracy[ga_best_x][ga_best_y])
print("GA Train Accuracy", ga_training_accuracy[ga_best_x][ga_best_y])
print('GA Done')

"""
Done with loss Curves
Save time, CV and Test Scores
"""

plt.figure()
plt.bar(['RHC', 'SA', 'GA', 'Backprop'], [rhc_fitness_time[rhc_best_x][rhc_best_y], sa_fitness_time[sa_best_x][sa_best_y],
                                          ga_fitness_time[ga_best_x][ga_best_y],
                                          bp_fitness_time[bp_best_x][bp_best_y]])
plt.xlabel("Algorithm")
plt.ylabel("Time (seconds)")
plt.title('Times Analysis')
plt.savefig('nn_bestTimes.png')

plt.figure()
plt.bar(['RHC', 'SA', 'GA', 'Backprop'],
        [rhc_test_accuracy[rhc_best_x][rhc_best_y], sa_test_accuracy[sa_best_x][sa_best_y],
         ga_test_accuracy[ga_best_x][ga_best_y], bp_test_accuracy[bp_best_x][bp_best_y]])
plt.xlabel("Algorithm")
plt.ylabel("Score (percent)")
plt.title('Test Score Analysis')
plt.ylim((0.75, 1.0))
plt.savefig('nn_testScores.png')

plt.figure()
plt.bar(['RHC', 'SA', 'GA', 'Backprop'],
        [rhc_training_accuracy[rhc_best_x][rhc_best_y], sa_training_accuracy[sa_best_x][sa_best_y],
         ga_training_accuracy[ga_best_x][ga_best_y], bp_training_accuracy[bp_best_x][bp_best_y]])
plt.xlabel("Algorithm")
plt.ylabel("Score (percent)")
plt.title('Training Score Analysis')
plt.ylim((0.75, 1.0))
plt.savefig('nn_trainScores.png')

test_data = [0.99, 0.8, 0.5, 0.2, 0.1, 0.01]

rhc_train_accuracy_learningCurve, rhc_validation_accuracy_learningCurve = [], []
sa_train_accuracy_learningCurve, sa_validation_accuracy_learningCurve = [], []
ga_train_accuracy_learningCurve, ga_validation_accuracy_learningCurve = [], []
bp_train_accuracy_learningCurve, bp_validation_accuracy_learningCurve = [], []


"""
Training new algorithms
Save Learning Curves
"""
for test in test_data:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test, random_state=3)
    rhc = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='random_hill_climb',
                                              max_iters=1000, bias=True, restarts=restarts[rhc_best_y],
                                              learning_rate=learning_rate[rhc_best_x], early_stopping=True,
                                              clip_max = 5,max_attempts=100, random_state=3, curve=True)
    sa = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='simulated_annealing',
                                             max_iters=1000, bias=True, schedule=decay_schedule[sa_best_y],
                                             learning_rate=learning_rate[sa_best_x], early_stopping=True,
                                             clip_max = 5,max_attempts=100, random_state=3, curve=True)
    ga = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='genetic_alg',
                                             max_iters=500, bias=True, pop_size=pop_size[ga_best_y],
                                             learning_rate=learning_rate[ga_best_x], early_stopping=True,
                                             clip_max = 5,max_attempts=100, random_state=3, curve=True)
    bp = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='gradient_descent',
                                                   max_iters=1000, bias=True, learning_rate=learning_rate[bp_best_x],
                                                   early_stopping=True,
                                                   clip_max = 5,max_attempts=100, random_state=3, curve=True)

    ga.fit(X_train, y_train)
    ga_yTrain_predict = ga.predict(X_train)
    ga_yTrain_accuracy = accuracy_score(y_train, ga_yTrain_predict)
    ga_train_accuracy_learningCurve.append(ga_yTrain_accuracy)

    ga_yTest_predict = ga.predict(X_test)
    ga_yTest_accuracy = accuracy_score(y_test, ga_yTest_predict)
    ga_validation_accuracy_learningCurve.append(ga_yTest_accuracy)

    bp.fit(X_train, y_train)
    bp_yTrain_predict = bp.predict(X_train)
    bp_yTrain_accuracy = accuracy_score(y_train, bp_yTrain_predict)
    bp_train_accuracy_learningCurve.append(bp_yTrain_accuracy)

    bp_yTest_predict = bp.predict(X_test)
    bp_yTest_accuracy = accuracy_score(y_test, bp_yTest_predict)
    bp_validation_accuracy_learningCurve.append(bp_yTest_accuracy)

    rhc.fit(X_train, y_train)
    rhc_yTrain_predict = rhc.predict(X_train)
    rhc_yTrain_accuracy = accuracy_score(y_train, rhc_yTrain_predict)
    rhc_train_accuracy_learningCurve.append(rhc_yTrain_accuracy)

    rhc_yTest_predict = rhc.predict(X_test)
    rhc_yTest_accuracy = accuracy_score(y_test, rhc_yTest_predict)
    rhc_validation_accuracy_learningCurve.append(rhc_yTest_accuracy)

    sa.fit(X_train, y_train)
    sa_yTrain_predict = sa.predict(X_train)
    sa_yTrain_accuracy = accuracy_score(y_train, sa_yTrain_predict)
    sa_train_accuracy_learningCurve.append(sa_yTrain_accuracy)

    sa_yTest_predict = sa.predict(X_test)
    sa_yTest_accuracy = accuracy_score(y_test, sa_yTest_predict)
    sa_validation_accuracy_learningCurve.append(sa_yTest_accuracy)

train_data = [1 - t for t in test_data]

plt.figure()
plt.plot(train_data, ga_train_accuracy_learningCurve, label='Training')
plt.plot(train_data, ga_validation_accuracy_learningCurve, label='Validation')
plt.grid()
plt.legend()
plt.xlabel('Training Examples (percent)')
plt.ylabel('Accuracy')
plt.title('Genetic Algorithm: Learning Curve')
plt.savefig('nn_ga_learningCurve.png')

plt.figure()
plt.plot(train_data, bp_train_accuracy_learningCurve, label='Training')
plt.plot(train_data, bp_validation_accuracy_learningCurve, label='Validation')
plt.grid()
plt.legend()
plt.xlabel('Training Examples (percent)')
plt.ylabel('Accuracy')
plt.title('Backpropagation: Learning Curve')
plt.savefig('nn_bp_learningCurve.png')

plt.figure()
plt.plot(train_data, sa_train_accuracy_learningCurve, label='Training')
plt.plot(train_data, sa_validation_accuracy_learningCurve, label='Validation')
plt.grid()
plt.legend()
plt.xlabel('Training Examples (percent)')
plt.ylabel('Accuracy')
plt.title('Simulated Annealing: Learning Curve')
plt.savefig('nn_sa_learningCurve.png')

plt.figure()
plt.plot(train_data, rhc_train_accuracy_learningCurve, label='Training')
plt.plot(train_data, rhc_validation_accuracy_learningCurve, label='Validation')
plt.grid()
plt.legend()
plt.xlabel('Training Examples (percent)')
plt.ylabel('Accuracy')
plt.title('Random Hill Climbing: Learning Curve')
plt.savefig('nn_rhc_learningCurve.png')
