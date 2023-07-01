import random
from numpy import random
import numpy as np
import matplotlib.pyplot as plt

seed = 5
np.random.seed(seed)
total_rows = 10
total_columns = 10
X = np.random.random(size=(total_rows, total_columns)) * 200 - 100
Y = np.arange(total_rows)
plt.scatter(X[:,0], Y)
plt.show()
print("done")
    # return X, Y
