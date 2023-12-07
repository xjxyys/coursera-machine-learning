import pandas as pd
import numpy as np

# contruct cost function
def computeCost(X, y, theta):
    # mathod 1 array
    inner = np.power((np.dot(X, theta.T) - y), 2)
    # method 2 matrix
    # inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# batch gradient descent
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.zeros(theta.shape) # (1, 2)
    parameters = int(theta.shape[1]) # 2
    cost = np.zeros(iters) # (1, iters)

    for i in range(iters):


# Read data
data_file = 'ex1data1.txt'
data = pd.read_csv(data_file, header=None, names=['Population', 'Profit'])

data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1] # (m, n=cols-1)
y = data.iloc[:, cols-1:cols] # (m, 1)

# method 1 to convert to numpy array
X = np.array(X.values) # (m, n)
y = np.array(y.values) # (m, 1)
theta = np.array([[0, 0]]) # (1, 2)

# method 2 to convert to numpy matrix
# X = np.matrix(X.values)
# y = np.matrix(y.values)
# theta = np.matrix(np.array([0,0]))

# print(computeCost(X, y, theta))