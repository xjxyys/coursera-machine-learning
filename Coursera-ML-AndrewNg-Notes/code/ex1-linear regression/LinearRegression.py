import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    parameters_num = int(theta.shape[1]) # 2
    m = X.shape[0] # m = 97代表训练的样本数
    cost = np.zeros(iters) # (1, iters)

    for i in range(iters):
        error = np.dot(X, theta.T) - y # (m, 1)
        for j in range(parameters_num):
            term = np.dot(error.T, X[:, j]) # (1, m) * (m, 1) = (1, 1)
            # temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            theta[0, j] = theta[0, j] - ((alpha / m) * np.sum(term))
            # print(temp)
            # theta[0, j] = temp # update theta
            # print(theta)
        # theta = temp
        cost[i] = computeCost(X, y, theta) # compute cost
    return theta, cost

def oneVariable_linearRegression():
    # set hyper parameters

    alpha = 0.01
    iters = 1000

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
    theta = np.array([[0., 0.]]) # (1, 2)

    # method 2 to convert to numpy matrix
    # X = np.matrix(X.values)
    # y = np.matrix(y.values)
    # theta = np.matrix(np.array([0,0]))

    # print(computeCost(X, y, theta))

    # perform gradient descent to fit the model parameters
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    print(g)

    print(computeCost(X, y, g))

    # plot the linear fit
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)

    # draw the line
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()

    # draw error
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


# 多元线性回归
def multipalVarieble_linearRegression():
    # set hyper parameters
    alpha = 0.01
    iters = 1000

    # Read data
    data_file = 'ex1data2.txt'
    data = pd.read_csv(data_file, header=None, names=['Size', 'Bedrooms', 'Price'])

    # feature normalization
    data = (data - data.mean()) / data.std()

    # add ones column
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1] # (m, n=cols-1)
    y = data.iloc[:, cols-1:cols] # (m, 1)

    # convert to numpy array
    X = np.array(X.values) # (m, n)
    y = np.array(y.values) # (m, 1)
    theta = np.array([[0., 0., 0.]]) # (1, 3)

    # perform gradient descent to fit the model parameters
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    print(g)

    print(computeCost(X, y, g))

    # draw error
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()

if __name__ == '__main__':
    oneVariable_linearRegression()
    multipalVarieble_linearRegression()
