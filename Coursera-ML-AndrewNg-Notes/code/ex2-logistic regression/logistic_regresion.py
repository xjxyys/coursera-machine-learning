import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model

def scatter(data):
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def check_sigmoid():
    nums = np.arange(-10, 10, step=1)
    plt.figure(figsize=(12, 8))
    plt.plot(nums, sigmoid(nums), 'r')
    plt.show()

def cost(theta, X, y):
    first = np.dot(-y.T, np.log(sigmoid(np.dot(X, theta))))
    second = -np.dot((1 - y).T, np.log(1 - sigmoid(X.dot(theta))))
    # print(first.shape, second.shape)
    return np.sum(first + second) / X.shape[0]

def gradient(theta, X, y):
    theta = theta.reshape(-1, 1)
    features = len(theta)
    grad = np.zeros(features)
    # theta虽然是一维数组，可以当成3*1的矩阵直接相乘
    # X:(m, n) theta:(n, 1) y:(m, 1)
    # print(np.dot(X, theta).shape)
    error = sigmoid(np.dot(X, theta)) - y

    # print(error.shape)
    for i in range(features):
        # print(error[1:3],)
        term = np.dot(error.T, X[:, i])
        grad[i] = np.sum(term) / X.shape[0]
    return grad

def logistic_regression():
    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    print(data.head())

    scatter(data)
    check_sigmoid()

    data.insert(0, 'Ones', 1)

    features = data.shape[1] - 1

    X = data.iloc[:, 0:features]
    y = data.iloc[:, features:features+1]

    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(features)
    # 把theta成二维数组
    print(X.shape, y.shape, theta.shape)
    # print(np.dot(X, theta).shape)
    # print(sigmoid(np.dot(X, theta)).shape) 
    # print((sigmoid(np.dot(X, theta) - y).shape))

    print(cost(theta, X, y))
    print(gradient(theta, X, y))
    # x0:初始值，需要是一维数组
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
    # print(result)
    print('The cost after is:',cost(result[0], X, y))

    theta_min = np.array(result[0])
    predictions = predict(theta_min, X)
    correct = [1 if (a == b) else 0 for (a, b) in zip(predictions, y)]
    accuracy = sum(correct) / len(correct)
    print('accuracy = {0}%'.format(accuracy * 100))
    

def predict(theta, X):
    probability = sigmoid(np.dot(X, theta))
    return [1 if x >= 0.5 else 0 for x in probability]


# 正则化逻辑回归
def costReg(theta, X, y, learningRate):
    first = np.dot(-y.T, np.log(sigmoid(np.dot(X, theta))))
    second = -np.dot((1 - y).T, np.log(1 - sigmoid(X.dot(theta))))
    reg = (learningRate / (2 * X.shape[0])) * np.sum(np.power(theta[1:], 2))
    return np.sum(first + second) / X.shape[0] + reg

def gradientReg(theta, X, y, learningRate):
    theta = theta.reshape(-1, 1)
    features = len(theta)
    grad = np.zeros(features)
    # theta虽然是一维数组，可以当成3*1的矩阵直接相乘
    error = sigmoid(np.dot(X, theta)) - y
    for i in range(features):

        term = np.dot(error.T, X[:, i])
        if i == 0:
            grad[i] = np.sum(term) / X.shape[0]
        else:
            grad[i] = np.sum(term) / X.shape[0] + (learningRate / X.shape[0]) * theta[i]
    return grad

def scatterReg(data):
    positive = data[data['Accepted'].isin([1])]
    negative = data[data['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    plt.show()


def logistic_regression_Reg():
    path = 'ex2data2.txt'
    data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
    print(data.head())
    scatterReg(data)

    degree = 5
    x1 = data['Test 1']
    x2 = data['Test 2']
    data.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(0, i):
            data["F"+str(i)+str(j)] = np.power(x1, i-j) * np.power(x2, j)
    # 去掉原来的x1, x2
    data.drop('Test 1', axis=1, inplace=True)
    data.drop('Test 2', axis=1, inplace=True)

    print(data.head())  

    # set X (training data) and y (target variable)
    features = data.shape[1] - 1
    X = data.iloc[:, 1 : 1+features]
    y = data.iloc[:, 0 : 1]

    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(features)

    learningRate = 1
    costReg(theta, X, y, learningRate)
    gradientReg(theta, X, y, learningRate)

    result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, learningRate))
    print(result)
    print('The cost after is:',costReg(result[0], X, y, learningRate))

    theta_min = np.array(result[0])
    predictions = predict(theta_min, X)
    correct = [1 if (a == b) else 0 for (a, b) in zip(predictions, y)]
    accuracy = sum(correct) / len(correct)
    print('accuracy = {0}%'.format(accuracy * 100))

    model = linear_model.LogisticRegression(penalty='l2', C=1.0)
    model.fit(X, y.ravel())
    print(model.score(X, y))



if __name__ == '__main__':
    # logistic_regression()
    logistic_regression_Reg()
