import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    # 按照元素相乘multiply
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1) # (5000, 401)
    z2 = np.dot(a1, theta1) # (5000, 401) * (401, 25) = (5000, 25)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1) # (5000, 26)
    z3 = np.dot(a2, theta2) # (5000, 26) * (26, 10) = (5000, 10)
    h = sigmoid(z3) # (5000, 10)

    return a1, z2, a2, z3, h

def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    # m, n代表样本数和特征数
    m = X.shape[0]

    theta1 = params[:hidden_size * (input_size + 1)].reshape((input_size + 1), hidden_size) # (401, 25)
    theta2 = params[hidden_size * (input_size + 1):].reshape((hidden_size + 1), num_labels) # (26, 10)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 计算梯度
    J = 0
    delta1 = np.zeros(theta1.shape) # (401, 25)
    delta2 = np.zeros(theta2.shape) # (26, 10)

    first_term = np.multiply(-y, np.log(h)) # (5000, 10)
    second_term = np.multiply((1 - y), np.log(1 - h)) # (5000, 10)
    J = np.sum(first_term - second_term) / m # 标量
    return J

def cost_reg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    theta1 = params[:hidden_size * (input_size + 1)].reshape((input_size + 1), hidden_size) # (401, 25)
    theta2 = params[hidden_size * (input_size + 1):].reshape((hidden_size + 1), num_labels) # (26, 10)

    J = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)
    J += (float(learning_rate) / (2 * X.shape[0])) * (np.sum(np.square(theta1[1:,:])) + np.sum(np.square(theta2[1:,:])))
    return J

def get_theta(params, input_size, hidden_size, num_labels):
    theta1 = params[:hidden_size * (input_size + 1)].reshape((input_size + 1), hidden_size) # (401, 25)
    theta2 = params[hidden_size * (input_size + 1):].reshape((hidden_size + 1), num_labels) # (26, 10)
    # theta1 (401, 25) theta2 (26, 10)
    return theta1, theta2

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    theta1, theta2 = get_theta(params, input_size, hidden_size, num_labels)
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)
    z2 = np.insert(z2, 0, values=np.ones(m), axis=1) # (5000, 26)
    d3 = h - y # (5000, 10) 表示J对z3的偏导数
    # d3 (5000, 10) * theta2.T (10, 26) = (5000, 26) * sigmoid_gradient(z2) (5000, 26) = (5000, 26
    d2 = np.multiply(np.dot(d3, theta2.T), sigmoid_gradient(z2)) # (5000, 26) 表示J对z2的偏导数

    # 此处d2[:,1:]是为了去掉偏置项，因为偏置项不参与反向传播
    delta1 = np.dot(a1.T, d2[:,1:]) / m # (401, 5000) * (5000, 25) = (401, 26)
    delta2 = np.dot(a2.T, d3) / m# (26, 5000) * (5000, 10) = (26, 10)

    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2))) # (10285, )
    return J, grad

def backprop_reg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    theta1, theta2 = get_theta(params, input_size, hidden_size, num_labels)
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = cost_reg(params, input_size, hidden_size, num_labels, X, y, learning_rate)
    z2 = np.insert(z2, 0, values=np.ones(m), axis=1) # (5000, 26)
    d3 = h - y # (5000, 10) 表示J对z3的偏导数
    # d3 (5000, 10) * theta2.T (10, 26) = (5000, 26) * sigmoid_gradient(z2) (5000, 26) = (5000, 26)
    d2 = np.multiply(np.dot(d3, theta2.T), sigmoid_gradient(z2)) # (5000, 26) 表示J对z2的偏导数
    delta1 = np.dot(a1.T, d2[:,1:]) / m # (401, 5000) * (5000, 26) = (401, 26)
    delta2 = np.dot(a2.T, d3) / m# (26, 5000) * (5000, 10) = (26, 10)

    delta1[1:, :] = delta1[1:, :] + (theta1[1:, :] * learning_rate) / m
    delta2[1:, :] = delta2[1:, :] + (theta2[1:, :] * learning_rate) / m

    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2))) # (10285, )
    return J, grad


def ex4_simple():
    data = loadmat('ex4data1.mat')
    X = data['X']
    y = data['y'] # X:(5000, 400), y:(5000, 1)


    # 把y标签进行一次one-hot编码
    encoder = OneHotEncoder(sparse=False) # 不产生稀疏矩阵
    y_onehot = encoder.fit_transform(y) # y_onehot:(5000, 10)
    
    input_size = 400
    hidden_size = 25
    num_labels = 10
    learning_rate = 1

    # 随机初始化params
    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
    m = X.shape[0]

    print('the cost does not contain regularization is:',cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate))
    print('the cost contain regularization is:',cost_reg(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate))

    J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

    J_reg, grad_reg = backprop_reg(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
    
    fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                    method='TNC', jac=True, options={'maxiter': 250})
    fmin_reg = minimize(fun=backprop_reg, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                    method='TNC', jac=True, options={'maxiter': 250})
    
    theta_1, theta_2 = get_theta(fmin.x, input_size, hidden_size, num_labels)
    theta_1_reg, theta_2_reg = get_theta(fmin_reg.x, input_size, hidden_size, num_labels)

    a1, z2, a2, z3, h = forward_propagate(X, theta_1, theta_2)
    a1_reg, z2_reg, a2_reg, z3_reg, h_reg = forward_propagate(X, theta_1_reg, theta_2_reg)

    # argmax返回沿轴axis最大值的索引
    y_pred = np.argmax(h, axis=1) + 1 # (5000, )
    y_pred_reg = np.argmax(h_reg, axis=1) + 1 # (5000, )
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    correct_reg = [1 if a == b else 0 for (a, b) in zip(y_pred_reg, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    accuracy_reg = (sum(map(int, correct_reg)) / float(len(correct_reg)))
    print('accuracy without regularization is:', accuracy)
    print('accuracy with regularization is:', accuracy_reg)

if __name__ == '__main__':
    ex4_simple()