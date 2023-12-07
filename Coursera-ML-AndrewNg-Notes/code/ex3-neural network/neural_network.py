import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
# 启示：存成matrix可能运算速度更快，或者是因为矩阵乘法比元素乘法快
# 事实证明，矩阵np.dot或者转成matrix类后直接*，比原始的按照元素相乘快的多,而且按照元素相乘可能出现除零错误和float溢出
# 但是，如果是矩阵相乘，np.dot和*速度差不多,但是np.dot更快一点,但可能导致精度差了点
# 最后，原版的代码精度可以达到94.48%，但我修改过后只能达到94.16%
# 注意！！multiply也是按照元素相乘！
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 优化版本，原本是按元素相乘，使用*
def cost(theta, X, y, learningRate):
    # m 代表样本数量
    m = X.shape[0]
    # first = -y * np.log(sigmoid(X @ theta))
    # second = -(1 - y) * np.log(1 - sigmoid(X @ theta))
    first = np.dot(-y.T, np.log(sigmoid(np.dot(X, theta))))
    second = -np.dot((1 - y).T, np.log(1 - sigmoid(X.dot(theta))))
    # reg = (learningRate / (2 * m)) * np.sum(theta[1:] ** 2)
    reg = (learningRate / (2 * m)) * np.sum(np.power(theta[1:], 2))
    return np.sum(first + second) / m + reg

# def cost(theta, X, y, learningRate):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
#     second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
#     reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
#     return np.sum(first - second) / len(X) + reg

def gradient_with_loop(theta, X, y, learningRate):
    # n 代表特征数量, m 代表样本数量
    n, m= X.shape[1], X.shape[0]
    grad = np.zeros(n)
    # theta 必须是二维的，才能进行矩阵运算
    error = sigmoid(X @ theta) - y
    
    for i in range(n):
        term = error * X[:, i]
        if i == 0:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = np.sum(term) / len(X) + (learningRate / m) * theta[i]
    return grad

def gradient(theta, X, y, learningRate):
    # n 代表特征数量, m 代表样本数量
    n, m= X.shape[1], X.shape[0]
    theta = theta.reshape(-1, 1)
    # error = sigmoid(X @ theta) - y
    error = sigmoid(np.dot(X, theta)) - y
    # (n, m) @ (m, 1) + (n, 1) -> (n, 1) + (n, 1)-> (n, 1)
    # grad = (X.T @ error) / m + (learningRate / m) * theta  
    grad = np.dot(X.T, error) / m + (learningRate / m) * theta
    # 正则化不对 theta0 进行惩罚
    grad[0, 0] = np.sum(np.dot(X[:,0].T, error)) / m 
    # print(grad.shape)
    # 展平数组, flatten()是获得副本，ravel()是获得视图，会影响原始的grad
    return grad.flatten()
    # return np.array(grad).ravel()

# def gradient(theta, X, y, learningRate):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
    
#     parameters = int(theta.ravel().shape[1])
#     error = sigmoid(X * theta.T) - y
    
#     grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    
#     # intercept gradient is not regularized
#     grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    
#     return np.array(grad).ravel()

def one_vs_all(X, y, num_labels, learning_rate):
    # m, n代表样本数和特征数
    m, n = X.shape[0], X.shape[1]
    # (n + 1) * k array to store parameters
    all_theta = np.zeros(((n + 1), num_labels))
    # a way of adding new column
    X = np.insert(X, 0, values=np.ones(m), axis=1)
    # 标签从1开始，而不是从0开始
    for i in range(1, num_labels+1):
        theta = np.zeros(n + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = y_i.reshape(-1, 1)

        # 把目标函数最小化,注意x0是参数的初值，必须是一维的
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate),method='TNC', jac=gradient)
        all_theta[:, i-1] = fmin.x
        print(fmin.x.shape)
    return all_theta

def predict_all(X, all_theta):
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
    h = sigmoid(X @ all_theta)

    # argmax 返回最大值的索引
    h_argmax = np.argmax(h, axis=1)
    # 因为标签是从1开始的，所以需要加1
    h_argmax = h_argmax + 1
    return h_argmax

def ex3():
    data = loadmat('ex3data1.mat')

    # (5000, 400) (5000, 1)
    print(data['X'].shape, data['y'].shape)
    print(np.unique(data['y']))
    all_theta = one_vs_all(data['X'], data['y'], 10, 1)
    y_pred = predict_all(data['X'], all_theta)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
    accuracy = sum(correct) / len(correct)
    print('accuracy = {0}%'.format(accuracy * 100))

if __name__ == '__main__':
    ex3()
    