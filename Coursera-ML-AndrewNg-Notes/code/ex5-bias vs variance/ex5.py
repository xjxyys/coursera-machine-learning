import scipy.io as sio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt

def load_data():
    """
    for ex5 d['X'] shape = (12, 1)
    pandas has trouble taking this 2d ndarray to construct a dataframe, so I ravel
    the results
    """
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])

def cost(theta, X, y):
    """
    X: R(m * n), m records, n features
    y: R(m)
    theta: R(n), linear regression parameters
    """
    m = X.shape[0]
    inner = X @ theta - y # R(m)
    print(inner.shape)
    square_sum = inner.T @ inner

    cost = square_sum / (2 * m)
    return cost

def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y) # R(n)
    return inner / m

def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()
    regularized_term[0] = 0 # don't regularize intercept theta
    regularized_term = (l / m) * regularized_term

    return gradient(theta, X, y) + regularized_term

def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = (l / (2 * m)) * np.square(theta[1:]).sum()

    return cost(theta, X, y) + regularized_term

def linear_regression_np(X, y, l=1):
    """
    linear regression
    args:
        X: feature matrix, (m, n) with incercept x0 = 1
        y: target vector, (m, )
        l: lambda constant for regularization
    
    return: trained parameters
    """
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0 = theta,
                       args=(X, y, l),
                       method='TNC',
                       jac = regularized_gradient,
                    #    options={'disp':True})
                       options={'disp':False})
    return res

def poly_features(x, power, as_ndarry=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power+1)}
    df = pd.DataFrame(data)

    return df.to_numpy() if as_ndarry else df

# 将特征归一化
def normalize_feature(df):
    """
    Applies function along input axis(default 0) of DataFrame.
    """
    return df.apply(lambda column:(column - column.mean()) / column.std())

def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
    return: 
    """
    def prepare(x):
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_feature(df).to_numpy()
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)
    return [prepare(x) for x in args]

# 画学习曲线此时的误差只考虑均方误差，不考虑惩罚项
def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m+1):
        res = linear_regression_np(X[:i,:], y[:i], l=l)
        # tc表示training_cost cv表示crossvalid_cost
        tc = cost(res.x, X[:i,:], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1,m+1), training_cost, label='training cost')
    plt.plot(np.arange(1,m+1), cv_cost, label='cv cost')
    plt.legend(loc=1)

def bias():
    X, y, Xval, yval, Xtest, ytest = load_data()
    df = pd.DataFrame({'water_level':X, 'flow':y})
    sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=7)
    plt.show()
    # 把X,y,Xval,yval,Xtest,ytest都变成m*1的矩阵
    X, Xval, Xtest = [np.insert(x.reshape(x.shape[0],1),0,np.ones(x.shape[0]),axis=1) for x in (X, Xval, Xtest)]
    
    theta = np.ones(X.shape[1])
    print(cost(theta, X, y))
    print(gradient(theta, X, y))
    print(regularized_gradient(theta, X, y))

    # 拟合数据 lambda = 0
    # get('x')即返回最优的参数
    final_theta = linear_regression_np(X, y, l=1).get('x')

    b = final_theta[0] # intercept
    a = final_theta[1] # slope

    # 画出拟合曲线
    plt.scatter(X[:,1], y, label="Training data")
    plt.plot(X[:,1], X[:,1]*a + b, label="Prediction")
    plt.legend(loc=2)
    plt.show()


    # 画出学习曲线
    training_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m+1):
        res = linear_regression_np(X[:i, :], y[:i], l=0)
        
        tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
        cv = regularized_cost(res.x, Xval, yval, l=0)
    #     print('tc={}, cv={}'.format(tc, cv))
    
        training_cost.append(tc)
        cv_cost.append(cv)
    plt.plot(np.arange(1, m+1), training_cost, label='training cost')
    plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
    plt.legend(loc=1)
    plt.show()
    # 欠拟合了


def variance():
    X, y, Xval, yval, Xtest, ytest = load_data()
    print(poly_features(X, power=3).head())

    # 准备8阶多项式回归, 不设置惩罚
    X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
    plt.show()
    # 此时过拟合了


    # 准备8阶多项式回归, 设置惩罚lambda=1
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=1)
    plt.show()
    # 此时training cost 不再一直是0,过拟合减轻了

    # lambda设置成100，
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)
    plt.show()
    # 此时由于惩罚太重，变成欠拟合了

    # 寻找最佳lambda
    l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 3, 10]
    training_cost, cv_cost = [], []
    
    # 对每个lambda都画出training cost和cv cost
    for l in l_candidate:
        res = linear_regression_np(X_poly, y, l)
        tc = cost(res.x, X_poly, y)
        cv = cost(res.x, Xval_poly, yval)
        training_cost.append(tc)
        cv_cost.append(cv)
    
    plt.plot(l_candidate, training_cost, label='training')
    plt.plot(l_candidate, cv_cost, label='cross validation')
    plt.legend(loc=2)

    plt.xlabel('lambda')
    plt.ylabel('cost')
    plt.show()

    # 从验证集中找出最小的cv_cost
    best_l = l_candidate[np.argmin(cv_cost)]
    print('best lambda in cv = {}'.format(best_l))
    
    for l in l_candidate:
        theta = linear_regression_np(X_poly, y, l).get('x')
        print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))


if __name__ == '__main__':
    bias()
    variance()

