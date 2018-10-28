import numpy as np
import matplotlib as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time

def exeTime(func):
    """
    计算函数消耗时间的装饰器
    :param func:
    :return:
    """
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        t1 = time.time()
        return back, t1 - t0
    return newFunc

def loadDataSet(filename):
    """
    读取数据

    在coursera《Machine Learning》里面数据格式如下:
     "feature1 Tab feature2 Tab ..... label "

    :param filename: 文件名
    :return:
     x:训练样本集矩阵
     y:标签集矩阵

    """
    numFeat = len(open(filename).readline().split('\t')) - 1
    X = []
    y = []
    file = open(filename)
    for line in file.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        X.append(lineArr)
        y.append(float(curLine[-1]))
    return np.mat(X), np.mat(y).T

def h(theta, x):
    '''
    预测函数
    :param theta:相关系数矩阵
    :param x: 特征向量
    :return: 预测结果
    '''
    return (theta.T * x)[0, 0] # 这里结果是一个两层List嵌套的二维矩阵,所以取用[0,0]只需要返回数组里数值

def J(theta, X, y):
    '''
    代价函数
    :param theta:相关系数矩阵
    :param x: 样本集矩阵
    :param y: 标签矩阵
    :return:  预测
    '''
    m = len(X)
    return ((X * theta - y).T * (X * theta - y))[0, 0] / (2 * m) # [0, 0]原因同上

@exeTime
def bgd(rate, maxLoop, epsilon, X, y):
    """
    批量梯度下降
    :param rate: 学习率
    :param maxLoop: 最大迭代次数
    :param epsilon: 收敛精度
    :param X: 样本矩阵
    :param y: 标签矩阵
    :return: theta: 结果参数 ，errors： 每一次误差的list，thetas ：每一轮迭代参数,二位数组
    """
    m, n = X.shape
    theta = np.zeros([n,1]) #系数矩阵初始化为n行1列全为0的矩阵

    count = 0  #记载迭代次数
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(n):
        thetas[j] = [theta[j,0]]
    while count <= maxLoop:
        if(converged) :   break
        count = count + 1
        for j in range(n) :
            deriv = (y - X*theta).T * X[:, j] / m
            theta[j, 0] = theta[j, 0] + rate * deriv
            thetas[j].append(theta[j, 0])

        # 向量化的写法
        """
        derivE = y - X* theta
        theta = theta + X.T * derivE * rate * (1.0/ m)  # 向量化后的更新方式
        thetas.append(theta)
        """

        error = J(theta, X, y)
        errors.append(error)
        if(error <= epsilon):  converged = True
    return theta, errors, thetas

@exeTime
def sgd(rate, maxLoop, epsilon, X, y):
    """
    随机梯度下降
    :param rate:
    :param maxLoop:
    :param epsilon:
    :param X:
    :param y:
    :return:
    """
    m, n = X.shape
    theta = np.zeros([n,1])
    count = 0  # 迭代次数
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(n):
        thetas[j] = [theta[j, 0]]
    while count <= maxLoop:
        if(converged) : break
        count = count +1
        for i in range(m):
            diff = y[i,0] - X[i,:] * theta
            for j in range(n):
               theta[j, 0] = theta[j, 0] + rate * diff * X[i,j]
               thetas[j].append(theta[j, 0])
            error = J(theta, X, y)
            errors.append(error)
            if (error <= epsilon): converged = True
    return theta, errors, thetas

@exeTime
def standRegres(X, y):
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0 :
        print('This matrix is singular,cannot do inverse')
        return
    ws = xTx.I * (X.T * y)
    return ws

def standardize(X):
    """
    特征标准化处理
    :param X:
    :return:
    """
    m , n = X.shape
    for j in range(n):
        features = X[:, j]
        meanVal = features.mean(axis = 0)
        std = features.std(axis = 0)
        if(std!= 0):
            X[:,j] = (X[:,j] - meanVal) / std
        else :
            X[:,j] = 0
    return X

def normalize(X):
    """特征归一化处理
    Args:
        X 样本集
    Returns:
        归一化后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff != 0:
           X[:,j] = (features-minVal)/diff
        else:
           X[:,j] = 0
    return X

