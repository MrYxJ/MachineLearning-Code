# coding: utf-8
# linear_regression/test_temperature_normal.py
import regression as re
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

if __name__ == '__main__':
    srcX, y = re.loadDataSet('data/temperature.txt');

    m, n = srcX.shape
    srcX = np.concatenate((srcX[:, 0], np.power(srcX[:, 0], 2)), axis=1)
    # 特征缩放

    X = re.standardize(srcX.copy())
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    rate = 0.1
    maxLoop = 1000
    epsilon = 0.01

    result, timeConsumed = re.bgd(rate, maxLoop, epsilon, X, y)
    theta, errors, thetas = result

    # 打印特征点
    fittingFig = plt.figure()
    title = 'polynomial with bgd: rate=%.2f, maxLoop=%d, epsilon=%.3f \n time: %ds' % (rate, maxLoop, epsilon, timeConsumed)
    ax = fittingFig.add_subplot(111, title=title)

    trainingSet = ax.scatter(srcX[:, 0].flatten().A[0], y[:, 0].flatten().A[0])

    # 打印拟合曲线
    xx = np.linspace(50, 100, 50)
    xx2 = np.power(xx, 2)
    yHat = []
    for i in range(50):
        normalizedSize = (xx[i] - xx.mean()) / xx.std(0)
        normalizedSize2 = (xx2[i] - xx2.mean()) / xx2.std(0)
        x = np.matrix([[1, normalizedSize, normalizedSize2]])
        yHat.append(re.h(theta, x.T))

    fittingLine, = ax.plot(xx, yHat, color='g')

    ax.set_xlabel('Yield')
    ax.set_ylabel('temperature')

    plt.legend([trainingSet, fittingLine], ['Training Set', 'Polynomial Regression']) #显示图中标签
    plt.show()

    # 打印误差曲线
    errorsFig = plt.figure()
    ax = errorsFig.add_subplot(111)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    ax.plot(range(len(errors)), errors)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost J')
    plt.show()