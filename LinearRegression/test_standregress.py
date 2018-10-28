import numpy as np
import matplotlib as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import regression as re


if __name__ == '__main__':
    X, y = re.loadDataSet("data/ex1.txt")  # coursera的《machine learning》第二周实验数据
    m, n = X.shape
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    theta, timeConsumed = re.standRegres(X, y)
    print('消耗[%s] s \n 参数矩阵:\n %s' % (timeConsumed, theta))

    fittingFig = plt.figure()
    title = 'StandRegress  time: %s' % timeConsumed
    ax = fittingFig.add_subplot(111, title=title)
    trainingSet = ax.scatter(X[:, 1].flatten().A[0], y[:, 0].flatten().A[0])
    xCopy = X.copy()
    xCopy.sort(0)
    yHat = xCopy * theta
    fittingLine, = ax.plot(xCopy[:, 1], yHat, color='g')
    ax.set_xlabel('Population of City in 10,000s')
    ax.set_ylabel('Profit in $10,000s')
    plt.legend([trainingSet, fittingLine], ['Training Set', 'Linear Regression'])
    plt.show()

