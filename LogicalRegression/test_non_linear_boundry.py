#coding:utf-8
#coder:MrYx

import numpy as np
import logical_regression as regression
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.preprocessing import PolynomialFeatures


if __name__ == '__main__':
    X, y = regression.loadDataSet('data/non_linear.txt')
    poly = PolynomialFeatures(6)
    XX = poly.fit_transform(X[:, 1:3])
    # for i in range(len(X)):
    #     print(X[i],'\t',XX[i])
    #     c = input('>>>')

    m, n = XX.shape
    options = [{
                   'rate': 1,
                   'epsilon': 0.01,
                   'theLambda': theLambda,

                   'maxLoop': 3000,

                   'method': 'bgd'
               } for theLambda in [0, 1.0, 100.0]]
    figures, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

    for idx, option in enumerate(options):
        result, timeConsumed = regression.gradient(XX, y, option)
        thetas, errors, iterationCount = result
        theta = thetas[-1]
        print("参数: [%s] 误差: [%s] 迭代次数: [%S]   " % (theta, errors[-1], iterationCount) )
        ax = axes[idx]
        # 绘制数据点
        title = '%s: rate=%.2f, iterationCount=%d, \n theLambda=%d, \n error=%.2f time: %.2fs' % (
            option['method'], option['rate'], iterationCount, option['theLambda'], errors[-1], timeConsumed)
        ax.set_title(title)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        for i in range(m):
            x = X[i].A[0]
            if y[i] == 1:
                ax.scatter(x[1], x[2], marker='*', color='black', s=50)
            else:
                ax.scatter(x[1], x[2], marker='o', color='green', s=50)

