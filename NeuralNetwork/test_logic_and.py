#!/usr/bin/env python
#coding:utf-8
'''
@author : MrYx
@github: https://github.com/MrYxJ
@file: test_logic_and.py
@time: 2018/12/1 17:47
'''
import numpy as np
import coursera.NeuralNetwork.neural_network as nn

def test():
    data = np.mat([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [1,1,1]
    ])

    X = data[:,0:2]
    Y = data[:,2]
    res = nn.train(X, Y, hiddenNum=0, alpha=10, maxIters=5000, precision=0.01)

if __name__ == '__main__':
     test()
