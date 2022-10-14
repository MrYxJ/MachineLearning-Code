#!/usr/bin/env python
# encoding: utf-8

'''
@author: MrYx
@author github: https://github.com/MrYxJ
@file: UseDecisionTree.py
@time: 18-10-20 下午1:50
'''


from math import log
import operator
import PlotDecisionTree

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,0,'no']
    ]
    labels  = ['no surfaceing','flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算给定数据集的信息熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet :
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2) #信息熵定义公式

    return shannonEnt

def splitDataSet(dataSet, axis ,value):
    """
    按照给定的特征划分数据集
    :param dataSet:
    :param axis:
    :param value:
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] # 抽取从0到这个数之前的数
            reducedFeatVec.extend(featVec[axis+1: ]) #抽取从这个数之后一位的所有数t
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
       # print('featList:', featList)
        uniqueVals = set(featList)
       # print('uniqueVal:' ,uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy -newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    当数据集已经处理所有属性，或者所有属性值都相同时，选出结果出现最多作为叶子节点结果
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems() , key = operator.itemgetter(1), reverse = True)
    print('classcount：', sortedClassCount)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0])  == len(classList):  #当前所有样本都属于同一类
        return classList[0]

    if len(dataSet[0]) == 1:     # 当前属性集为空，只有一列结果
        return majorityCnt(classList)  # 返回出现结果频率最大作为结果
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = labels[:] # 在python 里面 list是引用类型的变量，所以防止改变用一个新的变量代替。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLables)
    return myTree

def classify(inputTree, featLabels, testVec):
    """
    使用决策树分类，传入一颗以dict字典形式建好的决策树和标签，测试数据，输出决策树分类的结果。
    :param inputTree:
    :param featLabels:
    :param testVec:
    :return:
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else : classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    '''
    使用pickle模块存储决策树
    :param inputTree:
    :param filename:
    :return:
    '''
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    '''
    使用pickle模块导入存储的决策树数据
    :param filename:
    :return:
    '''
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    print("Test begin:")
    myDat, labels = createDataSet()
    print(myDat)
    print(calcShannonEnt(myDat))
    print('best choice index:',chooseBestFeatureToSplit(myDat))
    myTree = createTree(myDat, labels)
    print(myTree)
    PlotDecisionTree.createPlot(myTree)
    print(classify(myTree,createDataSet()[1],myDat[2]))

