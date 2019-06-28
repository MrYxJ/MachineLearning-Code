#!/usr/bin/python3
# encoding: utf-8
# Author MrYx
# @Time: 2019/6/12 20:38
from numpy import *

def loadDataSet():
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    """
    获取所有单词的集合
    :param dataSet: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocabSet = set()  # create empty set
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList) #[0,0......]
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    """
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNBO(trainMatrix, trainCategory):
    """
    训练数据优化版本
    :param trainMatrix: 文件单词矩阵
    :param trainCategory: 文件对应的类别
    :return:
    """
    # 总文件数
    numTrainDocs = len(trainMatrix)
    # 总单词数
    numWords = len(trainMatrix[0])

    #统计结果总共有多少种分类
    Category = set()
    for cat in trainCategory :
        Category.add(cat)

    #存不同结果ci的 P(ci)
    pStatus = list(ones(len(Category)))
    #print(pStatus)
    for item in Category:
        cnt = 0
        for train in trainCategory:
            if train == item:
                 cnt += 1
        pStatus[item] = cnt / float(numTrainDocs)
    # 构造单词出现次数列表
    pNum = []     # pNum[ci] 统计输出结果为ci的输入数据里各个特征出现值之和 ,由于如果出现次数为0，相乘都是0，所以修改版用1初始化
    pDenom = []   # 整个数据集单词出现总数，2.0根据样本/实际调查结果调整分母的值（2主要是避免分母为0，同时分子为，当然值可以调整）
    for cat in Category:
        pNum.append(ones(numWords))
        pDenom.append(2.0)

    for i in range(numTrainDocs):
        for cat in Category:
            if trainCategory[i] == cat:
                pNum[cat] += trainMatrix[i]         # 统计结果cat的i位置特征的值之和，是一个数组
                pDenom[cat] += sum(trainMatrix[i])  # 统计结果cat的所有位置特征值之和，是一个值

    pVect = list(ones(len(Category)))
    for cat in Category:
       pVect[cat] = log(pNum[cat] / pDenom[cat])  # 后面累乘运算可能小数位数过多，结果会溢出，所以取Log，变成累加。
    return pVect, pStatus

def classifyNB(vec2Classify,pVect ,pStatus):
    """
    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """
    ans = float('-Inf')
    ansindex = 0
    for index in range(len(pStatus)):
        p = sum(vec2Classify * pVect[index]) + log(pStatus[index]) # log(P(w|c1) * P(c1))
        if p > ans:
            ans = p
            ansindex = index
    return ansindex

def testingNB():
    """
    :return:
    """
    wordata , label = loadDataSet()
    Vocablist = createVocabList(wordata)
    trainMat = []
    for train in wordata:
        trainMat.append(setOfWords2Vec(Vocablist, train))

    pV, pStatus = trainNBO(array(trainMat), array(label))
    testEntry = ['love' , 'my', 'ate']
    #print(array(setOfWords2Vec(Vocablist, testEntry)))

    thisDoc  = array(setOfWords2Vec(Vocablist, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, pV, pStatus))
    testEntry = ['yexiaoju','is', 'stupid']
    thisDoc = array(setOfWords2Vec(Vocablist, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, pV, pStatus))

def textParse(bigString):
    pass

if __name__ == '__main__':
    testingNB()