#!/usr/bin/python3
# encoding: utf-8
# Author MrYx
# @Time: 2019/6/27 22:03

from NaiveBayes.bayes import setOfWords2Vec, createVocabList , trainNBO, classifyNB
import random ,os
from numpy import *

def textParse(bigString):
    '''
    Desc:
        接收一个大字符串并将其解析为字符串列表
    Args:
        bigString -- 大字符串
    Returns:
        去掉少于 2 个字符的字符串，并将所有字符串转换为小写，返回字符串列表
    '''
    import re
    # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    '''
    Desc:
        对贝叶斯垃圾邮件分类器进行自动化处理。
    Args:
        none
    Returns:
        对测试集中的每封邮件进行分类，若邮件分类错误，则错误数加 1，最后返回总的错误百分比。
    '''
    docList = []
    classList = []
    fullText = []
    end = 23
    for i in range(1, 26):
        # 切分，解析数据，并归类为 1 类别
        try:
        #print(os.getcwd() + '\email\ham\%d.txt')
            wordList = textParse(open(os.getcwd()+'\email\spam\%d.txt' % i).read())
            docList.append(wordList)
            classList.append(1)
            # 切分，解析数据，并归类为 0 类别
            #print(os.getcwd()+'\email\ham\%d.txt')
            wordList = textParse(open(os.getcwd()+'\email\ham\%d.txt' % i).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        except:
            end = i
            pass

    # 创建词汇表
    vocabList = createVocabList(docList)
    trainingSet = list(range(46))
    testSet = []
    # 随机取 10 个邮件用来测试
    for i in range(10):
        # random.uniform(x, y) 随机生成一个范围为 x ~ y 的实数
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    pVect, pStatus = trainNBO(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # print(classifyNB(array(wordVector), pVect, pStatus) )
        # print(classList[docIndex])
        # c = input('？？？')
        if classifyNB(array(wordVector), pVect, pStatus) != classList[docIndex]:
            errorCount += 1

    print('the errorCount is: ', errorCount)
    print('the testSet length is :', len(testSet))
    print('the error rate is :', float(errorCount)/len(testSet))

if __name__ == '__main__':
    spamTest()  # fuck 准确率好高
    # spamTest()
    # spamTest()
    # spamTest()
    # spamTest()
    # spamTest()
    # spamTest()
    # spamTest()
    # spamTest()
    # spamTest()
    # spamTest()
    # spamTest()