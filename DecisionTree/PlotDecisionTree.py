#!/usr/bin/env python
# encoding: utf-8

'''
@author: MrYx
@contact: 529674242@qq.com
@author github: https://github.com/MrYxJ
@file: UseDecisionTree.py
@time: 18-10-20 下午2:21
'''

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth" ,fc = "1.8")
leafNode = dict(boxstyle = "round4" , fc = "1.8")
arrow_args = dict(arrowstyle = "<-")

def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点中填充文本信息
    :param cntrPt:
    :param parentPt:
    :param txtString:
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    """
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    """
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yoff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0/ plotTree.totalD

def plotNode(nodeTxt , centerPt , parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,
                            xy = parentPt,
                            xycoords = 'axes fraction',
                            xytext = centerPt,
                            textcoords = 'axes fraction',
                            va="center",
                            ha = "center" ,
                            bbox = nodeType ,
                            arrowprops = arrow_args)

def getNumLeafs(myTree):
    """
    获取叶子节点个数
    :param myTree:
    :return:
    myTree格式 ：嵌套的字典
    eg : {'no surrfaceing:':{0 : 'no',1 : {'flippers':{0: 'no',1:'yes'} } }
    """

    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys() :
        if type(secondDict[key]).__name__ == 'dict' : # 测试节点类型是否为字典
            numLeafs += getNumLeafs(secondDict[key])
        else :
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    获取叶子层数
    :param myTree:
    :return:
    """
    maxDepth = 1
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            maxDepth = max(maxDepth, getTreeDepth(secondDict[key]) + 1)

    return maxDepth

def retieveTree(i):
    """
    存储两课决策树，方便测试使用。
    :param i:
    :return:
    """
    listOfTrees = [{ 'no surfacing':{0: 'no', 1:{'flippers':{0:'no', 1:'yes',2:'no','flag':'yes'}}}},
                     {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}
    }]
    return listOfTrees[i]


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5 / plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree, (0.5, 1.0), '')  # 树的引用作为父节点，但不画出来，所以用''
    plt.show()


if __name__ == '__main__':
    #createPlot()
    myTree = retieveTree(0)
    print(myTree)
    createPlot(myTree)

