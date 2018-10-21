#!/usr/bin/env python
# encoding: utf-8
'''
@author: MrYx
@contact: 529674242@qq.com
@author github: https://github.com/MrYxJ 
@file: TestDecisionTree.py
@time: 18-10-20 下午3:50
'''
import DecisionTree
import PlotDecisionTree
import re

def test_lenses():
     """
     测试 http://archive.ics.uci.edu/ml/machine-learning-databases/lenses 上隐形眼镜数据集
     数据格式需要处理一下再导入。
     :return:
     """
     lenses =[]
     dict = [{1:'young',2:'pre-presbyopic',3:'presbyopic'},
             {1:'myope',2:'hypermetrope'},{1:'no',2:'yes'},{1:'reduced',2:'normal'},
             {1:'hard',2:'soft',3:'not be fitted'}]

     fr = open("data/lenses.txt")
     for inst in fr.readlines():
          tmp = re.split(r'\s+',inst.strip())[1:]
          for index,value  in enumerate(dict):
               tmp[index] = value[int(tmp[index])]
          lenses.append(tmp)


     lenseLabels = ['age', 'prescript', 'astigmatic' , 'tearRate']
     lensesTree = DecisionTree.createTree(lenses, lenseLabels)
     PlotDecisionTree.createPlot(lensesTree)


if __name__ == '__main__':
     print("Test begin:")
     # myTree = DecisionTree.grabTree('data/example.txt')
     # print(myTree)
     test_lenses()
