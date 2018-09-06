#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2018.
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :  约会网站预测函数


import kNN
from numpy import *


resultList = ['not at all', 'in small doses', 'in large doses']
percentTats = float(input("玩游戏的时间百分比?"))
ffMiles = float(input("每年飞行里程数?"))
iceCream = float(input("每年消耗的冰淇淋?"))
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
inArr = array([ffMiles, percentTats, iceCream, ])
classifierResult = kNN.classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
print("你可能会喜欢这个人: %s" % resultList[classifierResult - 1])