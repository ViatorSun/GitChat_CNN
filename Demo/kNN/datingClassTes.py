#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2018.
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :  测试集检验分类器算法


import kNN
from numpy import *


hoRatio = 0.50  # 测试集占整个数据集的比重
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
m = normMat.shape[0]
numTestVecs = int(m * hoRatio)
errorCount = 0.0
for i in range(numTestVecs):
    classifierResult = kNN.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
    print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
    if (classifierResult != datingLabels[i]): errorCount += 1.0
print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
print(errorCount)