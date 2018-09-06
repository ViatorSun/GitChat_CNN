#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2018.
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :  手写体识别


import kNN
from numpy import *
from os import listdir

# kNN.handwritingClassTest()

hwLabels = []
trainingFileList = listdir('trainingDigits')           # 加载训练数据
m = len(trainingFileList)
trainingMat = zeros((m,1024))

for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0]                 #take off .txt
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat[i,:] = kNN.img2vector('trainingDigits/%s' % fileNameStr)

testFileList = listdir('testDigits')                    #iterate through the test set
errorCount = 0.0
mTest = len(testFileList)

for i in range(mTest):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]                 #take off .txt
    classNumStr = int(fileStr.split('_')[0])
    vectorUnderTest = kNN.img2vector('testDigits/%s' % fileNameStr)
    classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
    print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
    if (classifierResult != classNumStr): errorCount += 1.0

print("\nthe total number of errors is: %d" % errorCount)
print("\nthe total error rate is: %f" % (errorCount/float(mTest)))