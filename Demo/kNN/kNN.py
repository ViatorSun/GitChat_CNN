#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2018.
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :  kNN


from numpy import *


# k-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()        # 从小到大排序
    classCount={}          
    for i in range(k):                              # 计算距离最小的k 个点
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)    # 排序
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# 将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)               # 获取文件样本个数
    returnMat = zeros((numberOfLines,3))           # 创建NumPy “0”矩阵
    classLabelVector = []                          # 创建标签列表
    index = 0
    for line in arrayOLines:                       # 解析文本数据到列表
        line = line.strip()                        # 截取掉所有的回车字符
        listFromLine = line.split('\t')            # 使用tab字符“\t”将上一步得到的正好数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3]     # 将前3个元素存储到特征矩阵中
        if(listFromLine[-1].isdigit()):            # 将列表中的最后一列存储到标签向量中
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   # 特征值相除
    return normDataSet, ranges, minVals


# 图像转换为向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

