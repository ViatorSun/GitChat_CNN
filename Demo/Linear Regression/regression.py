#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2018.
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :  


from numpy import *

# 加载数据集，返回数据点与标签
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1  # 获取数据样本的个数
    dataMat = [];
    labelMat = []
    fr = open(fileName)

    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 计算回归函数
def standRegres(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    xTx = xMat.T * xMat

    # 计算矩阵的行列式，如果行列式为0，输出错误提示
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    # 求解权重核心代码
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 局部加权线性回归
# testPoint为输入，参数k控制衰减速度; 函数返回加权线性回归的预测值
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    m = shape(xMat)[0]               # 创建对角矩阵
    weights = mat(eye((m)))
    for j in range(m):               # 创建权重核，其中权重值大小以指数级衰减
        diffMat = testPoint - xMat[j, :]  #
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    # 求解权重核心代码
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


# 用于为数据集中每个点调用lwlr()，有助于求解参数 k 的大小
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat




