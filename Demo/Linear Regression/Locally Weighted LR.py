#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2018.
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :  加权线性回归


import regression
from numpy import *
import matplotlib.pyplot as plt

xArr , yArr = regression.loadDataSet("ex0.txt")
xMat = mat(xArr)
yMat = mat(yArr)

yHat = regression.lwlrTest(xArr , xArr , yArr , 0.003)
xMat = mat(xArr)

srtInd = xMat[ : ,1].argsort(0)   # 对数据点按序排列
xSort = xMat[srtInd][:,0,:]

#绘制数据集散列点
fig = plt.figure()             #创建子图
ax = fig.add_subplot(1,1,1)    #添加一个（1,1,1）子图
x = xMat[:,1].flatten().A[0]
y = yMat.T[:,0].flatten().A[0]

ax.scatter( x , y , c = 'r',marker = 'o')
                                # 给子图添加散列图
                                # flatten()方法能将matrix的元素变成一维的



#描绘拟合曲线
xcopy = xMat.copy()
xcopy.sort(0)                    #为防止拟合数据点出现次序混乱，将数据点升序排列

ax.plot( xSort[:,1] , yHat[srtInd])
plt.title("hello")              # 设置标题
plt.show()








