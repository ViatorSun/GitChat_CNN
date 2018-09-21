
#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2018.
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @GitHub ： https://github.com/ViatorSun
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :  误差反向传播算法实现数字识别

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

class NeuralNetwork:
    def __init__(self,layers):#(64,100,10)
        #权值的初始化，范围-1到1
        self.V = np.random.random((layers[0]+1,layers[1]+1))*2-1
        self.W = np.random.random((layers[1]+1,layers[2]))*2-1
        
    def train(self,X,y,lr=0.11,epochs=10000):
        #添加偏置
        temp = np.ones([X.shape[0],X.shape[1]+1])
        temp[:,0:-1] = X
        X = temp
        
        for n in range(epochs+1):
            i = np.random.randint(X.shape[0]) #随机选取一个数据
            x = [X[i]]
            x = np.atleast_2d(x)#转为2维数据
            
            L1 = sigmoid(np.dot(x,self.V))#隐层输出
            L2 = sigmoid(np.dot(L1,self.W))#输出层输出
            
            L2_delta = (y[i]-L2)*dsigmoid(L2)
            L1_delta= L2_delta.dot(self.W.T)*dsigmoid(L1)
            
            self.W += lr*L1.T.dot(L2_delta)
            self.V += lr*x.T.dot(L1_delta)
            
            #每训练1000次预测一次准确率
            if n%1000==0:
                predictions = []
                for j in range(X_test.shape[0]):
                    o = self.predict(X_test[j])
                    predictions.append(np.argmax(o))#获取预测结果
                accuracy = np.mean(np.equal(predictions,y_test))
                print('epoch:',n,'accuracy:',accuracy)
        
    def predict(self,x):
        #添加偏置
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        x = temp
        x = np.atleast_2d(x)#转为2维数据

        L1 = sigmoid(np.dot(x,self.V))#隐层输出
        L2 = sigmoid(np.dot(L1,self.W))#输出层输出
        return L2

# 载入数据
digits = load_digits()
X = digits.data#数据
y = digits.target#标签

# 输入数据归一化
X -= X.min()
X /= X.max()

# 创建网络
nm = NeuralNetwork([64,100,10]

# 分割数据1/4为测试数据，3/4为训练数据
X_train,X_test,y_train,y_test = train_test_split(X,y)

# 标签二值化
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

print('start')

nm.train(X_train,labels_train,epochs=20000)

print('end')




