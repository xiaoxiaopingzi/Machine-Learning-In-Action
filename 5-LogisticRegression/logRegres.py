# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: logRegres.py 
@time: 2018-04-24 15:45  
"""
import numpy as np
import matplotlib.pyplot as plt


# 从文本文件中装载数据集
def loadDataset():
    mat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        mat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return mat, labelMat


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 批量梯度上升算法
def gradAscent(dataMat, classLabels):
    dataMatrix = np.mat(dataMat)  # np.mat()表示numpy中的矩阵
    labelMat = np.mat(classLabels).T
    maxCycles = 500
    alpha = 0.001
    n = dataMatrix.shape[1]
    weights = np.ones((n, 1))
    for i in range(maxCycles):
        h = sigmoid(np.dot(dataMatrix, weights))
        error = labelMat - h
        weights = weights + alpha * np.dot(dataMatrix.T, error)
    return weights


# 随机梯度上升算法
def stoGradAscent(dataMat, classLabels):
    dataMatrix = np.array(dataMat)  # np.array()表示numpy中的数组
    alpha = 0.001
    m, n = dataMatrix.shape
    weights = np.ones(n)  # weights为numpy中的数组
    for i in range(m):
        h = sigmoid(np.sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 随机梯度上升法的改进版本
def stoGradAscent1(dataMat, classLabels, numIter=150):
    dataMatrix = np.array(dataMat)  # np.array()表示numpy中的数组
    m, n = dataMatrix.shape
    weights = np.ones(n)  # weights为numpy中的数组
    for j in range(numIter):  # 循环numIter次
        dataIndex = [i for i in range(m)]
        for i in range(m):
            # 随着迭代次数的增加，学习率不断减少，以便能够更好的收敛
            alpha = 4 / (1.0 + j + i) + 0.01
            # 在所有的样本中随机选取一个样本进行随机梯度下降
            randIndex = np.random.randint(0, len(dataIndex))
            h = sigmoid(np.sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    dataArr, labelMat = loadDataset()
    dataArr = np.mat(dataArr)
    m = dataArr.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-4.0, 4.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(x, weights):
    h = sigmoid(np.sum(x * weights))
    if h > 0.5:
        return 1
    else:
        return 0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(currLine[21]))
    weights = stoGradAscent1(trainingSet, trainingLabel, 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        result = classifyVector(lineArr, weights)
        if result != int(currLine[21]):
            errorCount += 1
    errorRate = errorCount / numTestVec
    print('本次逻辑回归的错误率为：{} %'.format(errorRate * 100))
    return errorRate


def mulTest():
    numTest = 10
    errorSum = 0.0
    for i in range(numTest):
        errorSum += colicTest()
    print('在经过{}次的测试后，平均的错误率为：{} %'.format(numTest, errorSum / numTest))


if __name__ == "__main__":
    # dataArr, labelMat = loadDataset()
    # weights = gradAscent(dataArr, labelMat)
    # weights = stoGradAscent(dataArr, labelMat)
    # weights = stoGradAscent1(dataArr, labelMat)
    # plotBestFit(weights)
    mulTest()
